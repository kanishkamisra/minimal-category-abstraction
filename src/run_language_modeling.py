# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple, Optional, Any
from collections.abc import Iterable
import itertools
import csv

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from cat_abs_utils import encode_text, extract_representation
from minicons.scorer import MaskedLMScorer, IncrementalLMScorer

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextDataset,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    AutoModelForPreTraining,
    BatchEncoding,
    default_data_collator,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


class BestValues:
    def __init__(self, value=-1):
        self.value = value
        self.ids = []

    def update(self, value, value_id):
        if self.value < value:
            self.value = value
            self.ids = [value_id]
        elif self.value == value:
            self.ids.append(value_id)


# initialize evaluation states
best_dev_diff = BestValues()
best_dev_ident = BestValues()


def update_evaluation_results(results, i):
    global best_dev_diff, best_dev_ident
    best_dev_diff.update(results['all_diff'], i)
    best_dev_ident.update(results['all_ident'], i)


class Stat:
    def __init__(self):
        self.n = 0
        self.s = 0

    def add(self, x):
        self.n += 1
        self.s += x

    def update(self, xs):
        for x in xs:
            self.add(x)

    def mean(self):
        return self.s / self.n


def replace_element(element, repl: list, seq: list, return_idxes=False):
    """Replace all occurrences of element in seq by repl
    """
    start_idx = 0
    idxes = []
    while True:
        try:
            element_idx = seq.index(element, start_idx)
        except ValueError:
            break
        idxes.append(element_idx)
        new_seq = seq[:element_idx] + repl
        start_idx = len(new_seq)
        new_seq += seq[element_idx+1:]
        seq = new_seq
    if return_idxes:
        return seq, idxes
    else:
        return seq


def replace_element_in_seqs(element, repl, seqs, return_idxes=False):
    """Replace all occurrences of element in seqs by repl
    Input:
        seqs: list of sequences
    Returns:
        new_seqs: new seqs after replacement
        idxes (optional): list of list of indexes where the repls start
    """
    rets = [
        replace_element(element, repl, seq, return_idxes=return_idxes)
        for seq in seqs]
    if return_idxes:
        return tuple(zip(*rets))
    else:
        return rets


def read_lines(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if line and not line.isspace():
                yield line


class MaskReplacedTextDataset(Dataset):
    """
    A Dataset for text with each occurrence of mask replaced by a designated str.
    Text is encoded by a tokenizer.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        raw_texts: Iterable,
        replace_strs: Optional[Iterable] = None,
        mask: Optional[str] = "[MASK]",
        prepend_space: Optional[str] = None,
        padding: Optional[str] = "longest",
        return_tensors='pt',
        return_attention_mask=True,
        **kwargs
    ):
        """
        tokenizer: The tokenizer used to tokenize text.
        raw_texts: Raw texts containing masks. An Iterable of str.
        replace_strs: If provided, it should be an Iterable of str,
            designating the word to replace the mask with.
        mask: The mask str. If not provided, use tokenizer.mask_token.
        prepend_space: If provided, it should be the space to prepend the text
            and to include in the mask token (most commonly, ' '). This is used
            for tokenizers like GPT-2's, which includes the preceding space as
            part of the token.
        padding: passed to tokenizer. Default to "longest" because it doesn't
            matter so much for our small datasets.
        return_tensors: passed to tokenizer.
        kwargs: passed to tokenizer.
        """
        self.mask = tokenizer.mask_token if mask is None else mask
        if prepend_space:
            self.mask = prepend_space + self.mask
            raw_texts = (prepend_space + raw_text for raw_text in raw_texts)
        if replace_strs is not None:
            self.text_words = [
                (raw_text.replace(self.mask, replace_str), replace_str)
                for raw_text, replace_str in zip(raw_texts, replace_strs)
            ]
        else:
            self.text_words = [(raw_text, None) for raw_text in raw_texts]
        texts = [text for text, word in self.text_words]
        self.encoded = tokenizer(
            texts,
            padding=padding,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            **kwargs
        )

    def __len__(self):
        return len(self.text_words)

    def __getitem__(self, i):
        return BatchEncoding({key: value[i] for key, value in self.encoded.items()})


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_embeddings_weight(model):
    embeddings = model.resize_token_embeddings()
    return embeddings.weight


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def switch_model_checkpoint(model, checkpoint_path, args):
    try:
        # if there are saved novel word embeddings, then simply load and set them
        novel_word_embeddings = torch.load(checkpoint_path + '/' + 'novel_word_embeddings.pt', map_location=args.device)

    except FileNotFoundError:
        # otherwise, assume there is a full checkpoint and load it
        model = model_class.from_pretrained(checkpoint_path)

    else:
        embs = novel_word_embeddings[0]
        embeddings_weight = get_embeddings_weight(model)
        with torch.no_grad():
            for token_id, embedding in zip(args.token_ids, embs):
                embeddings_weight[token_id] = embedding

    return model


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: args.replace_by_mask_probability MASK, args.replace_by_random_probability random, rest original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    nonce_tokens_mask = torch.tensor([
       list(map(lambda x: x in args.unused_list_ids_set, val)) for val in labels.tolist()
    ], dtype=torch.bool)
    special_tokens_mask = torch.tensor([
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ], dtype=torch.bool)
    special_tokens_mask &= ~nonce_tokens_mask  # exclude nonce tokens
    if True:
        print('special_tokens_mask:')
        print(special_tokens_mask)
        print('nonce_tokens_mask:')
        print(nonce_tokens_mask)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    if args.nonce_mlm_probability is not None:
        probability_matrix.masked_fill_(nonce_tokens_mask, value=args.nonce_mlm_probability)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, args.replace_by_mask_probability)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, args.replace_by_random_probability / (1. - args.replace_by_mask_probability))).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time, we keep the masked input tokens unchanged

    return inputs, labels


def collate_fn(features: List[Any]):
    return type(features[0])(default_data_collator(features))


def get_inputs_and_labels(batch, tokenizer, args):
    if args.mlm:
        return mask_tokens(batch.input_ids, tokenizer, args)
    else:
        inputs = batch.input_ids
        labels = batch.input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
    save_training_states=False, save_entire_model=False) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb_log'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#    scheduler = get_linear_schedule_with_warmup(
#        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
#    )
    scheduler = get_constant_schedule(optimizer)

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    def _evaluate(global_step, num_epoch):
        if False:  # do original evaluation
            results = evaluate(args, model, tokenizer)
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        results_prob = evaluate_prob(args, model, tokenizer, num_epoch, 'dev', longer=args.eval_longer, pll_whole_sentence=args.pll_whole_sentence)
        update_evaluation_results(results_prob, num_epoch)
        for key, value in results_prob.items():
            tb_writer.add_scalar("eval_prob_dev_{}".format(key), value, global_step)

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    _evaluate(global_step, epochs_trained - 1)  # evaluate before training

    token_ids_tensor = torch.tensor(args.token_ids, device=args.device)

    stimuli_sets = {'': train_dataset}  # '' as the key for train stimuli
    for split in args.save_embeddings_for_stimuli_splits:
        for ctg_n in range(2):
            stimuli_sets[f'_{split}_cat{ctg_n}'] = MaskReplacedTextDataset(
                tokenizer,
                raw_texts=read_lines(args.eval_prefix + f'_different_ctg{ctg_n}_{split}.txt'),
                replace_strs=itertools.repeat(args.token_sequences_text[ctg_n]),
                prepend_space=args.prepend_space,
                max_length=args.block_size,
                mask=tokenizer.mask_token
            )

    print(f"TRAIN STIMULI: {stimuli_sets[''].text_words}")
    print(f"STIMULI NUMBERS:")
    for key, stimuli in stimuli_sets.items():
        print(f'{key} STIMULI NUMBER: {len(stimuli)}')
    print(f'TOKEN IDS TENSOR: {token_ids_tensor}')

    print(f"TOKENIZED: {stimuli_sets[''][1]}")

    def _get_layerwise_embs_for_novel_words(stimuli):
        model.to('cpu')
        model.eval()

        # assume single token
        # save all layers.
        layerwise_embs = extract_representation(stimuli.text_words, layer='all', model=model, tokenizer=tokenizer)

        model.train()
        model.to(args.device)

        return layerwise_embs

    def _get_embs_for_novel_words(stimuli):
        embeddings_weight = get_embeddings_weight(model)
        embs = embeddings_weight.index_select(0, token_ids_tensor).detach().to('cpu')
        layerwise_embs = _get_layerwise_embs_for_novel_words(stimuli)
        return [embs] + layerwise_embs

    def _save_checkpoint(i):
        checkpoint_prefix = "checkpoint"
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, i))
        os.makedirs(output_dir, exist_ok=True)

        for key, stimuli in stimuli_sets.items():
            novel_word_embeddings = _get_embs_for_novel_words(stimuli)
            torch.save(novel_word_embeddings, os.path.join(output_dir, f"novel_word_embeddings{key}.pt"))

        if save_entire_model:
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))

        logger.info("Saving model checkpoint to %s", output_dir)

        _rotate_checkpoints(args, checkpoint_prefix)

        if save_training_states:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if epochs_trained == 0:
        _save_checkpoint(epochs_trained - 1)

    for num_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = get_inputs_and_labels(batch, tokenizer, args)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            attention_mask = batch.attention_mask.to(args.device)
            if True:
                print(f'inputs: {inputs}')
                print(f'labels: {labels}')
                print(f"attention_mask: {attention_mask}")
            model.train()
            outputs = model(inputs, labels=labels, attention_mask=attention_mask) if args.mlm else model(inputs, labels=labels)
            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            print(f"loss = {loss.item()}")
            loss = torch.nan_to_num(loss)

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Zero out the gradients of parts of the embedding matrix for known words
                weight_matrix = get_embeddings_weight(model)
                #print(len(weight_matrix))
                #print(weight_matrix.grad.data)
                for i in range(len(weight_matrix)):
                    if i not in args.token_ids:
                        weight_matrix.grad.data[i].fill_(0)

                # print(weight_matrix[token_ids_tensor,])
                # # print(weight_matrix.grad[token_ids_tensor,])
                # print(f"GRAD SHAPE: {weight_matrix.shape}")
                # print(weight_matrix.grad[token_ids_tensor, ].shape)
                # print(F"TOKENIZER LENGTH: {len(tokenizer)}")
                #print(weight_matrix)
                #print(weight_matrix.grad.data)
                #print(weight_matrix.grad.data[6])

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        _evaluate(global_step, num_epoch)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_checkpoint(global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # checkpoint at each epoch
        if True:
            _save_checkpoint(num_epoch)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = MaskReplacedTextDataset(
        tokenizer,
        raw_texts=read_lines(args.eval_data_file),
        replace_strs=None,
        prepend_space=args.prepend_space,
        max_length=args.block_size,
        mask = tokenizer.mask_token
    )

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = get_inputs_and_labels(batch, tokenizer, args)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs.loss
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity,
              "loss": eval_loss}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def get_logits_of_sequence(model, inputs, n_tokens, token_sequence, pad_token_id, mask_token_id, device=None):
    """Get pseudo-logits of token_sequence on inputs.
    WARNING: This is actually not the pseudo-log-likelihood, because the
    logits are not normalized for each token. It is wrong to compare this value
    obtained from multi-tokens.
    Input:
        inputs: tensor of shape [batch_size, max_length], input token ids
        n_tokens: length of token sequence
        token_sequences: token sequence
    Returns:
        logits of token_sequence, a tensor of shape [batch_size]
    """
    assert n_tokens == len(token_sequence)
    logits = torch.zeros(inputs.size(0), device=device)
    inputs = inputs.tolist()
    for i in range(n_tokens):
        # replace masks in inputs by token_sequence with i-th token masked
        masked_token_sequence = token_sequence[:]
        masked_token_sequence[i] = mask_token_id
        inputs_i, idxes = replace_element_in_seqs(
            mask_token_id, masked_token_sequence, inputs, return_idxes=True)
        # assume there are same number of masks (1) in each seq in batch inputs
        # so the lengths after replacement are still the same
        inputs_i = torch.tensor(inputs_i, device=device)
        idxes = torch.tensor(idxes, device=device)
        idxes += i

        predictions = model(inputs_i, attention_mask=(inputs_i != pad_token_id).int())
        all_mask_logits = predictions[0][..., token_sequence[i]].gather(1, idxes).squeeze(-1)
        logits += all_mask_logits

    return logits


def get_logits_of_sequences(model, inputs, n_tokens, token_sequences, pad_token_id, mask_token_id, device=None):
    """Get pseudo-logits of all token_sequences on inputs.
    Input:
        inputs: tensor of shape [batch_size, max_length], input token ids
        n_tokens: length of token sequence
        token_sequences: list of token sequences
    Returns:
        list of tensors, i-th of which is the logits of token_sequences[i],
        a tensor of shape [batch_size]
    """

    if n_tokens == 1 and False:  # TODO: fast path for evaluating single tokens
        if device is not None:
            inputs = inputs.to(device)
        predictions = model(inputs, attention_mask=(inputs != pad_token_id).int())
        all_mask_logits = []
        for inp, pred in zip(inputs, predictions[0]):
            mask_idx = (inp == mask_token_id).nonzero().int().item()
            mask_logits = pred[mask_idx]
            all_mask_logits.append(mask_logits)
        all_mask_logits = torch.stack(all_mask_logits)
        return [all_mask_logits[..., token_sequence[0]]
                for token_sequence in token_sequences]

    return [
        get_logits_of_sequence(
            model, inputs, n_tokens, token_sequence, pad_token_id, mask_token_id,
            device=device)
        for token_sequence in token_sequences]


def evaluate_prob(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, num_epoch, split, prefix="", longer=False, pll_whole_sentence=True) -> Dict:
    eval_output_dir = args.output_dir

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    data_kinds = {
        'diff': '_different_{}_{}.txt'.format,
        'ident': '_identical_{}_{}.txt'.format,
    }
    if longer:
        data_kinds['diff_longer'] = '_different_{}_{}_longer.txt'.format

    ctgs = ['ctg0', 'ctg1']
    all_ctgs = ['all'] + ctgs

    fieldnames = ['epoch'] + [
        f'{ctg}_{data_kind}'
        for data_kind in data_kinds
        for ctg in all_ctgs
    ]

    output_eval_file = os.path.join(eval_output_dir, prefix, f"eval_results_{split}.tsv")
    if not os.path.exists(output_eval_file):
        with open(output_eval_file, 'w') as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames, dialect='excel-tab')
            writer.writeheader()

    if pll_whole_sentence:
        datasets = {}
        for data_kind, file_suffix_fn in data_kinds.items():
            for ctg in ctgs:
                raw_texts = list(read_lines(args.eval_prefix + file_suffix_fn(ctg, split)))
                datasets[f'{ctg}_{data_kind}'] = [
                    MaskReplacedTextDataset(
                        tokenizer,
                        raw_texts=raw_texts,
                        replace_strs=itertools.repeat(token_sequence_text),
                        prepend_space=args.prepend_space,
                        max_length=args.block_size,
                        mask=tokenizer.mask_token
                    )
                    for token_sequence_text in args.token_sequences_text
                ]
    else:
        datasets = {
            f'{ctg}_{data_kind}': MaskReplacedTextDataset(
                tokenizer,
                raw_texts=read_lines(args.eval_prefix + file_suffix_fn(ctg, split)),
                replace_strs=None,
                prepend_space=args.prepend_space,
                max_length=args.block_size,
                mask=tokenizer.mask_token
            )
            for data_kind, file_suffix_fn in data_kinds.items()
            for ctg in ctgs
        }

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    stats = {
        data_kind: Stat()
        for data_kind in data_kinds
    }
    results = {}

    for name, dataset in datasets.items():
        for data_kind, stat in stats.items():
            if name.endswith(data_kind):
                break

        if pll_whole_sentence:
            eval_dataloader = [
                DataLoader(
                    dataset_,
                    sampler=SequentialSampler(dataset_),
                    batch_size=args.eval_batch_size,
                    collate_fn=collate_fn,
                )
                for dataset_ in dataset
            ]
            eval_dataloader = zip(*eval_dataloader)
        else:
            eval_sampler = SequentialSampler(dataset)
            eval_dataloader = DataLoader(
                dataset,
                sampler=eval_sampler,
                batch_size=args.eval_batch_size,
                collate_fn=collate_fn,
            )

        model.eval()

        # Eval!
        logger.info("***** Running evaluation on {} *****".format(name))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        stat_local = Stat()

        if pll_whole_sentence:
            if args.mlm:
                scorer = MaskedLMScorer(model, tokenizer=tokenizer, device=args.device)
            else:
                scorer = IncrementalLMScorer(model, tokenizer=tokenizer, device=args.device)
        if name.startswith('ctg0'):
            target_ctg, foil_ctg = 0, 1
        elif name.startswith('ctg1'):
            target_ctg, foil_ctg = 1, 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                if pll_whole_sentence:
                    ctg_logits = []
                    for batch_ in batch:
                        batch_.to(args.device)
                        ctg_logits.append(
                            torch.tensor(
                                scorer.sequence_score(
                                    batch_,
                                    reduction=lambda x: x.sum(0).item()
                                ),
                                dtype=torch.float64,
                            )
                        )
                else:
                    ctg_logits = get_logits_of_sequences(
                        model, batch.input_ids, args.n_tokens, args.token_sequences,
                        tokenizer.pad_token_id, tokenizer.mask_token_id, device=args.device)
                correct = ctg_logits[target_ctg] > ctg_logits[foil_ctg]
                correct = correct.cpu().numpy()
                stat_local.update(correct)
                stat.update(correct)

            results[name] = stat_local.mean()

    results.update({
        'epoch': num_epoch,
    })
    results.update({
        f'all_{data_kind}': stat.mean() for data_kind, stat in stats.items()
    })

    with open(output_eval_file, "a") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames, dialect='excel-tab')
        writer.writerow(results)

    for k in fieldnames:
        v = results[k]
        logger.info('%s: %s', k, v)

    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument('--eval_prefix', default=None, type=str)
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss."
    )
    parser.add_argument(
        "--nonce_mlm_probability", type=float, default=None, help="Ratio of nonce tokens to mask for masked language modeling loss. Default to mlm_probability."
    )
    parser.add_argument(
        "--replace_by_mask_probability", type=float, default=0.8, help="Probability of replacing a input token for masked language modeling by a mask token. Default to 0.8."
    )
    parser.add_argument(
        "--replace_by_random_probability", type=float, default=0.1, help="Probability of replacing a input token for masked language modeling by a random token. Default to 0.1. With probability (1 - replace_by_mask_probability - replace_by_random_probability), the input token is unchanged."
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--n_tokens", type=int, default=1, help="Length of the token sequence.")
    parser.add_argument("--tokens", type=str, nargs='+', default=None,
                        help="Tokens to train as the novel words in the two categories. Should have exactly 2 * n_tokens arguments (separated by space), each is a token. If a token is not in the tokenizer vocabulary (thus is novel), it will be added to the vocabulary with its embedding initialized according to the random seed. If unspecified, create default novel tokens.")
    parser.add_argument("--eval_longer", action="store_true")
    parser.add_argument("--pll_whole_sentence", action="store_true", help="Uses whole sentence pseudo log likelihood (instead of just logits on masked tokens) in evaluation.")
    parser.add_argument("--save_embeddings_for_stimuli_splits", type=str, choices=['dev', 'test'], nargs='*', default=['dev'],
                        help="save embeddings for stimuli splits in addition to the training stimuli.")

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        print('using cuda')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    model_class = AutoModelForMaskedLM if args.mlm else AutoModelForCausalLM

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir, output_hidden_states=True)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, output_hidden_states=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    print(f"Vocab size in initial config: {config.vocab_size}")

    n_total_tokens = 2 * args.n_tokens
    if args.tokens is None:
        args.tokens = [f'[novel-{ctg_n}-{k}]' for ctg_n in range(2) for k in range(args.n_tokens)]
    else:
        assert len(args.tokens) == n_total_tokens, f"Number of tokens should be {n_total_tokens}"
    args.unused_list = []
    if args.model_type in ["bert"]:
        args.unused_list.extend(map('[unused{}]'.format, range(0, 994)))
    if args.model_type in ["roberta"]:
        args.unused_list.extend(map('[unused{}]'.format, range(1, 3)))
    for token in args.tokens:
        if token not in args.unused_list:
            args.unused_list.append(token)
    if args.model_type == 'bert':
        special_tokens_dict = {
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "additional_special_tokens": args.unused_list,
        }
    elif args.model_type == 'roberta':
        special_tokens_dict = {
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "additional_special_tokens": args.unused_list,
        }
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir, **special_tokens_dict)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, **special_tokens_dict)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it, and load it from here, using --tokenizer_name"
        )
    print(f"Vocab size in final tokenizer: {len(tokenizer)}")
    if isinstance(tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
        args.prepend_space = ' '
    else:
        args.prepend_space = None
    args.unused_list_ids = tokenizer.additional_special_tokens_ids
    args.unused_list_ids_set = set(args.unused_list_ids)
    print(f'tokens: {" ".join(args.tokens)}')
    args.token_ids = tokenizer.convert_tokens_to_ids(args.tokens)
    print(f'token ids: {" ".join(map(str, args.token_ids))}')
    args.token_sequences = [args.token_ids[ctg_n * args.n_tokens : (ctg_n + 1) * args.n_tokens] for ctg_n in range(2)]
    print(f'token sequences: {args.token_sequences}')
    args.token_sequences_text = [tokenizer.decode(token_sequence) for token_sequence in args.token_sequences]
    print(f'token sequences text: {args.token_sequences_text}')


    if False:
        if args.block_size <= 0:
            # args.block_size = tokenizer.max_len
            args.block_size = tokenizer.model_max_length
            # Our input block size will be the max possible for the model
        else:
            # args.block_size = min(args.block_size, tokenizer.max_len)
            args.block_size = min(args.block_size, tokenizer.model_max_length)
    else:
        args.block_size = 512

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    # resize embeddings weight matrix in case new tokens are added
    initial_model_vocab_size = get_embeddings_weight(model).size(0)
    if initial_model_vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        if initial_model_vocab_size < len(tokenizer):
            # initialize new embeddings by the mean of all existing embeddings
            embeddings_weight = get_embeddings_weight(model)
            embeddings_weight.requires_grad = False
            mean_embedding = embeddings_weight[:initial_model_vocab_size].mean(0).detach()
            embeddings_weight[initial_model_vocab_size:] = mean_embedding
            embeddings_weight.requires_grad = True
            # TODO: support other initialization schemes
            # TODO: also save the inital embeddings (for visualization)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # evaluate untrained model
    evaluate_prob(args, model, tokenizer, -1, 'test', longer=args.eval_longer, pll_whole_sentence=args.pll_whole_sentence)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        # freeze all parameters
        for param_name, param in model.named_parameters():
            param.requires_grad = False
        # except the embeddings weight
        embeddings_weight = get_embeddings_weight(model)
        for name, param in model.named_parameters():
            if param is embeddings_weight:
                embeddings_weight_name = name
                break
        else:
            assert False, "Parameter name of embeddings weight is not found"
        print(f'Not freezing embeddings weight parameter {embeddings_weight_name} of size {embeddings_weight.size()}')
        embeddings_weight.requires_grad = True

        train_dataset = MaskReplacedTextDataset(
            tokenizer,
            raw_texts=read_lines(args.train_data_file),
            replace_strs=args.token_sequences_text,
            prepend_space=args.prepend_space,
            max_length=args.block_size,
            mask=tokenizer.mask_token
        )

        print(f'TRAIN DATASET: {train_dataset.text_words}')

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    if not (args.do_train and args.evaluate_during_training):
        for i in range(0, int(args.num_train_epochs)):
            model = switch_model_checkpoint(model, os.path.join(args.output_dir, f'checkpoint-{i}'), args)
            model.to(args.device)
            results = evaluate_prob(args, model, tokenizer, i, 'dev', longer=args.eval_longer, pll_whole_sentence=args.pll_whole_sentence)
            update_evaluation_results(results, i)

    print("Best dev for structurally different: ", best_dev_diff.value, best_dev_diff.ids)
    print("Best dev for structurally identical: ", best_dev_ident.value, best_dev_ident.ids)

    common_best_eps = set(best_dev_diff.ids) & set(best_dev_ident.ids)
    if common_best_eps:
        best_diff_ep = max(common_best_eps)
        best_ident_ep = max(common_best_eps)
    else:
        best_diff_ep = max(best_dev_diff.ids)
        best_ident_ep = max(best_dev_ident.ids)

    # evaluate the model after the first epoch and selected models
    for i in [0, best_diff_ep, best_ident_ep]:
        model = switch_model_checkpoint(model, os.path.join(args.output_dir, f'checkpoint-{i}'), args)
        model.to(args.device)
        evaluate_prob(args, model, tokenizer, i, 'test', longer=args.eval_longer, pll_whole_sentence=args.pll_whole_sentence)


if __name__ == "__main__":
    main()
