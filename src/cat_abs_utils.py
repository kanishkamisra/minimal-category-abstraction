"""Utils to extract representations of words in context.
Code borrowed from minicons."""

from typing import Iterable, Union, List, Dict, Optional, Tuple

import torch
from minicons.utils import find_pattern, find_index, find_paired_indices, character_span

def encode_text(text: Union[str, List[str]], layer: Union[int, List[int]] = None, model = None, tokenizer=None, device='cpu') -> Tuple:
    """
    Encodes batch of raw sentences using the model to return hidden
    states at a given layer.
    :param ``Union[str, List[str]]`` text: batch of raw sentences
    :param layer: layer from which the hidden states are extracted.
    :type layer: int
    :return: Tuple `(input_ids, hidden_states)`
    """
    sentences = [text] if isinstance(text, str) else text

    # Encode sentence into ids stored in the model's embedding layer(s).
    encoded = tokenizer.batch_encode_plus(sentences, padding = 'longest', pad_to_max_length=True, return_tensors="pt")

    # print(f'TESTING TOKENIZATION: {tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])}')
    # print(f'THE TEXT BEING ENCODED: {text}')
    # print(f'TESTING TOKENIZATION: {encoded}')

    total_layers = model.config.num_hidden_layers

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask'].unsqueeze(-1)

    if layer == 'static' or layer == 'pre':
        hidden_states = [model.embeddings.word_embeddings.weight.detach()[i] for i in input_ids]
        hidden_states = torch.stack(hidden_states)
        hidden_states = hidden_states * attention_mask
    else:
    # Compute hidden states for the sentence for the given layer.
        model.to('cpu')
        output = model(**encoded)

        # print(output)

        # Hidden states appear as the last element of the otherwise custom hidden_states object
        if isinstance(layer, list) or layer == 'all':
            hidden_states = output[-1]
            #print(f"SHAPE: {len(hidden_states)}")
            if "cuda" in device:
                input_ids = input_ids.cpu()
                hidden_states = [h.detach().cpu() for h in hidden_states]
            else:
                hidden_states = [h.detach() for h in hidden_states]
            if layer != 'all':
                hidden_states = [hidden_states[i] for i in sorted(layer)]

            hidden_states = [h * attention_mask for h in hidden_states]
        else:
            # if layer != 'all':
            if layer is None:
                layer = total_layers
            elif layer > total_layers:
                raise ValueError(f"Number of layers specified ({layer}) exceed layers in model ({total_layers})!")
            hidden_states = output.hidden_states[layer]
            if "cuda" in device:
                input_ids = input_ids.cpu()
                hidden_states = hidden_states.detach().cpu()
            else:
                hidden_states = hidden_states.detach()

            hidden_states = hidden_states * attention_mask
            # else:
            #     hidden_states = output.hidden_states

            #     if "cuda" in self.device:
            #         input_ids = input_ids.cpu()
            #         hidden_states = [h.detach().cpu() for h in hidden_states]
            #     else:
            #         hidden_states = [h.detach() for h in hidden_states]

    return input_ids, hidden_states


def extract_representation(sentence_words: Union[List[List[Union[str, Union[Tuple[int, int], str]]]], List[Union[str, Union[Tuple[int, int], str]]]], layer: Union[int, List[int]] = None, model=None, tokenizer=None) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Extract representations from the model at a given layer.
    :param ``Union[List[List[Union[str, Union[Tuple(int, int), str]]]], List[Union[str, Union[Tuple(int, int), str]]]]`` sentence_words: Input 
        consisting of `[(sentence, word)]`, where sentence is an
        input sentence, and word is a word present in the sentence
        that will be masked out, or `[(sentence, (start, end))]`,
        where (start, end) is a tuple consisting of the character
        span indices that form the word.
    :param ``Union[int, List[int]]`` layer: layer(s) from which the hidden states are extracted.
    :return: torch tensors or list of torch
        tensors corresponding to word representations
    """
    sentences = [sentence_words] if isinstance(sentence_words[0], str) else sentence_words

    num_inputs = len(sentences)

    input_ids, hidden_states = encode_text(list(list(zip(*sentences))[0]), layer, model, tokenizer)

    if isinstance(sentences[0][1], str):
        sentences = [(s, character_span(s, w)) for s, w in sentences]

    search_queries = []
    for s, idx in sentences:
        if 0 in idx:
            search_queries.append(tokenizer.encode_plus(f'{s[idx[0]:idx[1]]}', add_special_tokens = False)['input_ids'])
        else:
            ## this one really matters if we are using GPT2
            search_queries.append(tokenizer.encode_plus(f' {s[idx[0]:idx[1]]}', add_special_tokens = False)['input_ids'])

    query_idx = list(map(lambda x: find_pattern(x[0], x[1]), zip(search_queries, input_ids.tolist())))

    total_layers = model.config.num_hidden_layers

    if isinstance(layer, list) or layer == 'all':
        # representations = list(map(lambda x: x[torch.arange(num_inputs)[:, None], query_idx].mean(1), hidden_states))
        representations = list(map(lambda x: torch.stack([hs.squeeze()[idx[0]:idx[1]].mean(0) for hs, idx in zip(x.split([1] * num_inputs), query_idx)]), hidden_states))
    else:
        if layer is None:
            layer = total_layers
        elif layer == 'static' or 'pre':
            layer = layer
        elif layer > total_layers:
            raise ValueError(f"Number of layers specified ({layer}) exceed layers in model ({total_layers})!")
        # representations = hidden_states[torch.arange(num_inputs)[:, None], query_idx].mean(1)
        representations = torch.stack([hs.squeeze()[idx[0]:idx[1]].mean(0) for hs, idx in zip(hidden_states.split([1] * num_inputs), query_idx)])

    return representations
