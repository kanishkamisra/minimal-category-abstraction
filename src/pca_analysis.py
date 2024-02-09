import argparse
import csv
import os
import pathlib
import re
import random
import torch

import numpy as np

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from minicons import cwe
from rep_analysis_utils import project, reconfigure_dist
from semantic_memory import vsm, vsm_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from tqdm import tqdm
from wordfreq import word_frequency


def get_embeddings(exp_dir, epoch):
    initial_path = f"{exp_dir}/checkpoint--1/novel_word_embeddings.pt"
    final_path = f"{exp_dir}/checkpoint-{epoch}/novel_word_embeddings.pt"

    initial_embs = torch.load(initial_path)[0]
    final_embs = torch.load(final_path)[0]

    # stack initial and final embeddings for each cat
    embs_cat1 = torch.stack((initial_embs[0], final_embs[0]))
    embs_cat2 = torch.stack((initial_embs[1], final_embs[1]))

    return embs_cat1, embs_cat2


def main(args):
    results_file = args.results_file  # data/results/childes/model/cat_abs_results.csv
    vocab_dir = args.vocab_dir
    N = args.n
    num_states = 2

    # random.seed(seed)
    random.seed(42)

    pair2labels = {
        "adjadv": ["adj", "adv"],
        "adjn": ["n", "adj"],
        "adjv": ["adj", "v"],
        "advn": ["n", "adv"],
        "advv": ["adv", "v"],
        "nv": ["n", "v"],
    }

    labels2pair = {"".join(v): k for k, v in pair2labels.items()}

    label2cat = {
        "n": "noun",
        "v": "verb",
        "adj": "adjective",
        "adv": "adverb",
    }

    # parse dataset name
    dataset_name = results_file.split("/")[-3]

    cats = ["noun", "verb", "adjective", "adverb"]
    cat_words = defaultdict(list)
    for cat in cats:
        with open(f"{vocab_dir}/{cat}.vocab", "r", encoding="utf-8") as f:
            words = []
            for line in f:
                words.append(line.strip())

            cat_words[cat] = words
    cat_words = dict(cat_words)

    if "smolm" in results_file:
        sampled = {k: v[:N] for k, v in cat_words.items()}
        # print(sampled)
    else:
        sampled = {k: random.sample(v, min(N, len(v))) for k, v in cat_words.items()}

    model = args.results_file.split("/")[3]
    # print(model)
    lm = cwe.CWE(f"kanishka/{model}")
    model_embs = lm.model.resize_token_embeddings().weight.detach()

    cat_vectors = {"noun": None, "verb": None, "adjective": None, "adverb": None}

    for k, v in cat_vectors.items():
        idxes = []
        vocab = sampled[k]
        for word in vocab:
            idx = lm.tokenizer(word)["input_ids"][1]
            idxes.append(idx)
        cat_vectors[k] = model_embs[idxes,]

    catpairs = [
        ("adjective", "adverb"),
        ("noun", "adjective"),
        ("adjective", "verb"),
        ("noun", "adverb"),
        ("adverb", "verb"),
        ("noun", "verb"),
    ]

    cat_pcas = defaultdict()
    for pair in catpairs:
        c1, c2 = pair
        embs = torch.cat((cat_vectors[c1], cat_vectors[c2]))
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embs)
        cat_pcas[pair] = (pca, coords)

    cat_pcas = dict(cat_pcas)

    experiment_dirs = []

    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # construct path to checkpoints
            cat1, cat2 = row["cat1"], row["cat2"]
            categories = labels2pair[cat1 + cat2]
            checkpoints_path = f'checkpoints/{model}/unused_pairs_1/{dataset_name}_{categories}_unused_token_numbers_1_2_learning_rate_{row["lr"]}_seed_{row["seed"]}'
            experiment_dirs.append((checkpoints_path, [cat1, cat2], row["epoch"]))

            # initial_path = f"{checkpoints_path}/checkpoint--1"
            # final_path = f"{checkpoints_path}/checkpoint-{row['epoch']}"

            # model_data[row["model"]]["category_pair"].append((cat1, cat2))
            # model_data[row["model"]]["initial_path"].append(initial_path)
            # model_data[row["model"]]["final_path"].append(final_path)

            # print(checkpoints_path)
            # print(row["model"])

    unused_words = ["[unused_1]", "[unused_2]"]
    outpath = results_file.replace("cat_abs_results.csv", "")

    pca_data_all = []
    rep_stats_all = []
    # explained_variance_all = []

    for experiment in experiment_dirs:
        exp_dir, cats, epoch = experiment
        cats = [label2cat[l] for l in cats]
        # print(cats)

        query1, query2 = sampled[cats[0]], sampled[cats[1]]
        query = [*query1, *query2]

        embs = torch.cat((cat_vectors[cats[0]], cat_vectors[cats[1]]))
        pca, coords = cat_pcas[(cats[0], cats[1])]

        # print(coords.shape)

        explained_variance = pca.explained_variance_ratio_.sum()

        pca_prototype1 = coords[: len(query1)].mean(0)
        pca_prototype2 = coords[len(query2) :].mean(0)
        pca_prototype_tensor = torch.tensor(
            np.concatenate(([pca_prototype1], [pca_prototype2]))
        )

        emb_prototype1 = cat_vectors[cats[0]].mean(0)
        emb_prototype2 = cat_vectors[cats[1]].mean(0)
        emb_prototype_tensor = torch.stack((emb_prototype1, emb_prototype2))

        emb_cat1, emb_cat2 = get_embeddings(exp_dir, epoch)

        starts = torch.stack([emb_cat1[0,], emb_cat2[0,]])
        ends = torch.stack([emb_cat1[-1,], emb_cat2[-1,]])

        emb_dist_start = torch.cdist(starts, emb_prototype_tensor, p=2)
        emb_dist_end = torch.cdist(ends, emb_prototype_tensor, p=2)

        (
            emb_dist_start1,
            emb_dist_end1,
            emb_dist_start2,
            emb_dist_end2,
        ) = reconfigure_dist(emb_dist_start, emb_dist_end)

        # print(emb_dist_start1, emb_dist_end1, emb_dist_start2, emb_dist_end2)
        projections_start = project(emb_prototype1, emb_prototype2, starts)
        projections_end = project(emb_prototype1, emb_prototype2, ends)
        emb_movement = (
            (projections_end - projections_start) * torch.tensor([1, -1])
        ).tolist()

        start_coords = pca.transform(starts)
        end_coords = pca.transform(ends)

        pca_dist_start = torch.cdist(
            torch.tensor(start_coords), pca_prototype_tensor, p=2
        )
        pca_dist_end = torch.cdist(torch.tensor(end_coords), pca_prototype_tensor, p=2)

        (
            pca_dist_start1,
            pca_dist_end1,
            pca_dist_start2,
            pca_dist_end2,
        ) = reconfigure_dist(pca_dist_start, pca_dist_end)

        # print(pca_dist_start1, pca_dist_end1, pca_dist_start2, pca_dist_end2)
        projections_start = project(
            torch.tensor(pca_prototype1),
            torch.tensor(pca_prototype2),
            torch.tensor(start_coords),
        )
        projections_end = project(
            torch.tensor(pca_prototype1),
            torch.tensor(pca_prototype2),
            torch.tensor(end_coords),
        )
        pca_movement = (
            (projections_end - projections_start) * torch.tensor([1, -1])
        ).tolist()

        # print(emb_movement, pca_movement)
        pcaed1 = pca.transform(emb_cat1)
        pcaed2 = pca.transform(emb_cat2)

        xes = np.concatenate([x[:, 0] for x in (coords, pcaed1, pcaed2)])
        ys = np.concatenate([y[:, 1] for y in (coords, pcaed1, pcaed2)])

        types = ["real"] * len(query) + ["novel"] * num_states * 2
        states = ["none"] * len(query) + (
            ["start"] + ["intermediate"] * (num_states - 2) + ["end"]
        ) * 2

        # print(states)

        # print(len(xes), len(ys))
        cat_repeated = (
            [cats[0]] * len(query1)
            + [cats[1]] * len(query2)
            + [cats[0]] * num_states
            + [cats[1]] * num_states
        )
        words = query + [unused_words[0]] * num_states + [unused_words[1]] * num_states

        pca_data = list(
            zip(
                [exp_dir] * (len(query) + num_states * 2),
                words,
                xes,
                ys,
                cat_repeated,
                types,
                states,
            )
        )

        representational_stats = list(
            zip(
                [exp_dir] * 4,
                unused_words * 2,
                cats * 2,
                ["pca"] * 2 + ["emb"] * 2,
                [
                    pca_dist_start1,
                    pca_dist_start2,
                    emb_dist_start1,
                    emb_dist_start2,
                ],
                [pca_dist_end1, pca_dist_end2, emb_dist_end1, emb_dist_end2],
                [
                    pca_movement[0],
                    pca_movement[1],
                    emb_movement[0],
                    emb_movement[1],
                ],
            )
        )

        # ev_stats = [dirname, explained_variance]

        rep_stats_all.extend(representational_stats)
        pca_data_all.extend(pca_data)
        # explained_variance_all.append(ev_stats)

    # print(rep_stats_all)

    with open(f"{outpath}/pca_analysis.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dirkey", "word", "pc1", "pc2", "cat", "type", "state"])
        writer.writerows(pca_data_all)

    with open(f"{outpath}/rep_movement.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dirkey",
                "word",
                "cat",
                "rep_type",
                "start",
                "end",
                "projection_movement",
            ]
        )
        writer.writerows(rep_stats_all)


if __name__ == "__main__":
    # main(args)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_file",
        type=str,
        default="data/results/childes/smolm-autoreg-bpe-seed_111/cat_abs_results.csv",
    )
    parser.add_argument(
        "--vocab_dir",
        type=str,
        default="../smolm/data/tests/childes_samples/vocabulary",
    )
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    main(args)
