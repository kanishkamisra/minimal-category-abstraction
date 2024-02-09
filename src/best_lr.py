import argparse
import csv
import os
import pathlib

from collections import defaultdict

import numpy as np

category2pair = {
    "adjadv": ["adj", "adv"],
    "adjn": ["n", "adj"],
    "advn": ["n", "adv"],
    "adjv": ["adj", "v"],
    "advv": ["adv", "v"],
    "nv": ["n", "v"],
}


def parse_checkpoint(checkpoint_path, dataset_name):
    _dir, model, _pairs, experiment = checkpoint_path.split("/")
    dataset, categories, _, _, _, num1, num2, _, _, lr, _, seed = experiment.split("_")
    num1 = int(num1)
    num2 = int(num2)
    seed = int(seed)
    lr = float(lr)
    assert dataset == dataset_name
    return {
        "model": model,
        "dataset": dataset,
        "categories": categories,
        "num1": num1,
        "num2": num2,
        "lr": lr,
        "seed": seed,
    }


def parse_results(results_path):
    # read tsv with dict reader
    data = {
        "epoch": [],
        "all_diff": [],
    }
    with open(results_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["all_diff"].append(float(row["all_diff"]))

    return data


def main(args):
    checkpoints_dir = args.checkpoints_dir
    model_pattern = args.model

    # dev_results = defaultdict(lambda: defaultdict(list))
    # example_checkpoint_path = "checkpoints/smolm-autoreg-bpe-seed_111/unused_pairs_1/childes_adjadv_unused_token_numbers_1_2_learning_rate_0.01_seed_1"
    results = []

    for model in os.listdir(checkpoints_dir):
        if model_pattern in model:
            path = f"{args.checkpoints_dir}/{model}/unused_pairs_1/"
            # print(model)
            lr_results = defaultdict(list)
            # test_results = defaultdict(list)
            for directory in os.listdir(path):
                # print(f"==== {directory} ====")
                # print(f"{path}{directory}".split("/"))
                try:
                    metadata = parse_checkpoint(f"{path}{directory}", args.dataset)
                    dev_results = parse_results(
                        f"{path}{directory}/eval_results_dev.tsv"
                    )
                    # print(dev_results)
                    test_results = parse_results(
                        f"{path}{directory}/eval_results_test.tsv"
                    )

                    # get best dev epoch
                    best_idx = np.argmax(dev_results["all_diff"])
                    best_dev_epoch = dev_results["epoch"][best_idx]
                    best_dev_acc = dev_results["all_diff"][best_idx]

                    test_acc = test_results["all_diff"][-1]

                    # lr_results.append((metadata['lr'], best_dev_epoch, best_dev_acc))
                    lr_results[f"{metadata['categories']}_{metadata['seed']}"].append(
                        (metadata["lr"], best_dev_epoch, best_dev_acc, test_acc)
                    )
                except AssertionError:
                    print(f"Error: {path}{directory}")

            # print(lr_results)
            # select best lr based on dev set accuracy (best_dev_acc)
            best_lr = {}
            for key, value in lr_results.items():
                best_lr[key] = max(value, key=lambda x: x[2])

            # print(best_lr)
            for k, v in best_lr.items():
                categories, seed = k.split("_")
                cat1, cat2 = category2pair[categories]
                results.append((model, cat1, cat2, seed, v[0], v[1], v[2], v[3]))

    results_path = f"data/results/{args.dataset}/{args.model}/"
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

    with open(f"{results_path}/cat_abs_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "cat1", "cat2", "seed", "lr", "epoch", "dev_acc", "test_acc"]
        )
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--model", type=str, default="smolm-autoreg-bpe-seed_111")
    parser.add_argument("--dataset", type=str, default="childes")
    args = parser.parse_args()
    main(args)
