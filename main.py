import argparse
import json
import os
import random
import _pickle as pickle

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model.mamba_bisamnet import MambaBiSAMNet
from trainandtest.trainandtest import evaluate_model_across_datasets, train_unet
from utils.helpers import seed_everything


VAL_RATIO = 0.3
RATIO_LO = 0.25
RATIO_HI = 0.35
NOISE_CHOICES = ["bw", "ma", "em", "mixed"]
VERSION_CHOICES = ["nv1", "nv2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument(
        "--noise-types",
        nargs="+",
        choices=NOISE_CHOICES,
        default=["ma"],
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=VERSION_CHOICES,
        default=VERSION_CHOICES,
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    args = parser.parse_args()

    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")

    return args


def to_tensor(array):
    return torch.FloatTensor(array).permute(0, 2, 1)


def record_independent_split(num_samples, seed, flag_train):
    if flag_train.size == 0 or flag_train[0] != 0:
        raise ValueError("dataset/flag_train.npy must start with 0.")
    if np.any(np.diff(flag_train) <= 0) or flag_train[-1] >= num_samples:
        raise ValueError("dataset/flag_train.npy contains invalid record boundaries.")

    record_groups = np.split(np.arange(num_samples), flag_train[1:])
    num_records = len(record_groups)
    sizes = np.array([len(group) for group in record_groups])
    total = sizes.sum()
    record_indices = np.arange(num_records)

    for split_seed in range(seed, seed + 200):
        train_records, validation_records = train_test_split(
            record_indices,
            test_size=VAL_RATIO,
            random_state=split_seed,
        )
        validation_ratio = sizes[validation_records].sum() / total
        if RATIO_LO <= validation_ratio <= RATIO_HI:
            train_indices = np.concatenate(
                [record_groups[index] for index in train_records]
            )
            validation_indices = np.concatenate(
                [record_groups[index] for index in validation_records]
            )
            return (
                train_indices,
                validation_indices,
                validation_ratio,
                split_seed,
                len(train_records),
                len(validation_records),
            )

    raise RuntimeError(
        "Unable to find a record-level split within the requested validation ratio range."
    )


def summarize_seed_results(all_seed_results):
    summary = {}

    for noise_type, seed_results in all_seed_results.items():
        if not seed_results:
            continue

        metric_names = next(iter(seed_results.values())).keys()
        summary[noise_type] = {}

        for metric_name in metric_names:
            values = np.array(
                [metrics[metric_name] for metrics in seed_results.values()],
                dtype=float,
            )
            summary[noise_type][metric_name] = {
                "mean": float(np.mean(values)),
                "sample_sd": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                ),
            }

    return summary


def save_results(path, seeds, all_seed_results):
    payload = {
        "seeds": seeds,
        "per_seed": all_seed_results,
        "summary": summarize_seed_results(all_seed_results),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = config["device"]
    flag_train = np.load("dataset/flag_train.npy").astype(int).ravel()
    print(
        f"flag_train: {flag_train.shape[0]} records, "
        f"first5={flag_train[:5]}"
    )

    seeds = random.SystemRandom().sample(
        range(1, 1_000_000),
        args.num_seeds,
    )
    print(f"Random seeds: {seeds}")

    result_save_path = os.path.join(
        "results",
        "mamba_bisamnet",
        "seed_results.json",
    )
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    all_seed_results = {}
    save_results(result_save_path, seeds, all_seed_results)

    for seed in seeds:
        print(f"\n{'=' * 50}")
        print(f"Seed: {seed}")
        print(f"{'=' * 50}")
        seed_everything(seed)

        for noise_type in args.noise_types:
            test_loaders = {}

            for version in args.versions:
                seed_everything(seed)
                dataset_name = f"{noise_type}_{version}"
                print(f"\nTraining seed={seed}, dataset={dataset_name}")

                dataset_path = f"dataset/dataset_{dataset_name}.pkl"
                with open(dataset_path, "rb") as file:
                    x_train, y_train, x_test, y_test = pickle.load(file)

                (
                    train_indices,
                    validation_indices,
                    validation_ratio,
                    split_seed,
                    num_train_records,
                    num_validation_records,
                ) = record_independent_split(
                    len(x_train),
                    seed,
                    flag_train,
                )

                print(
                    f"split_seed={split_seed} | "
                    f"train {num_train_records} records/"
                    f"{len(train_indices)} beats, "
                    f"validation {num_validation_records} records/"
                    f"{len(validation_indices)} beats "
                    f"(validation ratio={validation_ratio:.3f})"
                )

                x_train_tensor = to_tensor(x_train[train_indices])
                y_train_tensor = to_tensor(y_train[train_indices])
                x_validation_tensor = to_tensor(x_train[validation_indices])
                y_validation_tensor = to_tensor(y_train[validation_indices])
                x_test_tensor = to_tensor(x_test)
                y_test_tensor = to_tensor(y_test)

                train_loader = DataLoader(
                    TensorDataset(x_train_tensor, y_train_tensor),
                    batch_size=config["train"]["batch_size"],
                    shuffle=True,
                    drop_last=True,
                )
                validation_loader = DataLoader(
                    TensorDataset(
                        x_validation_tensor,
                        y_validation_tensor,
                    ),
                    batch_size=config["train"]["batch_size"],
                    drop_last=True,
                )
                test_loader = DataLoader(
                    TensorDataset(x_test_tensor, y_test_tensor),
                    batch_size=config["test"]["batch_size"],
                )
                test_loaders[dataset_name] = test_loader

                model = MambaBiSAMNet().to(device)
                config["output_folder"] = os.path.join(
                    "results",
                    "mamba_bisamnet",
                    noise_type,
                    f"seed_{seed}",
                )
                os.makedirs(config["output_folder"], exist_ok=True)

                train_unet(
                    model,
                    train_loader,
                    validation_loader,
                    config,
                    dataset_name,
                )
                print(f"Finished seed={seed}, dataset={dataset_name}")

            metrics = evaluate_model_across_datasets(
                model,
                test_loaders,
                config,
            )
            all_seed_results.setdefault(noise_type, {})[str(seed)] = metrics
            save_results(result_save_path, seeds, all_seed_results)
            print(f"Results saved for seed={seed}, noise={noise_type}")

    summary = summarize_seed_results(all_seed_results)
    print(f"\nCompleted {len(seeds)} seed(s)")
    print(json.dumps(summary, indent=2))
    print(f"Results summary: {result_save_path}")


if __name__ == "__main__":
    main()
