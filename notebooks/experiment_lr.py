import pandas as pd
import numpy as np
import sys

sys.path.append("../")

from cfmining.algorithms import P2CE
from cfmining.predictors import MonotoneClassifier
from cfmining.baselines import MAPOCAM

from experiments_helper import (
    get_data_model,
    run_experiments,
    format_df_table,
    summarize_results,
    get_action_set,
)

import argparse


SEED = 0


def get_method(method_name, model, outlier_detection, X_train, Y_train, dataset):
    max_changes = 3
    objective = "abs_diff"

    if method_name == "p2ce_abs_diff":
        action_set = get_action_set(dataset, X_train, default_step_size=0.05)
        model_wrap = MonotoneClassifier(model, outlier_detection, X=X_train)
        method = P2CE(
            action_set=action_set,
            classifier=model_wrap,
            compare=objective,
            max_changes=max_changes,
            outlier_contamination=dataset.outlier_contamination,
            estimate_outlier=True,
            time_limit=np.inf,
        )

    elif method_name == "p2ce_ablation_abs_diff":
        action_set = get_action_set(dataset, X_train, default_step_size=0.05)
        model_wrap = MonotoneClassifier(model, outlier_detection, X=X_train)
        method = P2CE(
            action_set=action_set,
            classifier=model_wrap,
            compare=objective,
            max_changes=max_changes,
            outlier_contamination=dataset.outlier_contamination,
            estimate_outlier=False,
            time_limit=np.inf,
        )
    elif method_name == "mapocam_abs_diff":
        action_set = get_action_set(dataset, X_train, default_step_size=0.05)
        model_wrap = MonotoneClassifier(model, outlier_detection, X=X_train)
        for feat in action_set:
            feat.flip_direction = 1
            feat.update_grid()

        method = MAPOCAM(action_set, model_wrap, criteria=objective, max_changes=max_changes)

    return method, model_wrap


def run(dataset_name, method_name, n_samples=500):
    dataset, X_train, Y_train, model, outlier_detection, individuals = get_data_model(
        dataset_name, "LogisticRegression"
    )
    n = min(n_samples, len(individuals))
    individuals = individuals.sample(n=n, random_state=SEED)
    outlier_detection.contamination = dataset.outlier_contamination

    method, model_wrap = get_method(
        method_name, model, outlier_detection, X_train, Y_train, dataset
    )

    print(f"Running {method_name} on {dataset_name} with {n} samples.")
    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lr/{dataset}/{method_name}.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="idx",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples to use from the dataset",
    )
    datasets = ["german", "taiwan", "adult"]
    methods = [
        "p2ce_abs_diff",
        "p2ce_ablation_abs_diff",
        "mapocam_abs_diff",
    ]
    args = parser.parse_args()

    if len(datasets) * len(methods) <= args.idx:
        raise ValueError("idx out of range")
    
    dataset_id = args.idx // len(methods)
    method_id = args.idx % len(methods)

    dataset_name = datasets[dataset_id]
    method_name = methods[method_id]

    run(dataset_name, method_name, n_samples=args.n_samples)
