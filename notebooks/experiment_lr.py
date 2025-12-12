
import pandas as pd
import numpy as np
import sys
sys.path.append("../")

from cfmining.algorithms import P2CE
from cfmining.predictors import MonotoneClassifier
from cfmining.baselines import MAPOCAM

from experiments_helper import get_data_model, run_experiments, format_df_table, summarize_results, get_action_set

import argparse


SEED = 0


def run(dataset_name):
    max_changes = 3
    objective = "abs_diff"
    dataset, X_train, Y_train, model, outlier_detection, individuals = get_data_model(dataset_name, "LogisticRegression")
    n = min(200, len(individuals))
    individuals = individuals.sample(n = n, random_state=SEED)
    outlier_detection.contamination = dataset.outlier_contamination
    action_set = get_action_set(dataset, X_train, default_step_size=0.05)

    model = MonotoneClassifier(model, outlier_detection, X = X_train)

    method = P2CE(
        action_set = action_set,
        classifier = model,
        compare = objective,
        max_changes = max_changes,
        outlier_contamination= dataset.outlier_contamination,
        estimate_outlier=True,
        time_limit=np.inf,
    )

    print("Running P2CE with ", dataset_name)

    run_experiments(
        method,
        individuals=individuals,
        model=model,
        output_file=f"../results/lr/{dataset}/p2ce_{objective}.csv"
    );

    # ablation without outlier estimation

    print("Running P2CE ablation without outlier estimation with ", dataset_name)

    method = P2CE(
        action_set = action_set,
        classifier = model,
        compare = objective,
        max_changes = max_changes,
        outlier_contamination= dataset.outlier_contamination,
        estimate_outlier=False,
        time_limit=np.inf,
    )

    run_experiments(
        method,
        individuals=individuals,
        model=model,
        output_file=f"../results/lr/{dataset}/p2ce_ablation_{objective}.csv"
    );


    print("Running MAPOCAM with ", dataset_name)

    for feat in action_set:
        feat.flip_direction = 1
        feat.update_grid()
    
    method = MAPOCAM(
        action_set,
        model,
        criteria = objective,
        max_changes=max_changes
    )

    run_experiments(
        method,
        individuals=individuals,
        model=model,
        output_file=f"../results/lr/{dataset}/mapocam_{objective}.csv"
    );



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=0,
        help="ID of the dataset to use (0: german, 1: taiwan, 2: adult)",
    )
    datasets = [
        "german",
        "taiwan",
        "adult"
    ]
    args = parser.parse_args()
    dataset_name = datasets[args.dataset_id]
    
    run(dataset_name)
