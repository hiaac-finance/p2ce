import copy
import pandas as pd
import numpy as np
import sys

sys.path.append("../")

from cfmining.algorithms import P2CE
from cfmining.predictors import (
    GeneralClassifier_Shap,
    GeneralClassifier,
    TreeClassifier,
)
from cfmining.action_set import ActionSet
from cfmining.baselines import Bruteforce, MAPOCAM, Nice, Dice
from cfmining.criteria import *

from experiments_helper import (
    get_data_model,
    run_experiments,
    format_df_table,
    summarize_results,
    get_action_set,
    summarize_results_multi,
)

import argparse

SEED = 0


def get_method(method_name, model, outlier_detection, X_train, Y_train, dataset):
    max_changes = 3

    if method_name == "p2ce_tree_abs_diff":
        action_set = get_action_set(dataset, X_train, default_step_size=0.05)
        model_wrap = GeneralClassifier_Shap(
            model,
            outlier_detection,
            X_train,
            shap_explainer="tree",
        )

        method = P2CE(
            action_set=action_set,
            classifier=model_wrap,
            compare="abs_diff",
            max_changes=max_changes,
            outlier_contamination=dataset.outlier_contamination,
            estimate_outlier=True,
            time_limit=np.inf,
        )

    elif method_name == "mapocam_tree_abs_diff":
        action_set = get_action_set(dataset, X_train, default_step_size=0.05)
        for feat in action_set:
            feat.flip_direction = 1
            feat.update_grid()
        model_wrap = TreeClassifier(
            classifier=model,
            X=X_train,
            y=Y_train,
            use_predict_max=True,
            clf_type="lightgbm",
        )

        method = MAPOCAM(
            action_set=action_set,
            model=model_wrap,
            criteria="abs_diff",
            max_changes=max_changes,
        )
    elif method_name == "nice":
        model_wrap = GeneralClassifier(
            model,
            outlier_detection,
            X_train,
        )

        method = Nice(
            X_train,
            Y_train,
            model=model,
            cat_features=dataset.categoric_features,
        )
    elif method_name == "dice":
        model_wrap = GeneralClassifier(
            model,
            outlier_detection,
            X_train,
        )

        method = Dice(
            X_train,
            Y_train,
            model,
            n_cfs=1,
            mutable_features=dataset.mutable_features,
            continuous_features=dataset.continuous_features,
        )
    elif method_name == "p2ce_tree_multi":
        action_set = get_action_set(dataset, X_train, default_step_size=0.05)

        model_wrap = GeneralClassifier_Shap(
            model,
            outlier_detection,
            X_train,
            shap_explainer="tree",
        )

        # setting multiple criteria
        range_calc = RangeCalculator(action_set)

        def compare_call(pivot):
            criteria_list = [
                MaxDistCriterion(
                    pivot,
                    range_calc,
                ),
                NumberChangesCriterion(pivot),
                AbsDiffCriterion(pivot, range_calc),
            ]
            return MultiCriterion(criteria_list, pivot)

        method = P2CE(
            action_set=action_set,
            classifier=model_wrap,
            compare=compare_call,
            max_changes=max_changes,
            outlier_contamination=dataset.outlier_contamination,
            estimate_outlier=True,
            time_limit=np.inf,
        )
    elif method_name == "mapocam_tree_multi":
        action_set = get_action_set(dataset, X_train, default_step_size=0.05)

        for feat in action_set:
            feat.flip_direction = 1
            feat.update_grid()

        model_wrap = TreeClassifier(
            classifier=model,
            X=X_train,
            y=Y_train,
            use_predict_max=True,
            clf_type="lightgbm",
        )

        # setting multiple criteria
        range_calc = RangeCalculator(action_set)

        def compare_call(pivot):
            criteria_list = [
                MaxDistCriterion(
                    pivot,
                    range_calc,
                ),
                NumberChangesCriterion(pivot),
                AbsDiffCriterion(pivot, range_calc),
            ]
            return MultiCriterion(criteria_list, pivot)

        method = MAPOCAM(
            action_set=action_set,
            model=model_wrap,
            criteria=compare_call,
            max_changes=max_changes,
        )

    elif method_name == "dice_multi":
        n_cfs = 4

        model_wrap = GeneralClassifier(
            model,
            outlier_detection,
            X_train,
        )

        method = Dice(
            X_train,
            Y_train,
            model,
            n_cfs=n_cfs,
            mutable_features=dataset.mutable_features,
            continuous_features=dataset.continuous_features,
        )

    return method, model_wrap


def run(dataset_name, method_name, n_samples=500):
    # load dataset
    dataset, X_train, Y_train, model, outlier_detection, individuals = get_data_model(
        dataset_name, "LGBMClassifier_simple"
    )
    n = min(n_samples, len(individuals))
    individuals = individuals.sample(n=n, random_state=SEED)
    outlier_detection.contamination = dataset.outlier_contamination

    # get method
    method, model_wrap = get_method(
        method_name, model, outlier_detection, X_train, Y_train, dataset
    )

    print(f"Running {method_name} with ", dataset_name)
    # run experiments
    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lgbm/{dataset}/{method_name}.csv",
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
    datasets = ["german", "taiwan", "adult", "acsincome", "homecredit"]
    methods = [
        "p2ce_tree_abs_diff",
        "mapocam_tree_abs_diff",
        "nice",
        "dice",
        "p2ce_tree_multi",
        "mapocam_tree_multi",
        "dice_multi",
    ]
    args = parser.parse_args()
    dataset_id = args.idx // len(methods)
    method_id = args.idx % len(methods)

    dataset_name = datasets[dataset_id]
    method_name = methods[method_id]

    run(dataset_name, method_name, n_samples=args.n_samples)