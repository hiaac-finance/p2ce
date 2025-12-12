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


def run(dataset_name, n_samples=200):
    SEED = 0

    max_changes = 3
    objective = "abs_diff"

    dataset, X_train, Y_train, model, outlier_detection, individuals = get_data_model(
        dataset_name, "LGBMClassifier_simple"
    )
    n = min(n_samples, len(individuals))
    individuals = individuals.sample(n=n, random_state=SEED)
    outlier_detection.contamination = dataset.outlier_contamination
    action_set = get_action_set(dataset, X_train, default_step_size=0.05)

    print("Running P2CE with ", dataset_name)

    model_wrap = GeneralClassifier_Shap(
        model,
        outlier_detection,
        X_train,
        shap_explainer="tree",
    )

    method = P2CE(
        action_set=action_set,
        classifier=model_wrap,
        compare=objective,
        max_changes=max_changes,
        outlier_contamination=dataset.outlier_contamination,
        estimate_outlier=True,
        time_limit=np.inf,
    )

    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lgbm/{dataset}/p2ce_tree_{objective}.csv",
    )

    print("Running MAPOCAM with ", dataset_name)

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
        criteria=objective,
        max_changes=max_changes,
    )

    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lgbm/{dataset_name}/mapocam_tree_{objective}.csv",
    )

    print("Running NICE with ", dataset_name)

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

    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lgbm/{dataset_name}/nice.csv",
    )

    print("Running DICE with ", dataset_name)

    method = Dice(
        X_train,
        Y_train,
        model,
        n_cfs=1,
        mutable_features=dataset.mutable_features,
        continuous_features=dataset.continuous_features,
    )

    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lgbm/{dataset_name}/dice.csv",
    )

    print("Running P2CE with multiple objectives with ", dataset_name)

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

    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lgbm/{dataset}/p2ce_tree_multi.csv",
    )

    print("Running MAPOCAM with multiple objectives with ", dataset_name)

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

    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lgbm/{dataset_name}/mapocam_tree_multi.csv",
    )

    print("Running DICE with multiple objectives with ", dataset_name)

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

    run_experiments(
        method,
        individuals=individuals,
        model=model_wrap,
        output_file=f"../results/lgbm/{dataset_name}/dice_multi.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=0,
        help="ID of the dataset to use (0: german, 1: taiwan, 2: adult)",
    )
    datasets = ["german", "taiwan", "adult"]
    args = parser.parse_args()
    dataset_name = datasets[args.dataset_id]

    run(dataset_name)
