"""Recompute reported metrics and post-hoc diagnostics from the committed CSVs.

This script never fits a hardness predictor or chooses model parameters.
Run with --check to validate the saved diagnostics or --write to regenerate them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]


def compute_diagnostics():
    features = pd.read_csv(ROOT / "data/MASC_520_Person3_Features.csv")
    train = features.loc[features["split"].eq("train")].copy()
    test = features.loc[features["split"].eq("test")].copy()
    predictions = pd.read_csv(ROOT / "results/final_test_predictions.csv")
    metrics = pd.read_csv(ROOT / "results/final_test_metrics.csv")
    errors = pd.read_csv(ROOT / "results/per_image_errors.csv")
    assert len(train) == 76 and len(test) == 8
    assert train["condition_group"].nunique() == 28
    assert not features["filename"].duplicated().any()
    assert not predictions["filename"].duplicated().any()
    assert set(predictions["filename"]) == set(test["filename"])
    assert not set(train["condition_group"]) & set(test["condition_group"])
    reference = test.set_index("filename").loc[predictions["filename"]]
    np.testing.assert_allclose(predictions["hardness"], reference["hardness"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(predictions["magnification"], reference["magnification"])
    assert features["hardness_method"].eq("Shore D (durometer)").all()

    y = predictions["hardness"].to_numpy(dtype=float)
    tpu = predictions["filename"].str.contains("_tpu_", regex=False).to_numpy()
    assert tpu.sum() == 2
    sst = float(np.square(y - y.mean()).sum())
    assert sst > 0
    rows = []
    for item in metrics.to_dict(orient="records"):
        name = item["model"]
        pred = predictions[name].to_numpy(dtype=float)
        assert np.isfinite(pred).all()
        ae = np.abs(pred - y)
        se = np.square(pred - y)
        mae = float(ae.mean())
        rmse = float(np.sqrt(se.mean()))
        r2 = float(1 - se.sum() / sst)
        np.testing.assert_allclose([mae, rmse, r2], [item["MAE"], item["RMSE"], item["R2"]], rtol=1e-11, atol=1e-11)
        saved_errors = errors.loc[errors["model"].eq(name)].set_index("filename").loc[predictions["filename"]]
        np.testing.assert_allclose(saved_errors["absolute_error"], ae, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(saved_errors["squared_error"], se, rtol=1e-11, atol=1e-11)
        rows.append({
            "model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
            "squared_error_sum": float(se.sum()),
            "TPU_squared_error_sum": float(se[tpu].sum()),
            "TPU_squared_error_share_percent": float(100 * se[tpu].sum() / se.sum()),
            "TPU_MAE": float(ae[tpu].mean()),
            "other_six_images_MAE": float(ae[~tpu].mean()),
        })

    train_min, train_max = float(train["hardness"].min()), float(train["hardness"].max())
    # Optimistic bound for any predictions restricted to the training target range.
    # Standard direct-target RF leaf means and forest averages satisfy that range.
    closest_allowed = np.clip(y, train_min, train_max)
    smallest_possible_sse = float(np.square(closest_allowed - y).sum())
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[["magnification"]])
    test_encoding = encoder.transform(predictions[["magnification"]])
    cv = pd.read_csv(ROOT / "results/training_nested_group_cv_metrics.csv").set_index("model")
    selection = json.loads((ROOT / "results/selection_before_test.json").read_text())
    selected = str(cv["MAE"].idxmin())
    assert selected == "RF baseline"
    # The selection record is retained in its original schema.
    assert selected in json.dumps(selection)

    summary = {
        "scope": "Post-hoc diagnostics of the existing fixed test predictions. No model fitting or parameter selection.",
        "train_images": len(train), "test_images": len(test),
        "train_condition_proxy_groups": int(train["condition_group"].nunique()),
        "test_material_conditions": int(test["condition_group"].nunique()),
        "train_hardness_range_Shore_D": [train_min, train_max],
        "test_hardness_range_Shore_D": [float(y.min()), float(y.max())],
        "test_hardness_mean_Shore_D": float(y.mean()),
        "R2_reference": "Mean of the evaluated test targets, distinct from the training-mean predictor.",
        "test_total_sum_of_squares": sst,
        "selected_by_training_cv_MAE": selected,
        "ridge_CV_MAE_reduction_percent": float(100 * (cv.loc["Ridge fixed", "MAE"] - cv.loc["Ridge tuned", "MAE"]) / cv.loc["Ridge fixed", "MAE"]),
        "rf_CV_MAE_increase_percent": float(100 * (cv.loc["RF tuned", "MAE"] - cv.loc["RF baseline", "MAE"]) / cv.loc["RF baseline", "MAE"]),
        "current_standard_random_forest_bound": {
            "assumption": "Direct-target RandomForestRegressor with ordinary nonnegative leaf-mean and tree-mean averaging, as used in this experiment.",
            "smallest_possible_squared_error_sum_if_other_predictions_ideal": smallest_possible_sse,
            "test_R2_upper_bound": float(1 - smallest_possible_sse / sst),
            "interpretation": "An optimistic mathematical ceiling for this RF setup, not an achieved score and not a bound for Ridge or CNN.",
        },
        "magnification_encoding": {
            "training_categories": [int(v) for v in encoder.categories_[0]],
            "unseen_test_categories": sorted(int(v) for v in set(test["magnification"]) - set(train["magnification"])),
            "method": "Training-fitted OneHotEncoder(handle_unknown='ignore').",
            "test_encoding_shape": list(test_encoding.shape),
            "all_test_magnification_codes_zero": bool((test_encoding == 0).all()),
            "interpretation": "Both unseen magnifications have the same magnification code. Their separate effect on prediction error was not isolated.",
        },
        "per_model": rows,
        "limitations": [
            "Condition groups are proxies, not verified physical specimen IDs.",
            "The fixed eight-image test set was previously evaluated and is not a new blind test.",
            "CNN predictions are historical values rounded to three decimals, with a different original preprocessing protocol and no new CNN training.",
            "TPU error contribution is descriptive and does not establish the unique cause of poor generalization.",
            "Raw image-to-label pairing, units and supplied image hashes were not independently revalidated against all original source images in this cached-feature run.",
        ],
    }
    return summary, pd.DataFrame(rows)


def compare_nested(actual, expected, path="root"):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            compare_nested(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_nested(a, e, f"{path}[{i}]")
    elif isinstance(actual, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, err_msg=path)
    else:
        assert actual == expected, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="Regenerate saved post-hoc JSON and CSV, without fitting models.")
    group.add_argument("--check", action="store_true", help="Validate the committed post-hoc JSON and CSV against source results.")
    args = parser.parse_args()
    summary, frame = compute_diagnostics()
    json_path = ROOT / "results/posthoc_diagnostics.json"
    csv_path = ROOT / "results/posthoc_diagnostics.csv"
    if args.write:
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        frame.to_csv(csv_path, index=False)
    elif args.check:
        compare_nested(summary, json.loads(json_path.read_text()))
        pd.testing.assert_frame_equal(frame, pd.read_csv(csv_path), check_exact=False, rtol=1e-11, atol=1e-11)
    print("Validated all 7 test metric rows and all 56 per-image error records.")
    print("CNN TPU squared-error share: %.4f%%" % frame.set_index("model").loc["CNN historical", "TPU_squared_error_share_percent"])
    print("Current standard RF test R2 ceiling: %.8f" % summary["current_standard_random_forest_bound"]["test_R2_upper_bound"])
    print("Both unseen test magnifications encode to all zeros:", summary["magnification_encoding"]["all_test_magnification_codes_zero"])


if __name__ == "__main__":
    main()
