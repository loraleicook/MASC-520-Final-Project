#!/usr/bin/env python3
"""Regenerate the Person 3 figures from the supplied, fixed result tables.

This script draws figures only. It does not fit models, select parameters,
change the test split, or retrain the historical CNN.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BLUE = "#5D7F98"
GREEN = "#2D7A62"
GRAY = "#777D82"
CARDINAL = "#BE1C33"
GOLD = "#C49600"
INK = "#20272C"
GRID = "#E0E5E8"
SELECTED_MODEL = "RF baseline"
HISTORICAL_MODEL = "CNN historical"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#87949C",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"{stem}.{suffix}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.16,
        )
    plt.close(fig)


def short_label(filename: str) -> str:
    match = re.search(r"esd_(pla|abs|tpu|onyx)_(\d+)x", str(filename), re.I)
    if not match:
        raise ValueError(f"Unrecognized test filename: {filename!r}")
    material, magnification = match.groups()
    material = "Onyx" if material.lower() == "onyx" else material.upper()
    return f"{material}\n{magnification}×"


def model_color(model: str) -> str:
    if model == SELECTED_MODEL:
        return GREEN
    if model == HISTORICAL_MODEL:
        return GRAY
    return BLUE


def figure_cv(cv: pd.DataFrame) -> None:
    frame = cv.sort_values("MAE", kind="stable").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(
        np.arange(len(frame)),
        frame["MAE"],
        color=[model_color(x) for x in frame["model"]],
        height=0.66,
    )
    ax.set_yticks(np.arange(len(frame)), frame["model"])
    ax.invert_yaxis()
    ax.set_xlim(0, float(frame["MAE"].max()) * 1.19)
    ax.set_xlabel("Outer-CV MAE (Shore D), lower is better")
    ax.set_title("Training validation across 28 condition proxy groups", pad=13)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, frame["MAE"]):
        ax.text(
            value + 0.045,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    fig.text(
        0.02,
        0.015,
        "76 training images. Nested CV: 5 outer / 3 inner folds. "
        "Green marks the model selected by training-CV MAE.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "01_training_cv_mae")


def figure_test_metrics(metrics: pd.DataFrame) -> None:
    models = metrics["model"].tolist()
    y = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.9), sharey=True)
    for ax, metric in zip(axes, ("MAE", "RMSE", "R2")):
        values = metrics[metric].to_numpy(dtype=float)
        colors = (
            [CARDINAL if value < 0 else BLUE for value in values]
            if metric == "R2"
            else [model_color(model) for model in models]
        )
        bars = ax.barh(y, values, height=0.64, color=colors, alpha=0.94)
        ax.set_yticks(y, models)
        ax.set_ylim(len(models) - 0.4, -0.7)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if metric == "R2":
            ax.set_title("Test R², higher is better", fontsize=12)
            ax.set_xlim(min(-0.96, values.min() - 0.19), 0.035)
            ax.axvline(0, color="#87949C", linewidth=0.8)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.set_xlabel("Test mean defines the R² reference")
            for bar, value in zip(bars, values):
                ax.text(
                    value - 0.022,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    va="center",
                    ha="right",
                    color=CARDINAL,
                    fontsize=10,
                    fontweight="bold",
                )
        else:
            ax.set_title(f"Test {metric}, lower is better", fontsize=12)
            ax.set_xlim(0, values.max() * 1.23)
            ax.set_xlabel("Shore D")
            for bar, value in zip(bars, values):
                ax.text(
                    value + values.max() * 0.018,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    va="center",
                    fontsize=10,
                )
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    fig.suptitle("Fixed test set: 8 images from 4 material conditions", y=0.99)
    fig.legend(
        handles=[
            Patch(facecolor=GREEN, label="Selected by training CV"),
            Patch(facecolor=GRAY, label="Historical CNN"),
            Patch(facecolor=CARDINAL, label="Negative test R²"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.57, 0.94),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.02,
        0.015,
        "CNN uses saved predictions rounded to 3 decimals and a different preprocessing protocol. "
        "No CNN retraining. Test data were previously evaluated.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.82, bottom=0.15, wspace=0.19)
    save(fig, "02_final_test_metrics")


def figure_parity(predictions: pd.DataFrame, models: list[str]) -> None:
    styles = {
        "Training mean": ("#8B8B8B", "s"),
        "Linear regression": ("#75538D", "P"),
        "Ridge fixed": ("#4A7C99", "o"),
        "RF baseline": (GREEN, "D"),
        "Ridge tuned": ("#B57629", "v"),
        "RF tuned": ("#306251", "X"),
        "CNN historical": (GRAY, "^"),
    }
    measured = predictions["hardness"].to_numpy(dtype=float)
    all_values = np.concatenate(
        [measured] + [predictions[model].to_numpy(dtype=float) for model in models]
    )
    lo = min(30.0, float(np.floor(all_values.min() / 5) * 5))
    hi = max(85.0, float(np.ceil(all_values.max() / 5) * 5))
    fig, ax = plt.subplots(figsize=(7.8, 7.5))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#A0A6AA", linewidth=1.1)
    for model in models:
        color, marker = styles[model]
        ax.scatter(
            measured,
            predictions[model],
            label=model,
            color=color,
            marker=marker,
            s=58,
            linewidths=0.7,
            edgecolors="white",
            alpha=0.93,
            zorder=3,
        )
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured hardness (Shore D)")
    ax.set_ylabel("Predicted hardness (Shore D)")
    ax.set_title("Measured vs. predicted hardness on the fixed test set", pad=13)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
    )
    fig.text(
        0.05,
        0.02,
        "Dashed line: perfect prediction. All 8 images per model are shown.\n"
        "Coincident points overlap. CNN predictions are historical, without retraining.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.31)
    save(fig, "03_measured_vs_predicted")


def figure_errors(
    errors: pd.DataFrame, predictions: pd.DataFrame, models: list[str]
) -> None:
    filenames = predictions["filename"].tolist()
    if errors.duplicated(["model", "filename"]).any():
        raise ValueError("Duplicate model/image rows in per_image_errors.csv")
    matrix = errors.pivot(index="model", columns="filename", values="absolute_error")
    matrix = matrix.reindex(index=models, columns=filenames)
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite per-image errors")
    maximum = float(np.ceil(values.max() / 5) * 5)
    fig, ax = plt.subplots(figsize=(11.3, 5.6))
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=maximum)
    ax.set_xticks(np.arange(len(filenames)), [short_label(x) for x in filenames])
    ax.set_yticks(np.arange(len(models)), models)
    ax.tick_params(length=0, pad=7)
    ax.set_title("Absolute error for every test image (Shore D)", pad=16)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value >= maximum * 0.55 else INK,
                fontsize=10,
                fontweight="bold" if value >= maximum * 0.55 else "normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.033, pad=0.025)
    colorbar.set_label("Absolute error (Shore D)")
    fig.text(
        0.02,
        0.015,
        "Every test image is retained. CNN uses historical predictions and a different preprocessing protocol.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.19, right=0.95, top=0.87, bottom=0.17)
    save(fig, "04_per_image_absolute_error")


def figure_coverage(features: pd.DataFrame) -> None:
    train = features.loc[features["split"].eq("train"), "hardness"].to_numpy(dtype=float)
    test = features.loc[features["split"].eq("test"), "hardness"].to_numpy(dtype=float)
    if (len(train), len(test)) != (76, 8):
        raise ValueError("Expected the documented fixed split of 76 train / 8 test images")
    fig, ax = plt.subplots(figsize=(9.7, 4.0))
    ax.scatter(train, np.ones(len(train)), color=BLUE, s=57, alpha=0.73, zorder=3)
    ax.scatter(test, np.zeros(len(test)), color=CARDINAL, marker="^", s=75, zorder=4)
    ax.set_yticks([1, 0], [f"Training ({len(train)} images)", f"Test ({len(test)} images)"])
    ax.set(xlim=(30, 85), ylim=(-0.45, 1.5))
    ax.set_xticks([30, 40, 50, 60, 70, 80])
    ax.set_xlabel("Hardness (Shore D)")
    ax.set_title("Hardness coverage differs across the fixed split", pad=18)
    ax.text(
        (train.min() + train.max()) / 2,
        1.2,
        f"Training range: {train.min():.2f}–{train.max():.2f}",
        ha="center",
        color=BLUE,
        fontsize=11,
        fontweight="bold",
    )
    ax.annotate(
        f"TPU {test.min():.2f}",
        xy=(float(test.min()), 0),
        xytext=(0, 15),
        textcoords="offset points",
        ha="center",
        color=CARDINAL,
        fontweight="bold",
        fontsize=11,
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    fig.text(
        0.02,
        0.02,
        "Training: 60× / 500×. Test: 250× / 1000×. All 84 image labels are plotted without jitter.\n"
        "Identical labels overlap. Counts refer to images, not verified independent physical specimens.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.21, right=0.98, top=0.80, bottom=0.29)
    save(fig, "05_hardness_coverage")


def figure_original(original: pd.DataFrame) -> None:
    if len(original) != 8 or original["filename"].duplicated().any():
        raise ValueError("Expected 8 unique original model prediction rows")
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    x = np.arange(len(original))
    styles = [
        ("hardness", "Measured", "#111111", "D", 66),
        ("Ridge original", "Ridge", "#C33342", "o", 55),
        ("RF original", "Random forest", GOLD, "s", 53),
        ("CNN historical", "CNN", GREEN, "^", 66),
    ]
    for column, label, color, marker, size in styles:
        ax.scatter(
            x,
            original[column],
            label=label,
            color=color,
            marker=marker,
            s=size,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    ax.set_xticks(x, [short_label(name) for name in original["filename"]])
    ax.set(xlim=(-0.45, len(original) - 0.55), ylim=(0, 100))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Hardness (Shore D)")
    ax.set_title("Original model comparison: saved predictions", pad=38)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        borderaxespad=0,
    )
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(
        0.02,
        0.015,
        "Original Notebook outputs rounded to 3 decimals. "
        "These Ridge/RF predictions use the original protocol, separate from the new controlled baselines.\n"
        "Eight images represent four material conditions at two magnifications. CNN was not retrained.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.78, bottom=0.20)
    save(fig, "06_original_model_predictions")


def main() -> None:
    cv = read_csv("training_nested_group_cv_metrics.csv")
    metrics = read_csv("final_test_metrics.csv")
    predictions = read_csv("final_test_predictions.csv")
    errors = read_csv("per_image_errors.csv")
    features = pd.read_csv(ROOT / "data" / "MASC_520_Person3_Features.csv")
    original = read_csv("original_model_predictions.csv")
    models = metrics["model"].tolist()
    if len(cv) != 6 or len(metrics) != 7 or len(predictions) != 8:
        raise ValueError("Unexpected number of methods or test images")
    if not set(models).issubset(predictions.columns):
        raise ValueError("A test-model prediction column is missing")
    figure_cv(cv)
    figure_test_metrics(metrics)
    figure_parity(predictions, models)
    figure_errors(errors, predictions, models)
    figure_coverage(features)
    figure_original(original)
    print(f"Generated 6 PNG/PDF figure pairs in {FIGURES}")


if __name__ == "__main__":
    main()
