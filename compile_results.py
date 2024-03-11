import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib.patches import Patch
import numpy as np

PRECISION = 2
TH = 0.01  # Threshold for rounding values less than 0.1 to scientific notation
EXPONENT = -4  # Exponent for scientific notation

pd.options.display.float_format = "{:,.4f}".format


RUNS_DIR = "./bundle/runs"
OUT_DIR = os.path.join(RUNS_DIR, "results_summary")

COMPONENTS = {
    "ET": "Enhacing Tumor",
    "TC": "Tumor Core",
    "WT": "Whole Tumor",
    "mean": "Average",
}

COMPONENTS_RAW = {  # as the names appear in the raw csv files
    "class0": "Enhacing Tumor",
    "class1": "Tumor Core",
    "class2": "Whole Tumor",
    "mean": "Average",
}

METRICS = {
    "mean_dice": "Dice Score",
    "ece": "Expected Calibration Error",
    "ace": "Average Calibration Error",
    "mce": "Maximum Calibration Error",
}


RUNS = {
    "baseline_ce_brats_2021_high": "CE",
    "hardl1ace_ce_brats_2021_high": "CE + hL1-ACE",
    "baseline_dice_brats_2021_high": "Dice",
    "hardl1ace_dice_brats_2021_high": "Dice + hL1-ACE",
    "baseline_dice_ce_brats_2021_high": "Dice + CE",
    "hardl1ace_dice_ce_brats_2021_high": "Dice + CE + hL1-ACE",
}

RUNS_TEMP_SCALE = {key + "_temp_scaled": value + " + Ts" for key, value in RUNS.items()}

RUNS_SCATTER = {
    "baseline_ce_brats_2021_high": "baseline",
    "baseline_dice_brats_2021_high": "baseline",
    "baseline_dice_ce_brats_2021_high": "baseline",
    "hardl1ace_ce_brats_2021_high": "hL1-ACE",
    "hardl1ace_dice_brats_2021_high": "hL1-ACE",
    "hardl1ace_dice_ce_brats_2021_high": "hL1-ACE",
}


def format_value(value, precision, threshold, exponent):
    """Custom format function to display values in scientific notation with a fixed exponent."""
    if value < threshold:
        # Convert value to scientific notation with the fixed exponent
        value_scaled = value * (10**-exponent)
        return f"{value_scaled:.{precision}f}"
    else:
        # Standard decimal format
        return f"{value:.{precision}f}"


def set_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": True,  # Use LaTeX to write all text
            "font.family": "serif",
            "font.serif": ["Times"],  # or another LaTeX-like serif font
            "font.size": 10,  # Match the font size used in the document
            "axes.labelsize": 10,  # Size of axis labels
            "axes.titlesize": 10,  # Size of the title
            "xtick.labelsize": 10,  # Size of the tick labels
            "ytick.labelsize": 10,  # Size of the tick labels
            "legend.fontsize": 10,  # Size of the legend
            "figure.figsize": [
                4.8,
                3.0,
            ],  # Adjust figure size to match text width (12.2cm converted to inches)
        }
    )


def _create_box_plot_subplot(ax, df_box, metric, component, run_names):
    sns.set_theme(style="whitegrid")
    df_box = df_box.melt(var_name="run", value_name=metric)
    sns.boxplot(
        x="run",
        y=metric,
        data=df_box,
        ax=ax,
        notch=True,
        hue="run",
        palette="Set2",
        dodge=False,  # Ensure boxes are side-by-side for each run
    )

    ax.set_xticks(range(len(run_names)))
    ax.set_xticklabels(run_names, rotation=45, fontsize="x-small", ha="right")
    ax.set_xlabel("Loss Function")
    ax.set_ylabel(METRICS[metric])
    ax.set_title(COMPONENTS[component])
    ax.legend([], [], frameon=False)  # Hide the legend


def create_box_plots():
    run_names = list(RUNS.values())
    num_components = len(COMPONENTS)

    for metric in METRICS:
        fig, axs = plt.subplots(
            1, num_components, figsize=(20, 5), sharey=True
        )  # Adjust figsize as needed

        df_box = pd.DataFrame()
        for run, run_name in RUNS.items():
            df_raw = pd.read_csv(
                os.path.join(RUNS_DIR, f"{run}/inference_results/{metric}_raw.csv")
            )
            df_box[run_name] = df_raw["mean"].values

        for i, (component, component_name) in enumerate(COMPONENTS.items()):
            _create_box_plot_subplot(axs[i], df_box, metric, component, run_names)

        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/{metric}.pdf")
        plt.close(fig)


def save_metric_summary_csv(print_table=True):
    # Ensure OUT_DIR exists
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for metric in METRICS:
        # Initialize df_out with string data type
        df_out = pd.DataFrame(
            index=RUNS.values(), columns=COMPONENTS.values(), dtype="object"
        )

        for run, run_name in RUNS.items():
            df_raw = pd.read_csv(
                os.path.join(RUNS_DIR, f"{run}/inference_results/{metric}_raw.csv")
            )
            raw_mean = df_raw.mean(numeric_only=True)
            raw_std = df_raw.std(numeric_only=True)

            # Loop through COMPONENTS to set "mean ± std" in df_out
            for comp in COMPONENTS_RAW.keys():
                if comp in raw_mean and comp in raw_std:
                    mean = raw_mean[comp]
                    std = raw_std[comp]
                    mean_format = format_value(mean, PRECISION, TH, EXPONENT)
                    std_format = format_value(std, PRECISION, TH, EXPONENT)
                    df_out.at[run_name, COMPONENTS_RAW[comp]] = (
                        f"{mean_format} ± {std_format}"
                    )

        # Save to CSV
        df_out.to_csv(f"{OUT_DIR}/{metric}_summary.csv")

        # Print table if required
        if print_table:
            print(f"Metric: {METRICS[metric]}")
            print(tabulate(df_out, headers="keys", tablefmt="pipe", showindex=True))
            print("\nLaTeX version:")
            print(
                df_out.to_latex(
                    index=True, caption=METRICS[metric], label=f"tab:{metric}_summary"
                )
            )


def save_metric_summary_t_scale_csv(print_table=True):
    # Ensure OUT_DIR exists
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for metric in METRICS:
        # Initialize df_out with string data type
        df_out = pd.DataFrame(
            index=RUNS_TEMP_SCALE.values(), columns=COMPONENTS.values(), dtype="object"
        )

        for run, run_name in RUNS_TEMP_SCALE.items():
            df_raw = pd.read_csv(
                os.path.join(RUNS_DIR, f"{run}/inference_results/{metric}_raw.csv")
            )
            raw_mean = df_raw.mean(numeric_only=True)
            raw_std = df_raw.std(numeric_only=True)

            # Loop through COMPONENTS to set "mean ± std" in df_out
            for comp in COMPONENTS_RAW.keys():
                if comp in raw_mean and comp in raw_std:
                    mean = raw_mean[comp]
                    std = raw_std[comp]
                    mean_format = format_value(mean, PRECISION, TH, EXPONENT)
                    std_format = format_value(std, PRECISION, TH, EXPONENT)
                    df_out.at[run_name, COMPONENTS_RAW[comp]] = (
                        f"{mean_format} ± {std_format}"
                    )

        # Save to CSV
        df_out.to_csv(f"{OUT_DIR}/{metric}_summary.csv")

        # Print table if required
        if print_table:
            print(f"Metric: {METRICS[metric]}")
            print(tabulate(df_out, headers="keys", tablefmt="pipe", showindex=True))
            print("\nLaTeX version:")
            print(
                df_out.to_latex(
                    index=True, caption=METRICS[metric], label=f"tab:{metric}_summary"
                )
            )


def create_scatter_plots(
    metric_x="mean_dice", metric_y="ace", pattern=None, suffix="all"
):
    sns.set_theme(style="whitegrid")
    sns.set_palette("Set2")

    num_components = len(COMPONENTS_RAW)
    fig, axs = plt.subplots(
        1,
        num_components,
        figsize=(4.8 * num_components, 3.0),
        sharex=False,
        sharey=False,
    )

    filtered_runs = {
        k: v for k, v in RUNS_SCATTER.items() if not pattern or re.search(pattern, k)
    }

    if num_components == 1:
        axs = [axs]

    for i, (comp_key, comp_name) in enumerate(COMPONENTS_RAW.items()):
        for run, run_name in filtered_runs.items():
            df_x = pd.read_csv(
                os.path.join(RUNS_DIR, f"{run}/inference_results/{metric_x}_raw.csv")
            )
            df_y = pd.read_csv(
                os.path.join(RUNS_DIR, f"{run}/inference_results/{metric_y}_raw.csv")
            )

            if comp_key in df_x.columns and comp_key in df_y.columns:
                # Calculate the mean and standard deviation
                x_mean = 1 - df_y[comp_key].mean()
                y_mean = df_x[comp_key].mean()
                x_err = df_y[comp_key].std()
                y_err = df_x[comp_key].std()

                # Plot the mean with error bars
                ax = axs[i]
                ax.errorbar(
                    x_mean,
                    y_mean,
                    xerr=x_err,
                    yerr=y_err,
                    fmt="o",
                    label=run_name,
                    capsize=5,
                )

        ax.set_title(comp_name)
        ax.set_xlabel("1 - " + METRICS[metric_y])
        ax.set_ylabel(METRICS[metric_x])
        ax.legend(loc="lower left")

    # plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{metric_x}_vs_{metric_y}_{suffix}.pdf")
    plt.close()


if __name__ == "__main__":

    out_dir = "./bundle/runs/results_summary"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    set_plot_style()

    # create_box_plots()

    save_metric_summary_csv()
    save_metric_summary_t_scale_csv()

    # Plot CE runs:
    create_scatter_plots(
        pattern=r"(?<!_dice)_ce", suffix="ce", metric_x="mean_dice", metric_y="ace"
    )
    create_scatter_plots(
        pattern=r"(?<!_dice)_ce", suffix="ce", metric_x="mean_dice", metric_y="mce"
    )
    create_scatter_plots(
        pattern=r"(?<!_dice)_ce", suffix="ce", metric_x="mean_dice", metric_y="ece"
    )

    # Plot Dice runs:
    create_scatter_plots(
        pattern=r"_dice_(?!ce)", suffix="dice", metric_x="mean_dice", metric_y="ace"
    )
    create_scatter_plots(
        pattern=r"_dice_(?!ce)", suffix="dice", metric_x="mean_dice", metric_y="mce"
    )
    create_scatter_plots(
        pattern=r"_dice_(?!ce)", suffix="dice", metric_x="mean_dice", metric_y="ece"
    )

    # Plot Dice + CE runs:
    create_scatter_plots(
        pattern=r"_dice_ce", suffix="dice_ce", metric_x="mean_dice", metric_y="ace"
    )
    create_scatter_plots(
        pattern=r"_dice_ce", suffix="dice_ce", metric_x="mean_dice", metric_y="mce"
    )
    create_scatter_plots(
        pattern=r"_dice_ce", suffix="dice_ce", metric_x="mean_dice", metric_y="ece"
    )
