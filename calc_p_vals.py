# %%
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

PAIRS = [
    # ['bundle/runs/baseline_dice_brats_2021_high/inference_results/mean_dice_raw.csv', 'bundle/runs/hardl1ace_dice_brats_2021_high/inference_results/mean_dice_raw.csv'],
    # ['bundle/runs/baseline_ce_brats_2021_high/inference_results/mean_dice_raw.csv', 'bundle/runs/hardl1ace_ce_brats_2021_high/inference_results/mean_dice_raw.csv'],
    # ['bundle/runs/baseline_dice_ce_brats_2021_high/inference_results/mean_dice_raw.csv', 'bundle/runs/hardl1ace_dice_ce_brats_2021_high/inference_results/mean_dice_raw.csv'],
    # ['bundle/runs/baseline_dice_brats_2021_high/inference_results/ace_raw.csv', 'bundle/runs/hardl1ace_dice_brats_2021_high/inference_results/ace_raw.csv'],
    # ['bundle/runs/baseline_dice_brats_2021_high/inference_results/mce_raw.csv', 'bundle/runs/hardl1ace_dice_brats_2021_high/inference_results/mce_raw.csv'],
    [
        "bundle/runs/baseline_dice_brats_2021_high/inference_results/ece_raw.csv",
        "bundle/runs/hardl1ace_dice_brats_2021_high/inference_results/ece_raw.csv",
    ],
    # ['bundle/runs/baseline_ce_brats_2021_high/inference_results/mean_dice_raw.csv', 'bundle/runs/baseline_ce_brats_2021_high_temp_scaled/inference_results/mean_dice_raw.csv'],
    # ['bundle/runs/baseline_dice_ce_brats_2021_high/inference_results/mean_dice_raw.csv', 'bundle/runs/baseline_dice_ce_brats_2021_high_temp_scaled/inference_results/mean_dice_raw.csv'],
    # ['bundle/runs/baseline_dice_brats_2021_high/inference_results/mean_dice_raw.csv', 'bundle/runs/baseline_dice_ce_brats_2021_high_temp_scaled/inference_results/mean_dice_raw.csv'],
    # ['bundle/runs/hardl1ace_dice_brats_2021_high/inference_results_b10/ace_raw.csv', 'bundle/runs/hardl1ace_dice_brats_2021_high/inference_results_b100/ace_raw.csv'],
    #
]


def calc_p_values(pair):
    df1 = pd.read_csv(pair[0])
    df2 = pd.read_csv(pair[1])

    # Columns for comparison
    columns = ["class0", "class1", "class2", "mean"]

    # Calculate p-values and plot histograms
    for col in columns:
        t_stat, p_value = ttest_rel(df1[col].dropna(), df2[col].dropna())

        print(f"--- Metrics for {col} ---")
        print(f"t-statistic: {t_stat}")
        print(f"p-value: {p_value}")

        # Plot histograms
        plt.figure()
        plt.hist(df1[col], alpha=0.5, label=pair[1])
        plt.hist(df2[col], alpha=0.5, label=pair[0])
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        # save fig:
        plt.savefig("hist.png")
        plt.close()


# %%
if __name__ == "__main__":
    plt.ion()
    for pair in PAIRS:
        calc_p_values(pair)
        stop = "here"
