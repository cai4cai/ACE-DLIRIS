import pandas as pd
import matplotlib.pyplot as plt


PATHS = {
    5: "./bundle/runs/hardl1ace_dice_brats_2021_high/inference_results_b5/metrics.csv",
    10: "./bundle/runs/hardl1ace_dice_brats_2021_high/inference_results_b10/metrics.csv",
    20: "./bundle/runs/hardl1ace_dice_brats_2021_high/inference_results/metrics.csv",  # default value
    50: "./bundle/runs/hardl1ace_dice_brats_2021_high/inference_results_b50/metrics.csv",
    100: "./bundle/runs/hardl1ace_dice_brats_2021_high/inference_results_b100/metrics.csv",
    1000: "./bundle/runs/hardl1ace_dice_brats_2021_high/inference_results_b1000/metrics.csv",
}


def set_rc_params():
    # Apply rc_params for styling
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 10,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "figure.figsize": [4.8, 1.2],
        }
    )


def load_data():
    data = {"num_bins": [5, 10, 20, 50, 100, 1000], "ACE": [], "MCE": [], "ECE": []}

    for num_bins in data["num_bins"]:
        # Read the CSV file for the current number of bins
        df = pd.read_csv(PATHS[num_bins])

        # column labels don't make much sense in this CSV
        # Extract the values for ACE, MCE, and ECE
        ace_value = df[df["mean_dice"] == "ace"].iloc[0, 1]
        mce_value = df[df["mean_dice"] == "mce"].iloc[0, 1]
        ece_value = df[df["mean_dice"] == "ece"].iloc[0, 1]

        # Append the extracted values to the respective lists in the data dictionary
        data["ACE"].append(ace_value)
        data["MCE"].append(mce_value)
        data["ECE"].append(ece_value)

    return data


def make_plot(data):
    # Adjust ECE values by multiplying by 10^3 and plot all metrics on the same y-axis
    fig, ax = plt.subplots()

    # Plot ACE, MCE, and adjusted ECE on the same y-axis
    ax.plot(
        data["num_bins"],
        data["ACE"],
        label="ACE",
        marker="o",
        color="green",
        ms=4,
        linewidth=1,
    )
    ax.plot(
        data["num_bins"],
        data["MCE"],
        label="MCE",
        marker="^",
        color="red",
        ms=4,
        linewidth=1,
    )
    ax.plot(
        data["num_bins"],
        [d * 1e3 for d in data["ECE"]],
        label="ECE x $10^3$",
        marker="s",
        color="blue",
        ms=4,
        linewidth=1,
    )  # ECE values scaled

    ax.set_xlabel("Number of Bins")
    ax.set_ylabel("Metric Value", color="black")
    ax.tick_params(axis="y", labelcolor="black")
    ax.legend(loc="center right")
    ax.set_xscale("log")  # Set the x-axis to logarithmic scale
    ax.set_ylim(0, 0.7)

    fig.savefig(
        "bin_size_vs_metric.pdf",
        format="pdf",
        dpi=100,
        transparent=True,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    data = load_data()
    set_rc_params()
    make_plot(data)
