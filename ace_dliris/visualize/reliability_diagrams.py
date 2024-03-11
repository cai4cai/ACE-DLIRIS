from typing import TYPE_CHECKING, Any

import os
import numpy as np
from monai.utils.module import optional_import

if TYPE_CHECKING:
    from matplotlib import cm
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
else:
    plt, _ = optional_import("matplotlib", name="pyplot")
    cm, _ = optional_import("matplotlib", name="cm")
    ticker, _ = optional_import("matplotlib", name="ticker")


# code adapted from: https://github.com/hollance/reliability-diagrams


__all__ = ["draw_case_reliability_diagrams", "draw_dataset_reliability_diagrams"]


def draw_case_reliability_diagrams(
    binning_data: np.ndarray,
    output_dir: str,
    figsize: tuple[int, int] = (6, 6),
    case_names: list[str] = ["case1", "case2", "case3"],
    class_names: list[str] = ["class1", "class2", "class2"],
    print_ece: bool = True,
    print_ace: bool = True,
    print_mce: bool = True,
    draw_histograms: bool = True,
    savefig_kwargs: dict[str, Any] = {},
    rc_params: dict[str, Any] = {},
):
    """
    Draw reliability diagrams for multiple cases.

    This function creates and saves reliability diagrams for each case in the provided binning data.
    Each case can represent a different model, experiment, or any other division of data. The function
    supports drawing additional histograms to show the distribution of examples in each bin.

    Args:
        binning_data (np.ndarray): An array of shape [B, C, 3, num_bins] containing binning data for B cases,
                                   C classes, and a specified number of bins. The three components in the third
                                   dimension represent mean predicted probabilities, mean ground truth frequencies,
                                   and bin counts, respectively.
        output_dir (str): Directory where the diagrams will be saved.
        figsize (tuple[int, int]): Size of each diagram. Defaults to (6, 6).
        case_names (list[str]): Names of the cases corresponding to the binning data. Defaults to ["case1", "case2", "case3"].
        class_names (list[str]): Names of the classes. Defaults to ["class1", "class2", "class3"].
        print_ece (bool): Whether to print Expected Calibration Error on the diagrams. Defaults to True.
        print_ace (bool): Whether to print Average Calibration Error on the diagrams. Defaults to True.
        print_mce (bool): Whether to print Maximum Calibration Error on the diagrams. Defaults to True.
        draw_histograms (bool): Whether to draw histograms showing the distribution of examples in each bin. Defaults to True.
        savefig_kwargs (dict[str, Any]): Additional keyword arguments to pass to `plt.savefig`.
        rc_params (dict[str, Any]): Additional keyword arguments to pass to `plt.rc`.

    Raises:
        ValueError: If the binning data does not have the expected shape or dimensions.
    """

    # shape: [B, C, 3, num_bins]
    # mean_p_per_bin, mean_gt_per_bin, bin_counts = binning_data[:, :, 0, :], binning_data[:, :, 1, :], binning_data[:, :, 2, :]

    if binning_data.ndim != 4:
        raise ValueError(
            f"Expected binning_data to be 4D, got {binning_data.ndim}D, should have shape [batch_size, num_channels, 3, num_bins]"
        )

    if binning_data.shape[2] != 3:
        raise ValueError(
            f"Expected binning_data to have 3 components in the third dimension, got {binning_data.shape[2]}"
        )

    b, c, _, num_bins = binning_data.shape

    for i in range(b):
        draw_reliability_diagram(
            binning_data[i],
            output_dir=output_dir,
            figsize=figsize,
            case_name=case_names[i],
            class_names=class_names,
            print_ece=print_ece,
            print_ace=print_ace,
            print_mce=print_mce,
            draw_histograms=draw_histograms,
            savefig_kwargs=savefig_kwargs,
            rc_params=rc_params,
        )


def draw_reliability_diagram(
    binning_data: np.ndarray,
    output_dir: str,
    figsize: tuple[int, int] = (
        6,
        6,
    ),  # just the size of one square reliability diagram
    case_name: str = "case1",
    class_names: list[str] = ["class1", "class2", "class2"],
    print_ece: bool = True,
    print_ace: bool = True,
    print_mce: bool = True,
    draw_histograms: bool = True,
    savefig_kwargs: dict[str, Any] = {},
    rc_params: dict[str, Any] = {},
):
    """
    Draw a reliability diagram for a single case.

    This function creates and saves a reliability diagram for the provided binning data of a single case.
    The diagram visualizes the calibration of predicted probabilities against the ground truth. It also
    supports drawing additional histograms to show the distribution of examples in each bin.

    Args:
        binning_data (np.ndarray): An array of shape [C, 3, num_bins] containing binning data for C classes
                                   and a specified number of bins. The three components in the second dimension
                                   represent mean predicted probabilities, mean ground truth frequencies, and
                                   bin counts, respectively.
        output_dir (str): Directory where the diagram will be saved.
        figsize (tuple[int, int]): Size of the diagram. Defaults to (6, 6).
        case_name (str): Name of the case corresponding to the binning data. Defaults to "case1".
        class_names (list[str]): Names of the classes. Defaults to ["class1", "class2", "class3"].
        print_ece (bool): Whether to print Expected Calibration Error on the diagram. Defaults to True.
        print_ace (bool): Whether to print Average Calibration Error on the diagram. Defaults to True.
        print_mce (bool): Whether to print Maximum Calibration Error on the diagram. Defaults to True.
        draw_histograms (bool): Whether to draw histograms showing the distribution of examples in each bin. Defaults to True.
        savefig_kwargs (dict[str, Any]): Additional keyword arguments to pass to `plt.savefig`.
        rc_params (dict[str, Any]): Additional keyword arguments to pass to `plt.rc`.

    Raises:
        ValueError: If the binning data does not have the expected shape or dimensions.
    """
    # shape: [C, 3, num_bins]
    # mean_p_per_bin, mean_gt_per_bin, bin_counts = binning_data[:, 0, :], binning_data[:, 1, :], binning_data[:, 2, :]

    if binning_data.ndim != 3:
        raise ValueError(
            f"Expected binning_data to be 3D, got {binning_data.ndim}D, should have shape [num_channels, 3, num_bins]"
        )

    if binning_data.shape[1] != 3:
        raise ValueError(
            f"Expected binning_data to have 3 components in the second dimension, got {binning_data.shape[1]}"
        )

    c, _, num_bins = binning_data.shape
    ncols = c

    plt.rcParams.update(**rc_params)

    if not draw_histograms:
        nrows = 1
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            figsize=(figsize[0] * c, figsize[1]),
            # constrained_layout=True,
        )
        if ncols == 1:
            ax = np.array([[ax]])
        elif ncols > 1:
            ax = ax[np.newaxis, ...]
    else:
        nrows = 2
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            figsize=(figsize[0] * c, figsize[1] * 2),
            gridspec_kw={"height_ratios": [4, 1]},
            # constrained_layout=True,
        )
        if ncols == 1:
            ax = ax[:, np.newaxis]

        plt.subplots_adjust(hspace=-0.2)

    assert ax.ndim == 2

    # plt.tight_layout()

    for i in range(ncols):
        if draw_histograms:
            _draw_reliability_diagram_subplot(
                ax[0, i],
                binning_data[i],
                title=class_names[i],
                xlabel="",
                print_ece=print_ece,
                print_ace=print_ace,
                print_mce=print_mce,
            )
            _draw_histogram_subplot(
                ax[1, i],
                binning_data[i],
                title="",
            )

        else:
            _draw_reliability_diagram_subplot(
                ax[0, i],
                binning_data[i],
                title=class_names[i],
                print_ece=print_ece,
                print_ace=print_ace,
                print_mce=print_mce,
            )

    if "format" in savefig_kwargs:
        _format = savefig_kwargs["format"]
    else:
        _format = "png"

    # Save the figure
    fig.savefig(
        os.path.join(output_dir, f"{case_name}_reliability_diagram.{_format}"),
        **savefig_kwargs,
    )

    # Optionally display the figure if in an interactive environment (like Jupyter)
    if plt.isinteractive():
        fig.show()

    # Close the figure to prevent it from displaying in non-interactive environments
    plt.close(fig)


def _draw_reliability_diagram_subplot(
    ax: plt.Axes,
    binning_data: np.ndarray,
    title: str = "Reliability Diagram",
    xlabel: str = "Mean Predicted Foreground Probability (per bin)",
    ylabel: str = "Mean Empirical Foreground Frequency (per bin)",
    print_ece: bool = True,
    print_ace: bool = True,
    print_mce: bool = True,
):
    """
    Draw a reliability diagram subplot.

    This internal function draws a reliability diagram on a given Axes object. It visualizes the calibration
    of predicted probabilities against the ground truth for a specific class or case.

    Args:
        ax (plt.Axes): Matplotlib Axes object to draw the diagram on.
        binning_data (np.ndarray): An array of shape [3, num_bins] containing binning data. The three components
                                   represent mean predicted probabilities, mean ground truth frequencies, and
                                   bin counts, respectively.
        title (str): Title of the subplot. Defaults to "Reliability Diagram".
        xlabel (str): Label for the x-axis. Defaults to "Mean Predicted Foreground Probability (per bin)".
        ylabel (str): Label for the y-axis. Defaults to "Mean Empirical Foreground Frequency (per bin)".
        print_ece (bool): Whether to print Expected Calibration Error on the diagram. Defaults to True.
        print_ace (bool): Whether to print Average Calibration Error on the diagram. Defaults to True.
        print_mce (bool): Whether to print Maximum Calibration Error on the diagram. Defaults to True.
    """
    # shape: [3, num_bins]

    mean_p_per_bin, mean_gt_per_bin, bin_counts = binning_data

    num_bins = len(mean_p_per_bin)
    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    positions = bins[:-1] + bin_size / 2.0
    widths = bin_size
    # min_count = np.min(bin_counts)
    # max_count = np.max(bin_counts)
    # normalized_bin_counts = (bin_counts - min_count) / (max_count - min_count)

    # TODO: bin importance from: https://github.com/hollance/reliability-diagrams could be used
    # TODO: could stop the legend from being drawn on top of ece and ace values

    colors = np.zeros((num_bins, 4))
    colors[:, 0] = 240 / 255.0
    colors[:, 1] = 60 / 255.0
    colors[:, 2] = 60 / 255.0
    colors[:, 3] = 0.3

    gap_plt = ax.bar(
        positions,
        np.abs(mean_p_per_bin - mean_gt_per_bin),
        bottom=np.minimum(mean_p_per_bin, mean_gt_per_bin),
        width=widths,
        edgecolor=colors,
        color=colors,
        linewidth=1,
        label="Gap",
    )

    acc_plt = ax.bar(
        positions,
        0,
        bottom=mean_gt_per_bin,
        width=widths,
        edgecolor="black",
        color="black",
        alpha=1.0,
        linewidth=3,
        label="Frequency",
    )

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # y = x dashed line

    abs_diff = np.abs(mean_p_per_bin - mean_gt_per_bin)

    if print_ece:
        ece = np.nansum(abs_diff * bin_counts) / np.sum(bin_counts)
        ax.text(
            0.98,
            0.02,
            "ECE=%.2e" % ece,
            color="black",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

    if print_ace:
        ace = np.nanmean(abs_diff)
        ax.text(
            0.98,
            0.06,
            "ACE=%.2e" % ace,
            color="black",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

    if print_mce:
        abs_diff = np.abs(mean_p_per_bin - mean_gt_per_bin)
        abs_diff[np.isnan(abs_diff)] = 0
        max_gap_index = np.argmax(abs_diff)
        mce_position = positions[max_gap_index]
        mce = np.max(abs_diff)

        # Determine position and direction of the arrow based on the gap
        if (mean_gt_per_bin[max_gap_index] - mean_p_per_bin[max_gap_index]) > 0:
            # mean_gt_per_bin is above mean_p_per_bin, arrow points down
            vertical_text_position = mean_gt_per_bin[max_gap_index] + 0.05
            va = "bottom"
            arrowhead_position = mean_gt_per_bin[max_gap_index]
        else:
            # mean_gt_per_bin is below mean_p_per_bin, arrow points up
            vertical_text_position = mean_gt_per_bin[max_gap_index] - 0.05
            va = "top"
            arrowhead_position = mean_gt_per_bin[max_gap_index]

        # Stop the arrow from going off the plot
        if vertical_text_position > 1:
            vertical_text_position = mean_p_per_bin[max_gap_index] - 0.05
            va = "top"
            arrowhead_position = mean_p_per_bin[max_gap_index]
        elif vertical_text_position < 0:
            vertical_text_position = mean_p_per_bin[max_gap_index] + 0.05
            va = "bottom"
            arrowhead_position = mean_p_per_bin[max_gap_index]

        ax.annotate(
            f"MCE={mce:.2e}",
            xy=(mce_position, arrowhead_position),  # Point where the arrow points to
            xytext=(mce_position, vertical_text_position),  # Position of the text
            arrowprops=dict(
                facecolor="black", shrink=0.05, width=1, headwidth=5, headlength=7
            ),
            ha="center",
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", lw=2
            ),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #    ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt])


def _draw_histogram_subplot(
    ax: plt.Axes,
    binning_data: np.ndarray,
    title: str = "Examples per bin",
    xlabel: str = "Mean Predicted Foreground Probability (per bin)",
    ylabel: str = "Count",
    draw_averages=False,
):
    """
    Draw a histogram subplot for bin counts.

    This internal function draws a histogram on a given Axes object. It visualizes the distribution of examples
    across the bins for a specific class or case.

    Args:
        ax (plt.Axes): Matplotlib Axes object to draw the histogram on.
        binning_data (np.ndarray): An array of shape [3, num_bins] containing binning data. The three components
                                   represent mean predicted probabilities, mean ground truth frequencies, and
                                   bin counts, respectively.
        title (str): Title of the histogram. Defaults to "Examples per bin".
        xlabel (str): Label for the x-axis. Defaults to "Mean Predicted Foreground Probability (per bin)".
        ylabel (str): Label for the y-axis. Defaults to "Count".
    """

    mean_p_per_bin, mean_gt_per_bin, bin_counts = binning_data
    num_bins = len(mean_p_per_bin)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_size = 1.0 / num_bins
    positions = bins[:-1] + bin_size / 2.0

    ax.bar(positions, bin_counts, width=bin_size * 0.9)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Determine if y-axis should be log or linear
    # Filter out zero values for scale determination
    non_zero_counts = bin_counts[bin_counts > 0]
    if (
        len(non_zero_counts) > 0
        and np.max(non_zero_counts) / np.min(non_zero_counts) > 100
    ):  # Threshold for log scale
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda y, _: f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "0"
            )
        )
    else:
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if draw_averages:
        avg_fore_prob = np.sum(mean_p_per_bin * bin_counts) / np.sum(bin_counts)
        avg_fore_freq = np.sum(mean_gt_per_bin * bin_counts) / np.sum(bin_counts)

        conf_plt = ax.axvline(
            x=avg_fore_prob,
            ls="dotted",
            lw=3,
            c="#444",
            label="Average foreground probability",
        )

        acc_plt = ax.axvline(
            x=avg_fore_freq,
            ls="solid",
            lw=3,
            c="black",
            label="Average foreground frequency",
        )
        ax.legend(handles=[acc_plt, conf_plt])


def draw_dataset_reliability_diagrams(
    heatmap: np.ndarray,  # shape: [C, num_bins, num_bins]
    output_dir: str,
    figsize: tuple[int, int] = (6, 6),
    class_names: list[str] = ["class1", "class2", "class2"],
    draw_histograms: bool = True,
    binning_data=None,  # shape [C, 3, num_bins]
    imshow_kwargs: dict[str, Any] = {},
    savefig_kwargs: dict[str, Any] = {},
    rc_params: dict[str, Any] = {},
):
    """
    Draw reliability diagrams for a dataset.

    This function creates and saves reliability diagrams for each class in the provided heatmap data.
    The diagrams visualize the calibration of predicted probabilities against the ground truth for the entire dataset.
    The function supports drawing additional histograms to show the distribution of examples in each bin.

    Args:
        heatmap (np.ndarray): An array of shape [C, num_bins, num_bins] containing heatmap data for C classes.
                               Each element in the heatmap corresponds to the count of ground truth values
                               falling into a specific bin.
        output_dir (str): Directory where the diagrams will be saved.
        figsize (tuple[int, int]): Size of each diagram. Defaults to (6, 6).
        class_names (list[str]): Names of the classes. Defaults to ["class1", "class2", "class3"].
        draw_histograms (bool): Whether to draw histograms showing the distribution of examples in each bin. Defaults to True.
        binning_data (np.ndarray): Optional. An array of shape [C, 3, num_bins] containing binning data for C classes.
                                   Required if `draw_histograms` is True.
        savefig_kwargs (dict[str, Any]): Additional keyword arguments to pass to `plt.savefig`.
        rc_params (dict[str, Any]): Additional keyword arguments to pass to `plt.rc`.

    Raises:
        ValueError: If the heatmap does not have the expected shape or dimensions, or if `binning_data` is required but not provided.
    """

    if heatmap.ndim != 3:
        raise ValueError(
            f"Expected heatmap to be 3D, got {heatmap.ndim}D, should have shape [num_classes, num_bins, num_bins]"
        )

    if heatmap.shape[1] != heatmap.shape[2]:
        raise ValueError(f"Expected heatmap to be square, got shape {heatmap.shape}")

    if draw_histograms and binning_data is None:
        raise ValueError(
            "Expected aggregated binning_data to be provided when draw_histograms is True"
        )

    num_classes, _, num_bins = heatmap.shape

    ncols = num_classes

    plt.rcParams.update(**rc_params)

    if not draw_histograms:
        nrows = 1
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            figsize=(figsize[0] * num_classes, figsize[1]),
            # constrained_layout=True,
        )
        if ncols == 1:
            ax = np.array([[ax]])
        elif ncols > 1:
            ax = ax[np.newaxis, ...]
    else:
        nrows = 2
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            figsize=(figsize[0] * num_classes, figsize[1] * 2),
            gridspec_kw={"height_ratios": [4, 1]},
            # constrained_layout=True,
        )
        if ncols == 1:
            ax = ax[:, np.newaxis]

        plt.subplots_adjust(hspace=-0.2)

    assert ax.ndim == 2

    if draw_histograms:  # TODO: ammend colorbar, just have one for each subplot
        cbar_ax = fig.add_axes(
            [0.91, 0.31, 0.02, 0.455]
        )  # TODO: manually setting this values isn't good
    else:
        cbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.77])

    for c in range(ncols):
        if draw_histograms:
            _draw_diagram_reliability_diagram_subplot(
                ax[0, c],
                heatmap[c, :, :],
                title=class_names[c],
                xlabel="",
                imshow_kwargs=imshow_kwargs,
            )
            _draw_histogram_subplot(
                ax[1, c],
                binning_data[c, :, :],
                title="",
                xlabel="Predicted Foreground Probability",
            )

        else:
            _draw_diagram_reliability_diagram_subplot(
                ax[0, c],
                heatmap[c, :, :],
                title=class_names[c],
                imshow_kwargs=imshow_kwargs,
            )

        # Draw the colorbar based on the last imshow plot
        if c == ncols - 1:
            im = ax[0, c].get_images()[0]  # Get the last imshow object
            fig.colorbar(im, cax=cbar_ax)  # Draw the colorbar

    # Optionally display the figure if in an interactive environment (like Jupyter)
    if plt.isinteractive():
        fig.show()

    if "format" in savefig_kwargs:
        _format = savefig_kwargs["format"]
    else:
        _format = "png"

    # Save the figure
    fig.savefig(
        os.path.join(output_dir, f"dataset_reliability_diagram.{_format}"),
        **savefig_kwargs,
    )

    # Close the figure to prevent it from displaying in non-interactive environments
    plt.close(fig)


def _draw_diagram_reliability_diagram_subplot(
    ax: plt.Axes,
    heatmap: np.ndarray,
    normalise_columns: bool = True,
    title: str = "Reliability Diagram",
    xlabel: str = "Predicted Foreground Probability",
    ylabel: str = "Empirical Foreground Frequency",
    imshow_kwargs: dict[str, Any] = {},
):
    """
    Draw a reliability diagram subplot using heatmap data.

    This internal function draws a reliability diagram on a given Axes object using heatmap data.
    It visualizes the distribution of ground truth values across different bins for a specific class.

    Args:
        ax (plt.Axes): Matplotlib Axes object to draw the diagram on.
        heatmap (np.ndarray): An array of shape [num_bins, num_bins] representing the heatmap for a class.
        normalise_columns (bool): Whether to normalise the heatmap columns. Defaults to True.
            This is to help with visualisation, as the heatmap values can be very large towards the edges
            For segmentation in medical imaging at least
        title (str): Title of the subplot. Defaults to "Reliability Diagram".
        xlabel (str): Label for the x-axis. Defaults to "Predicted Foreground Probability".
        ylabel (str): Label for the y-axis. Defaults to "Empirical Foreground Frequency".
    """
    # shape: [N, 3, num_bins]

    if normalise_columns:
        heatmap = heatmap / np.sum(heatmap, axis=0)

    if "entent" not in imshow_kwargs:
        imshow_kwargs["extent"] = [0, 1, 0, 1]

    ax.imshow(
        heatmap,
        **imshow_kwargs,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
