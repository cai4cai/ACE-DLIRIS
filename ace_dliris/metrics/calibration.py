from typing import Any, Callable
import os
from monai.config import TensorOrList
import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction
from monai.utils.enums import StrEnum
from monai.metrics import CumulativeIterationMetric

from ace_dliris.losses import hard_binned_calibration
from ace_dliris.visualize import (
    draw_case_reliability_diagrams,
    draw_dataset_reliability_diagrams,
)


__all__ = [
    "calibration_binning",
    "CalibrationErrorMetric",
    "CalibrationReduction",
    "calculate_heatmap_from_bins",
    "ReliabilityDiagramMetric",
]


def calibration_binning(
    y_pred: torch.Tensor, y: torch.Tensor, num_bins: int = 20, right: bool = False
) -> torch.Tensor:
    """
    Compute the calibration bins for the given data. This function calculates the mean predictions,
    mean ground truths, and bin counts for each bin in a hard binning calibration approach.

    The function operates on input and target tensors with batch and channel dimensions,
    handling each batch and channel separately. For bins that do not contain any elements,
    the mean predicted values and mean ground truth values are set to NaN.


    Args:
        y_pred (torch.Tensor): predicted tensor with shape [batch, channel, spatial], where spatial
            can be any number of dimensions. The y_pred tensor represents predicted values or probabilities.
        y (torch.Tensor): Target tensor with the same shape as y_pred. It represents ground truth values.
        num_bins (int, optional): The number of bins to use for calibration. Defaults to 20.
        right (bool, optional): If False (default), the bins include the left boundary and exclude the right boundary.
            If True, the bins exclude the left boundary and include the right boundary.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - mean_p_per_bin (torch.Tensor): Tensor of shape [batch_size, num_channels, num_bins] containing
              the mean predicted values in each bin.
            - mean_gt_per_bin (torch.Tensor): Tensor of shape [batch_size, num_channels, num_bins] containing
              the mean ground truth values in each bin.
            - bin_counts (torch.Tensor): Tensor of shape [batch_size, num_channels, num_bins] containing
              the count of elements in each bin.

    Raises:
        ValueError: If the input and target shapes do not match or if the input is not three-dimensional.

    Note:
        This function currently uses nested for loops over batch and channel dimensions
        for binning operations. Future improvements may include vectorizing these operations
        for enhanced performance.
    """
    batch_size, num_channels = y_pred.shape[:2]
    boundaries = torch.linspace(
        start=0.0,
        end=1.0 + torch.finfo(torch.float32).eps,
        steps=num_bins + 1,
        device=y_pred.device,
    )

    mean_p_per_bin = torch.zeros(
        batch_size, num_channels, num_bins, device=y_pred.device
    )
    mean_gt_per_bin = torch.zeros_like(mean_p_per_bin)
    bin_counts = torch.zeros_like(mean_p_per_bin)

    y_pred = y_pred.flatten(start_dim=2).float()
    y = y.flatten(start_dim=2).float()

    for b in range(batch_size):
        for c in range(num_channels):
            bin_idx = torch.bucketize(y_pred[b, c, :], boundaries[1:], right=right)
            bin_counts[b, c, :] = torch.zeros_like(boundaries[1:]).scatter_add(
                0, bin_idx, torch.ones_like(y_pred[b, c, :])
            )

            mean_p_per_bin[b, c, :] = torch.empty_like(boundaries[1:]).scatter_reduce(
                0, bin_idx, y_pred[b, c, :], reduce="mean", include_self=False
            )
            mean_gt_per_bin[b, c, :] = torch.empty_like(boundaries[1:]).scatter_reduce(
                0, bin_idx, y[b, c, :].float(), reduce="mean", include_self=False
            )

    # Remove nonsense bins:
    mean_p_per_bin[bin_counts == 0] = torch.nan
    mean_gt_per_bin[bin_counts == 0] = torch.nan

    return mean_p_per_bin, mean_gt_per_bin, bin_counts


class CalibrationReduction(StrEnum):

    EXPECTED = "expected"
    AVERAGE = "average"
    MAXIMUM = "maximum"


class CalibrationErrorMetric(CumulativeIterationMetric):
    """
    Compute the Calibration Error between predicted probabilities and ground truth labels.
    This metric is suitable for multi-class tasks and supports batched inputs.

    The input `y_pred` represents the model's predicted probabilities, and `y` represents the ground truth labels.
    `y_pred` is expected to have probabilities, and `y` should be in one-hot format. You can use suitable transforms
    in `monai.transforms.post` to achieve the desired format.

    The `include_background` parameter can be set to `False` to exclude the first category (channel index 0),
    which is conventionally assumed to be the background. This is particularly useful in segmentation tasks where
    the background class might skew the calibration results.

    The metric supports both single-channel and multi-channel data. For multi-channel data, the input tensors
    should be in the format of BCHW[D], where B is the batch size, C is the number of channels, and HW[D] are the spatial dimensions.

    Args:
        num_bins (int): Number of bins to divide probabilities into for calibration calculation. Defaults to 20.
        include_background (bool): Whether to include ACE computation on the first channel of the predicted output.
            Defaults to `True`.
        calibration_reduction (CalibrationReduction | str): Method for calculating calibration error values from binned data.
            Available modes are `"expected"`, `"average"`, and `"maximum"`. Defaults to `"expected"`.
        metric_reduction (MetricReduction | str): Mode of reduction to apply to the metrics. Reduction is only applied to non-NaN values.
            Available reduction modes are `"none"`, `"mean"`, `"sum"`, `"mean_batch"`, `"sum_batch"`, `"mean_channel"`, and `"sum_channel"`.
            Defaults to `"mean"`. If set to `"none"`, no reduction will be performed.
        get_not_nans (bool): Whether to return the count of non-NaN values. If `True`, `aggregate()` returns a tuple (metric, not_nans).
            Defaults to `False`.
        right (bool): Whether to use the right or left bin edge for binning. Defaults to `False` (left).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Example:
        >>> metric = CalibrationErrorMetric(num_bins=15, include_background=False, calibration_reduction="expected")
        >>> for batch_data in dataloader:
        >>>     preds, labels = model(batch_data)
        >>>     metric(y_pred=preds, y=labels)
        >>> ace = metric.aggregate()
    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        calibration_reduction: (
            CalibrationReduction | str
        ) = CalibrationReduction.EXPECTED,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        right=False,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.include_background = include_background
        self.calibration_reduction = calibration_reduction
        self.metric_reduction = metric_reduction
        self.get_not_nans = get_not_nans
        self.right = right

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: input data to compute. It should be in the format of (batch, channel, spatial...).
                    It represents logits (before softmax) predictions of the model.
            y: ground truth in one-hot format. It should be in the format of (batch, channel, spatial...).
               The values should be binarized.
        """
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        mean_p_per_bin, mean_gt_per_bin, bin_counts = hard_binned_calibration(
            input=y_pred, target=y, num_bins=self.num_bins, right=self.right
        )

        # Calculate the absolute differences, ignoring nan values
        abs_diff = torch.abs(mean_p_per_bin - mean_gt_per_bin)

        if self.calibration_reduction == CalibrationReduction.EXPECTED:
            # Calculate the weighted sum of absolute differences
            return torch.nansum(abs_diff * bin_counts, dim=-1) / torch.sum(
                bin_counts, dim=-1
            )
        elif self.calibration_reduction == CalibrationReduction.AVERAGE:
            return torch.nanmean(
                abs_diff, dim=-1
            )  # Average across all dimensions, ignoring nan
        elif self.calibration_reduction == CalibrationReduction.MAXIMUM:
            abs_diff[torch.isnan(abs_diff)] = 0
            return torch.max(
                abs_diff, dim=-1
            ).values  # Maximum across all dimensions, ignoring nan
        else:
            raise ValueError(
                f"Unsupported calibration reduction: {self.calibration_reduction}"
            )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_ace`.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.metric_reduction)
        return (f, not_nans) if self.get_not_nans else f


def calculate_heatmap_from_bins(mean_gt_per_bin: torch.Tensor) -> torch.Tensor:
    """
    Calculate a heatmap representation from binning data of ground truth values.

    This function processes binning data to create a heatmap where each cell represents
    the count of ground truth values falling into a specific bin. The function operates
    on a tensor containing mean ground truth values for each bin, across multiple cases
    and classes. The resulting heatmap is useful for visualizing the distribution of
    ground truth values across different bins, which can be helpful in tasks like
    calibration analysis.

    The heatmap is calculated for each class separately and is flipped vertically to
    align with typical diagram representations, where the bottom left corner corresponds
    to the first bin.

    Args:
        mean_gt_per_bin (torch.Tensor): A tensor of shape [N, C, num_bins] where N is the number
            of cases, C is the number of classes, and num_bins is the number of bins. This tensor
            contains the mean ground truth values for each bin, for each class, across all cases.

    Returns:
        torch.Tensor: A tensor of shape [C, num_bins, num_bins] representing the heatmap for each
            class. Each element in the heatmap corresponds to the count of ground truth values
            falling into a specific bin.

    Note:
        The function assumes that the input tensor `mean_gt_per_bin` contains valid mean ground
        truth values for each bin and class. NaN values are ignored in the calculation.
    """
    num_cases, num_classes, num_bins = mean_gt_per_bin.shape

    boundaries = torch.linspace(
        start=0.0,
        end=1.0 + torch.finfo(torch.float32).eps,
        steps=num_bins + 1,
        device=mean_gt_per_bin.device,
    )

    heatmap = torch.zeros(
        num_classes,
        num_bins,
        num_bins,
        dtype=torch.int64,
        device=mean_gt_per_bin.device,
    )

    for c in range(num_classes):
        for b in range(num_bins):
            # For each bin and each channel, select all the cases
            mean_gts_per_case_in_bin_i = mean_gt_per_bin[:, c, b].contiguous()

            # Calculate which bin these mean gts fall into
            bin_idx = torch.bucketize(
                mean_gts_per_case_in_bin_i, boundaries[1:], right=False
            )

            # Calculate nan mask
            nan_mask = torch.isnan(mean_gts_per_case_in_bin_i)

            # Filter out NaN values and their corresponding indices
            valid_bin_idx = bin_idx[~nan_mask]
            valid_values = torch.ones_like(valid_bin_idx, dtype=heatmap.dtype)

            # Update the heatmap using scatter_add_
            heatmap[c, :, b].scatter_add_(0, valid_bin_idx, valid_values)

    heatmap = heatmap.flip(1)  # Flip the heatmap vertically to match the diagram

    return heatmap


def _aggregate_binning_data(binning_data: torch.Tensor):
    """
    Aggregate binning data across all cases.

    This function takes binning data for multiple cases and classes and aggregates
    it into a single tensor containing the mean predicted values, mean ground truth
    values, and bin counts for each bin. The function operates on a tensor containing
    binning data for multiple cases and classes, where the first dimension represents
    the number of cases and the second dimension represents the number of classes.

    Args:
        binning_data (torch.Tensor): A tensor of shape [N, C, 3, num_bins] where N is the number
            of cases, C is the number of classes, and num_bins is the number of bins. This tensor
            contains the mean predicted values, mean ground truth values, and bin counts for each
            bin, for each class, across all cases.

    Returns:
        torch.Tensor: A tensor of shape [C, 3, num_bins] representing the aggregated mean predicted values,
            mean ground truth values, and bin counts for each bin, for each class, across all cases.
    """
    num_cases, num_classes, _, num_bins = binning_data.shape

    # Extracting individual components
    mean_p_per_bin = binning_data[:, :, 0, :]  # shape: [N, C, num_bins]
    mean_gt_per_bin = binning_data[:, :, 1, :]  # shape: [N, C, num_bins]
    bin_counts = binning_data[:, :, 2, :]  # shape: [N, C, num_bins]

    # Aggregating across cases
    mean_p_per_bin_aggregated = torch.nanmean(mean_p_per_bin, dim=0)
    mean_gt_per_bin_aggregated = torch.nanmean(mean_gt_per_bin, dim=0)
    bin_counts_aggregated = torch.nansum(bin_counts, dim=0)

    # Stacking along dimension 1
    binning_data_aggregated = torch.stack(
        [mean_p_per_bin_aggregated, mean_gt_per_bin_aggregated, bin_counts_aggregated],
        dim=1,
    )

    return binning_data_aggregated


class ReliabilityDiagramMetric(CumulativeIterationMetric):
    """
    Compute and visualize reliability diagrams for model calibration assessment.

    This metric computes the calibration error between predicted probabilities and ground truth labels
    for multi-class tasks. It supports batched inputs and can generate reliability diagrams for both
    individual cases and aggregated dataset-level analysis.

    The input `y_pred` represents the model's predicted probabilities, and `y` represents the ground truth labels.
    The metric supports both single-channel and multi-channel data, with the input tensors expected to be in
    BCHW[D] format, where B is the batch size, C is the number of channels, and HW[D] are the spatial dimensions.

    Args:
        num_classes (int): Number of classes in the prediction.
        num_bins (int): Number of bins to divide probabilities into for calibration calculation. Defaults to 20.
        include_background (bool): Whether to include computation on the first channel of the predicted output.
            Defaults to `True`.
        output_dir (str | None): Directory to save the generated diagrams. If `None`, diagrams are not saved.
        figsize (tuple[int, int]): Size of the figure for the diagrams.
        class_names (list[str] | None): Names of the classes. If `None`, class names are auto-generated.
        draw_case_diagrams (bool): Whether to draw reliability diagrams for individual cases.
        draw_case_histograms (bool): Whether to draw histograms for individual cases.
        case_name_transform (Callable): Function to generate names for individual cases.
            Typically this is a function that extracts the case name from the metadata dictionary.
            If `None`, case names are auto-generated.
        print_case_ece (bool): Whether to print Expected Calibration Error for individual cases.
        print_case_ace (bool): Whether to print Average Calibration Error for individual cases.
        print_case_mce (bool): Whether to print Maximum Calibration Error for individual cases.
        draw_dataset_diagrams (bool): Whether to draw reliability diagrams for the aggregated dataset.
        draw_dataset_histograms (bool): Whether to draw histograms for the aggregated dataset.
        dataset_diagram_cmap (str): Colormap for the dataset-level diagrams.
        dataset_imshow_kwargs (dict[str, Any]): Additional keyword arguments for plotting dataset diagrams.
            eg: {"vmin": 0.0, "vmax": 0.2, "cmap": "YlOrRd"}
        savefig_kwargs (dict[str, Any]): Additional keyword arguments for saving the diagrams.
            eg: {"dpi": 300, "bbox_inches": "tight"}
        rc_params (dict[str, Any]): Additional keyword arguments for matplotlib rc_params.
            eg: {"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12}
        right (bool): Whether to use the right or left bin edge for binning. Defaults to `False` (left).

    Raises:
        ValueError: If the number of classes does not match the provided class names or if background is ignored
                    but only one class is present.
    """

    def __init__(
        self,
        num_classes: int,
        num_bins: int = 20,
        include_background: bool = True,
        output_dir: str | None = None,
        figsize: tuple[int, int] = (6, 6),
        class_names: list[str] | None = None,
        draw_case_diagrams: bool = False,
        draw_case_histograms: bool = False,
        case_name_transform: Callable = None,
        print_case_ece: bool = True,
        print_case_ace: bool = True,
        print_case_mce: bool = True,
        draw_dataset_diagrams: bool = True,
        draw_dataset_histograms: bool = False,
        dataset_imshow_kwargs: dict[str, Any] = {},
        savefig_kwargs: dict[str, Any] = {},
        rc_params: dict[str, Any] = {},
        right=False,
    ) -> None:

        super().__init__()
        # General parameters:
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.include_background = include_background
        self.output_dir = output_dir
        self.figsize = figsize
        self.savefig_kwargs = savefig_kwargs
        self.rc_params = rc_params
        # Case parameters:
        self.draw_case_diagrams = draw_case_diagrams
        self.draw_case_histograms = draw_case_histograms
        self.case_name_transform = case_name_transform
        self.print_case_ece = print_case_ece
        self.print_case_ace = print_case_ace
        self.print_case_mce = print_case_mce
        # Dataset parameters:
        self.draw_dataset_diagrams = draw_dataset_diagrams
        self.draw_dataset_histograms = draw_dataset_histograms
        self.dataset_imshow_kwargs = dataset_imshow_kwargs

        self.right = right

        if class_names is not None:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"Number of class names ({len(class_names)}) does not match number of classes ({num_classes})"
                )
        else:
            class_names = [f"class{i}" for i in range(0, num_classes)]

        if not include_background:
            if num_classes == 1:
                raise ValueError(
                    "Cannot ignore background when there is only one class"
                )

            self.class_names = class_names[1:]
        else:
            self.class_names = class_names

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        else:
            if draw_case_diagrams or draw_case_histograms:
                raise ValueError(
                    "Cannot draw case diagrams or histograms without an output directory"
                )

    def _compute_tensor(
        self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any
    ) -> TensorOrList:
        """
        Compute the binning data for calibration analysis from predictions and ground truth.

        This method computes the mean predicted values, mean ground truth values, and bin counts for each bin
        for each batch and channel. It also handles the generation of reliability diagrams for individual cases
        if specified.

        Args:
            y_pred (torch.Tensor): Predicted probabilities or logits with shape [batch, channel, spatial...].
            y (torch.Tensor): Ground truth labels in one-hot format with shape [batch, channel, spatial...].

        Returns:
            TensorOrList: Binning data as a tensor of shape [batch_size, num_classes, 3, num_bins].
        """

        if y.shape != y_pred.shape:
            raise ValueError(
                f"Shape of y ({y.shape}) does not match shape of y_pred ({y_pred.shape})"
            )

        if y_pred.shape[1] != self.num_classes:
            raise ValueError(
                f"Number of classes in y_pred ({y_pred.shape[1]}) does not match num_classes ({self.num_classes}) set during initialization"
            )

        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        binning_data = hard_binned_calibration(
            input=y_pred, target=y, num_bins=self.num_bins, right=self.right
        )
        binning_data = torch.stack(binning_data, dim=-2)  # shape: [B, C, 3, num_bins]

        if self.draw_case_diagrams:

            if self.case_name_transform is not None:
                case_names = self.case_name_transform(y)
            else:
                case_names = [f"case{i}" for i in range(1, y_pred.shape[0] + 1)]

            draw_case_reliability_diagrams(
                binning_data.cpu().numpy(),
                output_dir=self.output_dir,
                figsize=self.figsize,
                case_names=case_names,
                class_names=self.class_names,
                print_ece=self.print_case_ece,
                print_ace=self.print_case_ace,
                print_mce=self.print_case_mce,
                draw_histograms=self.draw_case_histograms,
                savefig_kwargs=self.savefig_kwargs,
                rc_params=self.rc_params,
            )

        return binning_data

    def aggregate(self, *args: Any, **kwargs: Any) -> Any:
        """
        Aggregate the binning data across all cases and generate dataset-level reliability diagrams.

        This method processes the accumulated binning data to compute a heatmap representation and
        aggregates the data across all cases. It also handles the generation of dataset-level reliability
        diagrams and histograms if specified.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the heatmap tensor and the aggregated
            binning data tensor. The heatmap tensor has a shape of [num_classes, num_bins, num_bins],
            and the aggregated binning data tensor has a shape of [num_classes, 3, num_bins].
        """
        data = self.get_buffer()  # shape: [N*B, C, 3, num_bins]

        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        heatmap = calculate_heatmap_from_bins(
            data[:, :, 1, :]
        )  # mean_gt_per_bin is at index 1
        binning_data = _aggregate_binning_data(data)

        if self.draw_dataset_diagrams:
            draw_dataset_reliability_diagrams(
                heatmap.cpu().numpy(),
                output_dir=self.output_dir,
                figsize=self.figsize,
                class_names=self.class_names,
                draw_histograms=self.draw_dataset_histograms,
                binning_data=binning_data.cpu().numpy(),
                imshow_kwargs=self.dataset_imshow_kwargs,
                savefig_kwargs=self.savefig_kwargs,
                rc_params=self.rc_params,
            )

        return heatmap
