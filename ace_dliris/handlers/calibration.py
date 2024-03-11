from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..metrics import (
    CalibrationErrorMetric,
    CalibrationReduction,
    ReliabilityDiagramMetric,
)
from monai.utils import MetricReduction
from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.handlers.tensorboard_handlers import TensorBoardHandler

from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import
from monai.visualize import plot_2d_or_3d_image

Events, _ = optional_import(
    "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events"
)

if TYPE_CHECKING:
    from ignite.engine import Engine
    from tensorboardX import SummaryWriter as SummaryWriterX
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import(
        "ignite.engine",
        IgniteInfo.OPT_IMPORT_VERSION,
        min_version,
        "Engine",
        as_type="decorator",
    )
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")
    SummaryWriterX, _ = optional_import("tensorboardX", name="SummaryWriter")


__all__ = ["CalibrationError", "ReliabilityDiagramHandler"]


class CalibrationError(IgniteMetricHandler):
    """
    Computes Calibration Error and collects the average over batch, class-channels, iterations.
    Can return the expected, average, or maximum calibration error.

    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        calibration_reduction: (
            CalibrationReduction | str
        ) = CalibrationReduction.EXPECTED,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """
        Args:
            num_bins: number of bins to calculate calibration.
            include_background: whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            calibration_reduction (CalibrationReduction | str): Method for calculating calibration error values from binned data.
                Available modes are `"expected"`, `"average"`, and `"maximum"`. Defaults to `"expected"`.
            metric_reduction (MetricReduction | str): Mode of reduction to apply to the metrics. Reduction is only applied to non-NaN values.
                Available reduction modes are `"none"`, `"mean"`, `"sum"`, `"mean_batch"`, `"sum_batch"`, `"mean_channel"`, and `"sum_channel"`.
            Defaults to `"mean"`. If set to `"none"`, no reduction will be performed.
            num_classes: number of input channels (always including the background). When this is None,
                ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
                single-channel class indices and the number of classes is not automatically inferred from data.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: mean dice of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        """
        metric_fn = CalibrationErrorMetric(
            num_bins=num_bins,
            include_background=include_background,
            calibration_reduction=calibration_reduction,
            metric_reduction=metric_reduction,
        )

        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )


class ReliabilityDiagramHandler(IgniteMetricHandler):
    """
    Handler to compute and save reliability diagrams during training/validation.
    This handler extends IgniteMetricHandler and utilizes ReliabilityDiagramMetric
    to generate reliability diagrams, which are visual representations of model calibration.

    Reliability diagrams compare the predicted probabilities of a model with the actual outcomes,
    helping to understand how well the model's predicted probabilities are calibrated.
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
        savefig_kwargs: dict[str, Any] = {},  # Updated to use savefig_kwargs
        rc_params: dict[str, Any] = {},  # Added rc_params
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """
        Initializes the handler with the given parameters for computing and saving reliability diagrams.

        Args:
            num_classes: Number of classes (including background) to compute the reliability diagrams for.
            num_bins: Number of bins to use for the reliability diagrams.
            include_background: Whether to include background class in the reliability diagrams.
            output_dir: Directory where the diagrams will be saved.
            figsize: Size of each diagram.
            class_names: Names of the classes for the diagrams.
            draw_case_diagrams: Whether to draw diagrams for individual cases.
            draw_case_histograms: Whether to draw histograms for individual cases.
            case_name_transform: Function to transform case names.
            print_case_ece: Whether to print Expected Calibration Error for cases.
            print_case_ace: Whether to print Average Calibration Error for cases.
            print_case_mce: Whether to print Maximum Calibration Error for cases.
            draw_dataset_diagrams: Whether to draw diagrams for the entire dataset.
            draw_dataset_histograms: Whether to draw histograms for the entire dataset.
            dataset_imshow_kwargs: Additional keyword arguments for imshow function for dataset diagrams.
            savefig_kwargs: Additional keyword arguments for saving figures.
            rc_params: Additional keyword arguments for matplotlib rcParams.
            output_transform: Function to transform the output for metric computation.
            save_details: Whether to save detailed results.
        """

        metric_fn = ReliabilityDiagramMetric(
            num_classes=num_classes,
            num_bins=num_bins,
            include_background=include_background,
            output_dir=output_dir,
            figsize=figsize,
            class_names=class_names,
            draw_case_diagrams=draw_case_diagrams,
            draw_case_histograms=draw_case_histograms,
            case_name_transform=case_name_transform,
            print_case_ece=print_case_ece,
            print_case_ace=print_case_ace,
            print_case_mce=print_case_mce,
            draw_dataset_diagrams=draw_dataset_diagrams,
            draw_dataset_histograms=draw_dataset_histograms,
            dataset_imshow_kwargs=dataset_imshow_kwargs,
            savefig_kwargs=savefig_kwargs,  # Use savefig_kwargs here
            rc_params=rc_params,  # Use rc_params here
        )

        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )


class TensorBoardReliabilityDiagramHandler(TensorBoardHandler):
    """
    Text
    """

    pass
