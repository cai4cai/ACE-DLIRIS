import pytest
import torch
import numpy as np
import os
from monai.utils import MetricReduction
import tempfile
from pathlib import Path
import hashlib
import pickle

from ace_dliris.metrics import (
    calibration_binning,
    CalibrationErrorMetric,
    CalibrationReduction,
    ReliabilityDiagramMetric,
    calculate_heatmap_from_bins,
)
from monai.metrics.utils import ignore_background
import unittest.mock as mock


DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


@pytest.fixture(params=DEVICES, ids=[str(d) for d in DEVICES])
def device(request):
    return request.param


import matplotlib

matplotlib.use("Agg")  # non-interactive backend


# test binning operations:  -- same cases as for test_losses
test_cases_calibration_binning = [
    {
        "case_name": "small_mid",  # small case with probabilities in middle of bin
        "input": [[[[0.7, 0.3], [0.1, 0.9]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[0.1, 0.3, float("nan"), 0.7, 0.9]]],
        "expected_mean_gt_per_bin": [[[0.0, 0.0, float("nan"), 1.0, 1.0]]],
        "expected_bin_counts": [[[1.0, 1.0, 0.0, 1.0, 1.0]]],
    },
    {
        "case_name": "large_mid",  # larger case with probabilities in middle of bin
        "input": [
            [[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]],
            [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]],
        ],
        "target": [
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
        ],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [
            [[0.1, 0.3, torch.nan, 0.7, 0.9], [torch.nan, 0.3, 0.5, 0.7, torch.nan]],
            [
                [torch.nan, 0.3, torch.nan, torch.nan, 0.9],
                [0.1, torch.nan, torch.nan, 0.7, 0.9],
            ],
        ],
        "expected_mean_gt_per_bin": [
            [[0.0, 0.0, torch.nan, 1.0, 1.0], [torch.nan, 1.0, 0.5, 0.0, torch.nan]],
            [
                [torch.nan, 0.0, torch.nan, torch.nan, 1.0],
                [0.0, torch.nan, torch.nan, 1.0, 1.0],
            ],
        ],
        "expected_bin_counts": [
            [[1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 1.0, 1.0]],
        ],
    },
    {
        "case_name": "small_off",  # small case with probabilities off center of bin
        "input": [[[[0.65, 0.25], [0.15, 0.95]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[0.15, 0.25, torch.nan, 0.65, 0.95]]],
        "expected_mean_gt_per_bin": [[[0.0, 0.0, torch.nan, 1.0, 1.0]]],
        "expected_bin_counts": [[[1.0, 1.0, 0.0, 1.0, 1.0]]],
    },
    {
        "case_name": "small_left_edge",  # small case with probabilities on left bin boundaries
        "input": [[[[0.8, 0.2], [0.4, 0.6]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[0.2, 0.4, 0.6, 0.8, torch.nan]]],
        "expected_mean_gt_per_bin": [[[0.0, 0.0, 1.0, 1.0, torch.nan]]],
        "expected_bin_counts": [[[1.0, 1.0, 1.0, 1.0, 0.0]]],
    },
    {
        "case_name": "small_right_edge",  # small case with probabilities on right bin boundaries
        "input": [[[[0.8, 0.2], [0.4, 0.6]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": True,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[torch.nan, 0.2, 0.4, 0.6, 0.8]]],
        "expected_mean_gt_per_bin": [[[torch.nan, 0.0, 0.0, 1.0, 1.0]]],
        "expected_bin_counts": [[[0.0, 1.0, 1.0, 1.0, 1.0]]],
    },
]


@pytest.mark.parametrize(
    "case",
    test_cases_calibration_binning,
    ids=[c["case_name"] for c in test_cases_calibration_binning],
)
def test_binning(device, case):

    input_tensor = torch.tensor(case["input"], device=device)
    target_tensor = torch.tensor(case["target"], device=device)

    # Use mock.patch to replace torch.linspace
    # This is to avoid floating point precision issues when looking at edge conditions
    mock_boundaries = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=device)
    with mock.patch("torch.linspace", return_value=mock_boundaries):
        mean_p_per_bin, mean_gt_per_bin, bin_counts = calibration_binning(
            input_tensor, target_tensor, num_bins=case["num_bins"], right=case["right"]
        )

    expected_mean_p_per_bin = torch.tensor(
        case["expected_mean_p_per_bin"], device=device
    )
    expected_mean_gt_per_bin = torch.tensor(
        case["expected_mean_gt_per_bin"], device=device
    )
    expected_bin_counts = torch.tensor(case["expected_bin_counts"], device=device)

    assert torch.allclose(mean_p_per_bin, expected_mean_p_per_bin, equal_nan=True)
    assert torch.allclose(mean_gt_per_bin, expected_mean_gt_per_bin, equal_nan=True)
    assert torch.allclose(bin_counts, expected_bin_counts)


@pytest.fixture(
    params=[
        CalibrationReduction.EXPECTED,
        CalibrationReduction.AVERAGE,
        CalibrationReduction.MAXIMUM,
    ],
    ids=lambda cr: cr.value,
)
def calibration_reduction(request):
    return request.param


# @pytest.fixture(params=[MetricReduction.NONE,
#                         MetricReduction.MEAN,
#                         MetricReduction.SUM], ids=lambda mr: mr.value)
# def metric_reduction(request):
#     return request.param

value_test_cases = [
    {
        "case_name": "1b1c",  # small case with probabilities in middle of bin
        "y_pred": [[[[0.7, 0.3], [0.1, 0.9]]]],
        "y": [[[[1, 0], [0, 1]]]],
        "num_bins": 5,
        "right": False,
        "expected_average_value": [[0.2]],
        "expected_maximum_value": [[0.3]],
        "expected_expected_value": [[0.2]],
    },
    {
        "case_name": "2b2c",  # larger case with probabilities in middle of bin
        "y_pred": [
            [[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]],
            [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]],
        ],
        "y": [
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
        ],
        "right": False,
        "num_bins": 5,
        "expected_average_value": [[0.2000, 0.4667], [0.2000, 0.1667]],
        "expected_maximum_value": [[0.3000, 0.7000], [0.3000, 0.3000]],
        "expected_expected_value": [[0.2000, 0.3500], [0.2000, 0.1500]],
    },
]


@pytest.mark.parametrize(
    "case", value_test_cases, ids=[c["case_name"] for c in value_test_cases]
)
def test_calibration_error_metric_value(device, calibration_reduction, case):
    y_pred = torch.tensor(case["y_pred"], device=device)
    y = torch.tensor(case["y"], device=device)

    b, c, *_ = y_pred.shape

    metric = CalibrationErrorMetric(
        num_bins=case["num_bins"],
        include_background=True,
        calibration_reduction=calibration_reduction,
        metric_reduction=MetricReduction.NONE,
        get_not_nans=False,
    )

    metric(y_pred=y_pred, y=y)
    f = metric.aggregate()

    # f = CalibrationErrorMetric._compute_tensor(metric, y_pred, y)
    assert f.shape == (b, c)

    if calibration_reduction == CalibrationReduction.EXPECTED:
        expected_value = case["expected_expected_value"]
    elif calibration_reduction == CalibrationReduction.AVERAGE:
        expected_value = case["expected_average_value"]
    elif calibration_reduction == CalibrationReduction.MAXIMUM:
        expected_value = case["expected_maximum_value"]
    else:
        raise ValueError(f"Unsupported calibration reduction: {calibration_reduction}")

    assert torch.allclose(f, torch.tensor(expected_value, device=device), atol=1e-4)


heat_map_test_cases = [
    {
        "case_name": "case1",
        "mean_gt_per_bin": [
            [[torch.nan, 1.0, torch.nan, 0.5, torch.nan]],
            [[torch.nan, torch.nan, 0.5, 0.0, 1.0]],
            [[torch.nan, 1.0, 0.0, torch.nan, 1.0]],
            [[torch.nan, 0.0, 0.0, 0.0, torch.nan]],
        ],
        "expected_heatmap": [
            [
                [0, 2, 0, 0, 2],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 2, 2, 0],
            ]
        ],
    }
]


@pytest.mark.parametrize(
    "case", heat_map_test_cases, ids=[c["case_name"] for c in heat_map_test_cases]
)
def test_calculate_heatmap_from_bins(device, case):
    mean_gt_per_bin = torch.tensor(case["mean_gt_per_bin"], device=device)
    heatmap = calculate_heatmap_from_bins(mean_gt_per_bin)
    expected_heatmap = torch.tensor(
        case["expected_heatmap"], dtype=torch.int64, device=device
    )
    assert torch.allclose(heatmap, expected_heatmap)


heat_map_numpy_test_cases = [
    {
        "case_name": "case1",
        "shape": (10, 2, 5),
    },
    {
        "case_name": "case2",
        "shape": (50, 4, 20),
    },
]


@pytest.mark.parametrize(
    "case",
    heat_map_numpy_test_cases,
    ids=[c["case_name"] for c in heat_map_numpy_test_cases],
)
def test_calculate_heatmap_from_bins_compare_numpy(case):
    mean_gt_per_bin = torch.rand(case["shape"])
    heatmap_torch = calculate_heatmap_from_bins(mean_gt_per_bin)

    # numpy method:
    def calculate_heatmap_from_bins_np(mean_gt_per_bin):
        num_cases, num_classes, num_bins = mean_gt_per_bin.shape
        heatmap = np.zeros((num_classes, num_bins, num_bins))

        boundaries = np.linspace(
            start=0.0, stop=1.0 + np.finfo(np.float32).eps, num=num_bins + 1
        )

        for c in range(num_classes):
            for b in range(num_bins):
                mean_gts_per_case_in_bin_i = mean_gt_per_bin[:, c, b]
                bin_idx = np.digitize(mean_gts_per_case_in_bin_i, bins=boundaries[1:])
                nan_mask = np.isnan(mean_gts_per_case_in_bin_i)
                filtered_bin_idx = np.where(nan_mask, -1, bin_idx)
                np.add.at(heatmap[c, :, b], filtered_bin_idx, ~nan_mask)

        heatmap = heatmap[:, ::-1, :]
        return heatmap

    heatmap_numpy = calculate_heatmap_from_bins_np(mean_gt_per_bin.cpu().numpy())
    assert torch.allclose(heatmap_torch, torch.from_numpy(heatmap_numpy.copy()).long())


reliability_diagram_cases = [
    {
        "case_name": "1b1c_2x2",
        "shape": (1, 1, 2, 2),
        "num_iterations": 2,
        "seed": 0,
        "num_bins": 5,
        "right": False,
    },
    {
        "case_name": "2b2c_2x2",
        "shape": (2, 2, 2, 2),
        "num_iterations": 2,
        "right": False,
        "num_bins": 5,
    },
    {
        "case_name": "1b4c_16x16x16",
        "shape": (1, 4, 16, 16, 16),
        "num_iterations": 4,
        "right": False,
        "num_bins": 20,
    },
]


@pytest.fixture(params=[True, False], ids=["BG", "noBG"])
def include_background(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["hist", "nohist"])
def draw_histograms(request):
    return request.param


def case_name_transform_generator(
    device, include_background, draw_histograms, case_name
):
    iteration = 0  # Initialize a counter

    def name_transform(y):
        nonlocal iteration
        names = []
        for i in range(y.size(0)):
            # Include flags in the name
            bg_flag = "BG" if include_background else "noBG"
            hist_flag = "hist" if draw_histograms else "nohist"

            # Generate name with the case name, index, flags, and iteration
            name = f"{device}_{bg_flag}_{hist_flag}_{case_name}_i{iteration}_b{i}"
            names.append(name)

        iteration += 1  # Increment the iteration for the next call
        return names

    return name_transform


@pytest.mark.parametrize(
    "case",
    reliability_diagram_cases,
    ids=[c["case_name"] for c in reliability_diagram_cases],
)
def test_reliability_diagram_metric(device, include_background, draw_histograms, case):

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        temp_dir = Path(temp_dir)

        # Set a fixed random seed for reproducibility
        seed_value = 42
        torch.manual_seed(seed_value)

        b, c, *_ = case["shape"]

        # Define the output directory for the figures
        # tests_dir = Path(__file__).parent

        case_name_transform = case_name_transform_generator(
            device, include_background, draw_histograms, case["case_name"]
        )

        if c == 1 and not include_background:
            with pytest.raises(ValueError):
                metric = ReliabilityDiagramMetric(
                    num_classes=c,
                    num_bins=case["num_bins"],
                    include_background=include_background,
                )
            return

        metric = ReliabilityDiagramMetric(
            num_classes=c,
            num_bins=case["num_bins"],
            include_background=include_background,
            output_dir=temp_dir,
            figsize=(6, 6),
            class_names=None,
            draw_case_diagrams=True,
            draw_case_histograms=draw_histograms,
            case_name_transform=case_name_transform,
            print_case_ece=True,
            print_case_ace=True,
            print_case_mce=True,
            draw_dataset_diagrams=True,
            draw_dataset_histograms=draw_histograms,
        )

        # seperate name transform for checking files exist -- otherwise counter will incremenet from original factory closure
        case_name_transform = case_name_transform_generator(
            device, include_background, draw_histograms, case["case_name"]
        )

        for _ in range(case["num_iterations"]):
            y_pred = torch.rand(case["shape"], device=device)
            y = torch.randint(0, 2, case["shape"], device=device)

            metric(y_pred=y_pred, y=y)

            if not include_background:
                y_pred, y = ignore_background(y_pred=y_pred, y=y)

            case_names = case_name_transform(y)

            for name in case_names:
                file_path = temp_dir / f"{name}_reliability_diagram.png"
                assert file_path.exists()

        f = metric.aggregate()

        c = c - 1 if not include_background else c

        # Check aggregated shapes
        assert f.shape == torch.Size([c, case["num_bins"], case["num_bins"]])  # heatmap

        # Check dataset reliability diagram exists:
        file_path = temp_dir / "dataset_reliability_diagram.png"
        assert file_path.exists()

        stop = "set breakpoint here to inspect temp_dir contents"
        del stop
