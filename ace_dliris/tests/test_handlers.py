import tempfile
import os
import pytest
import torch
from ignite.engine import Engine, Events

from monai.handlers import from_engine

from ace_dliris.handlers import CalibrationError, ReliabilityDiagramHandler


DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


@pytest.fixture(params=DEVICES, ids=[str(d) for d in DEVICES])
def device(request):
    return request.param


test_cases = [
    {
        "case_name": "simple",
        "y_pred": [
            [[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]],
            [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]],
        ],
        "y": [
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
        ],
        "num_bins": 5,
        "include_background": True,
        "calibration_reduction": "expected",
        "metric_reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
        "num_iterations": 2,
        "expected_value": [[0.2250]],
    },
    {
        "case_name": "simple_ignore_background",
        "y_pred": [
            [[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]],
            [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]],
        ],
        "y": [
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
        ],
        "num_bins": 5,
        "include_background": False,
        "calibration_reduction": "expected",
        "metric_reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
        "num_iterations": 2,
        "expected_value": [[0.2500]],
    },
]


@pytest.mark.parametrize("case", test_cases, ids=[c["case_name"] for c in test_cases])
def test_calibration_error_handler(device, case):
    y_pred = torch.tensor(case["y_pred"], device=device)
    y = torch.tensor(case["y"], device=device)

    b, c, *_ = y_pred.shape

    c = c if case["include_background"] else c - 1

    handler = CalibrationError(
        num_bins=case["num_bins"],
        include_background=case["include_background"],
        calibration_reduction=case["calibration_reduction"],
        metric_reduction=case["metric_reduction"],
        output_transform=case["output_transform"],
    )

    engine = Engine(lambda e, b: None)
    handler.attach(engine, name="calibration_error")

    for _ in range(case["num_iterations"]):
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

    engine.fire_event(Events.EPOCH_COMPLETED)

    assert torch.allclose(
        torch.tensor(engine.state.metrics["calibration_error"]),
        torch.tensor(case["expected_value"]),
        atol=1e-4,
    )

    assert engine.state.metric_details["calibration_error"].shape == torch.Size(
        [b * case["num_iterations"], c]
    )


def test_reliability_diagrams(device):

    num_iterations = 4
    num_bins = 20
    shape = (2, 3, 16, 16, 16)
    b, c = shape[:2]

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = ReliabilityDiagramHandler(
            num_classes=c,
            num_bins=num_bins,
            output_dir=temp_dir,
            draw_case_diagrams=True,
            draw_dataset_diagrams=True,
            output_transform=from_engine(["pred", "label"]),
        )

        engine = Engine(lambda e, b: None)
        handler.attach(engine, name="reliability_diagrams")

        for _ in range(num_iterations):
            y_pred = torch.rand(*shape, device=device)
            y = torch.randint(0, 2, shape, device=device)
            engine.state.output = {"pred": y_pred, "label": y}
            engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)

        assert engine.state.metric_details["reliability_diagrams"].shape == torch.Size(
            [b * num_iterations, c, 3, num_bins]
        )

        # Check that temp directory is not empty:
        assert os.listdir(temp_dir) != []
