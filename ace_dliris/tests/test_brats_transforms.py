import torch
import pytest
from ace_dliris.brats_transforms import (
    ConvertToBratsClassesd,
    ConvertToBratsClassesSoftmaxd,
)

####################################################################################################
# Test cases for ConvertToBratsClassesd
####################################################################################################


@pytest.fixture
def labelmap_data():
    # Creating a dummy tensor of shape [1, H, W, D]
    H, W, D = 10, 10, 10  # Example dimensions
    data = torch.randint(0, 4, (1, H, W, D))
    return {"key": data}


@pytest.fixture(
    params=[
        {"data": torch.randint(0, 4, (2, 10, 10, 10)), "msg": "shape [1, H, W, D]"},
        {"data": torch.randint(0, 4, (1, 10, 10)), "msg": "shape [1, H, W, D]"},
        {
            "data": torch.randint(-1, 5, (1, 10, 10, 10)),
            "msg": "values outside of [0, 1, 2, 3]",
        },
        {"data": torch.rand((1, 10, 10, 10)), "msg": "values outside of [0, 1, 2, 3]"},
    ]
)
def invalid_labelmap_data(request):
    return request.param


# Test cases for ConvertToBratsClassesd
def test_convert_to_brats_classesd(labelmap_data):
    transform = ConvertToBratsClassesd(keys=["key"])
    transformedd = transform(labelmap_data)
    transformed = transformedd["key"]

    # Check shapes and values
    assert transformed.shape == (4, 10, 10, 10), "Output shape is not as expected"
    assert transformed.dtype == torch.float32, "Output type should be float32"
    assert torch.all(
        (transformed == 0) | (transformed == 1)
    ), "Output should be one-hot encoded"

    # Check Background (BG) class
    assert torch.all(
        transformed[0, ...] == (labelmap_data["key"] == 0)
    ).item(), "BG class not matching"

    # Check Enhancing tumor (ET) class
    assert torch.all(
        transformed[1, ...] == (labelmap_data["key"] == 3)
    ).item(), "ET class not matching"

    # Check Tumor core (TC) class
    assert torch.all(
        transformed[2, ...]
        == torch.logical_or(labelmap_data["key"] == 1, labelmap_data["key"] == 3)
    ).item(), "TC class not matching"

    # Check Whole tumor (WT) class
    wt_condition = torch.logical_or(
        torch.logical_or(labelmap_data["key"] == 1, labelmap_data["key"] == 2),
        labelmap_data["key"] == 3,
    )
    assert torch.all(
        transformed[3, ...] == wt_condition
    ).item(), "WT class not matching"


# def test_convert_to_brats_classesd_error_handling(invalid_labelmap_data):
#     transform = ConvertToBratsClassesd(keys=['key'])
#     invalid_data = {"key": invalid_labelmap_data["data"]}

#     # Expecting a ValueError due to incorrect input
#     with pytest.raises(ValueError):  # match=invalid_input_data["msg"]
#         transform(invalid_data)


####################################################################################################
# Test cases for ConvertToBratsClassesSoftmaxd
####################################################################################################


@pytest.fixture
def softmax_data():
    # Creating a dummy tensor of shape [C, H, W, D] with random softmax probabilities
    C, H, W, D = 4, 10, 10, 10  # Example dimensions
    data = torch.randn(C, H, W, D)
    data = torch.softmax(data, dim=0)
    return {"key": data}


def test_convert_to_brats_classes_softmaxd(softmax_data):
    transform = ConvertToBratsClassesSoftmaxd(keys=["key"])
    transformed = transform(softmax_data)["key"]

    # Check shapes
    assert transformed.shape == (4, 10, 10, 10), "Output shape is not as expected"
    assert transformed.dtype == torch.float32, "Output type should be float32"

    # Verify probabilities are in [0, 1] range
    assert torch.all(
        (transformed >= 0) & (transformed <= 1)
    ), "Probabilities should be between 0 and 1"

    # Additional checks for combined classes can be added here


# @pytest.fixture(params=[
#     # Add different invalid input data scenarios here
# ])
# def invalid_softmax_input_data(request):
#     return request.param

# def test_convert_to_brats_classes_softmaxd_error_handling(invalid_softmax_input_data):
#     transform = ConvertToBratsClassesSoftmaxd(keys=['key'])
#     with pytest.raises(ValueError):  # Adjust as needed based on specific error expectations
#         transform({"key": invalid_softmax_input_data["data"]})
