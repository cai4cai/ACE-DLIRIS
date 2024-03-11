from typing import Callable, Dict, List, Tuple

import torch
from torch import nn, optim
from torch.nn import functional as F

from monai.handlers import from_engine
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.config import KeysCollection
from monai.utils import ImageMetaKey as Key

from monai.transforms import AsDiscrete

__all__ = [
    "discrete_from_engine",
    "meta_data_batch_transform",
    "meta_data_image_transform",
    "TemperatureScaling",
]


def discrete_from_engine(
    keys: KeysCollection, first: bool = False, threshold: list[float] = [0.5]
) -> Callable:
    """
    Factory function to create a callable for extracting and discretizing data
    from `ignite.engine.state.output`.

    Args:
        keys (KeysCollection): Keys to extract data from the input dictionary or list of dictionaries.
        first (bool): Whether to only extract data from the first dictionary if the input is a list of dictionaries.
        threshold (list[float]): List of threshold values for discretization, one for each key.

    Returns:
        Callable: A function that takes data and returns a tuple of discretized values for each key.
    """
    _keys = ensure_tuple(keys)
    _from_engine_func = from_engine(keys=_keys, first=first)
    # Ensuring that the threshold list is of the same length as keys
    _threshold = ensure_tuple_rep(threshold, len(_keys))

    def _wrapper(data):
        extracted_data = _from_engine_func(data)
        return tuple(
            [AsDiscrete(threshold=thr)(arr) for arr, thr in zip(lst, _threshold)]
            for lst in extracted_data
        )

    return _wrapper


def meta_data_batch_transform(batch):
    """
    Takes in batch from engine.state and returns case name from meta dict
    for the BraTs dataset
    """
    paths = [e["image"].meta[Key.FILENAME_OR_OBJ] for e in batch]
    names = [
        {Key.FILENAME_OR_OBJ: "_".join(path.split("/")[-1].split("_")[:2])}
        for path in paths
    ]
    return names


def meta_data_image_transform(images):
    """
    Takes in images from engine.state and returns case name from meta dict
    for the BraTs dataset
    """
    paths = [i.meta[Key.FILENAME_OR_OBJ] for i in images]
    names = ["_".join(path.split("/")[-1].split("_")[:2]) for path in paths]
    return names


class TemperatureScaling(nn.Module):
    """
    A class to wrap a neural network with temperature scaling.
    Output of network needs to be "raw" logits, not probabilities.
    """

    def __init__(
        self,
        network: nn.Module,
        network_ckpt_path: str | None = None,
    ):
        super(TemperatureScaling, self).__init__()
        # load network
        self.network = network
        if network_ckpt_path is not None:
            self.network.load_state_dict(torch.load(network_ckpt_path))
            self.network.eval()  # set to eval mode as we don't want to train the network
        device = next(self.network.parameters()).device
        self.temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)

    def forward(self, input):
        logits = self.network(input)
        return logits / self.temperature

    def parameters(self, recurse: bool = True):
        # Yield only the temperature parameter
        yield self.temperature
