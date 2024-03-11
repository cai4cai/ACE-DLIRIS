import torch

from monai.transforms import (
    MapTransform,
)

__all__ = ["ConvertToBratsClassesd", "ConvertToBratsClassesSoftmaxd"]


class ConvertToBratsClassesd(MapTransform):
    """
    Convert class labels [0, 1, 2, 3] to "one-hot" encoded format for BraTS segmentation task.
    Works with unbatched data with shape [1, H, W, D].

    IN (As stored on disk):
    1: necrotic tumour core (NCR)
    2: peritumoral edema (ED)
    3: GD-enhancing tumor (ET) - previously label 4

    OUT ONEHOT (As required for eval):
    1: Enhancing tumour (ET) = ET
    2: Tumour core (TC) = NCR + ET
    3: Whole tumour (WT) = NCR + ED + ET
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = d[key]  # x is expected to have the shape [1, H, W, D]

            # # Error handling: Check input tensor shape  - Error handling is slow here
            # if x.ndim != 4 or x.shape[0] != 1:
            #     raise ValueError(f"Input tensor for key '{key}' must have shape [1, H, W, D] but got shape {x.shape}")

            # # Error handling: Check input tensor values  - Error handling is slow here
            # if (len(x.unique()) != len(x.int().unique())) or not torch.all((x >= 0) & (x <= 3)):
            #     raise ValueError(f"Input tensor for key '{key}' contains values outside of [0, 1, 2, 3]")

            result = []
            # 0 - Background (BG)
            result.append((x == 0).float().squeeze(0))  # Remove the singleton dimension
            # 1 - ET (Enhancing tumour)
            result.append((x == 3).float().squeeze(0))
            # 2 - TC (Tumour core) = NCR (1) + ET (3)
            tc = (torch.logical_or(x == 1, x == 3)).float().squeeze(0)
            result.append(tc)
            # 3 - WT (Whole tumour) = NCR (1) + ED (2) + ET (3)
            wt = (
                torch.logical_or(torch.logical_or(x == 1, x == 2), x == 3)
                .float()
                .squeeze(0)
            )
            result.append(wt)

            # Stack along the channel dimension to get the shape [C, H, W, D]
            d[key] = torch.stack(result, dim=0)
        return d


class ConvertToBratsClassesSoftmaxd(MapTransform):
    """
    This transformation converts softmax class probabilities into combined BRATS tumor
    classes. The transformation is designed for use with unbatched data, where each
    item has the shape [C, H, W, D] and C represents the number of classes.

    IN: Multichannel softmax values
    1: necrotic tumour core (NCR)
    2: peritumoral edema (ED)
    3: GD-enhancing tumor (ET)

    OUT (As required for eval):
    1: Enhancing tumour (ET) = ET
    2: Tumour core (TC) = NCR + ET
    3: Whole tumour (WT) = NCR + ED + ET


    Args:
        data (dict): Dictionary containing the input data with keys corresponding to the data
                     items to be transformed.

    Returns:
        dict: Dictionary containing the transformed data with combined class probabilities
              for tumor regions. The output maintains the original shape [C, H, W, D] with
              the channel dimension now representing the combined classes.

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            probabilities = d[
                key
            ]  # probabilities are expected to have the shape [C, H, W, D]
            result = []
            # 0 - Background (BG)
            result.append(probabilities[0, ...])
            # 1 - ET (Enhancing tumour)
            result.append(probabilities[3, ...])
            # 2 - TC (Tumour core)
            tc = probabilities[1, ...] + probabilities[3, ...]
            result.append(tc)
            # 3 - WT (Whole tumour)
            wt = probabilities[1, ...] + probabilities[2, ...] + probabilities[3, ...]
            result.append(wt)

            # Stack along the channel dimension to maintain the shape [C, H, W, D]
            d[key] = torch.stack(result, dim=0)
        return d
