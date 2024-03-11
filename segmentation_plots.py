import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import matplotlib.colors as mcolors
from skimage.morphology import remove_small_objects
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy.ndimage import center_of_mass
import tqdm
import torch

import multiprocessing as mp

from ace_dliris.brats_transforms import (
    ConvertToBratsClassesd,
    ConvertToBratsClassesSoftmaxd,
)


def load_case(case):
    flair_path = f"../data/BraTS2021_TestingData/{case}/{case}_flair.nii.gz"
    gt_seg_path = f"../data/BraTS2021_TestingData/{case}/{case}_seg_c.nii.gz"

    baseline_seg_path = f"./bundle/runs/baseline_dice_brats_2021_high/inference_results/{case}_t1/{case}_t1_seg.nii.gz"
    baseline_prob_path = f"./bundle/runs/baseline_dice_brats_2021_high/inference_results/{case}_t1/{case}_t1_prob.nii.gz"
    l1ace_seg_path = f"./bundle/runs/hardl1ace_dice_brats_2021_high/inference_results/{case}_t1/{case}_t1_seg.nii.gz"
    l1ace_prob_path = f"./bundle/runs/hardl1ace_dice_brats_2021_high/inference_results/{case}_t1/{case}_t1_prob.nii.gz"

    # Load the images using nibabel
    flair_img = nib.load(flair_path)
    gt_seg_img = nib.load(gt_seg_path)
    baseline_seg_img = nib.load(baseline_seg_path)
    baseline_prob_img = nib.load(baseline_prob_path)
    l1ace_seg_img = nib.load(l1ace_seg_path)
    l1ace_prob_img = nib.load(l1ace_prob_path)

    # Convert nibabel image objects to numpy arrays
    flair_array = flair_img.get_fdata()
    gt_seg_array = gt_seg_img.get_fdata()
    baseline_seg_array = baseline_seg_img.get_fdata()
    baseline_prob_array = baseline_prob_img.get_fdata().transpose(3, 0, 1, 2)
    l1ace_seg_array = l1ace_seg_img.get_fdata()
    l1ace_prob_array = l1ace_prob_img.get_fdata().transpose(3, 0, 1, 2)

    # return dictionary of arrays
    return {
        "flair": flair_array,
        "gt_seg": gt_seg_array,
        "baseline_seg": baseline_seg_array,
        "baseline_prob": baseline_prob_array,
        "l1ace_seg": l1ace_seg_array,
        "l1ace_prob": l1ace_prob_array,
    }


def convert_to_pytorch_tensors(arrays):
    for key, value in arrays.items():
        arrays[key] = torch.tensor(value)
    return arrays


def convert_to_numpy(arrays):
    for key, value in arrays.items():
        arrays[key] = value.numpy()
    return arrays


def transform_to_brats_classes(arrays):
    arrays = convert_to_pytorch_tensors(arrays)
    transform = ConvertToBratsClassesd(keys=["gt_seg", "baseline_seg", "l1ace_seg"])
    arrays = transform(arrays)

    transform = ConvertToBratsClassesSoftmaxd(keys=["baseline_prob", "l1ace_prob"])
    arrays = transform(arrays)

    arrays = convert_to_numpy(arrays)

    return arrays


def extract_component(
    arrays, component=3, keys_to_extract=None, exclude_keys=["flair"]
):
    """Extracts a specified component from multiple arrays in a dictionary, with the
       option to provide keys to process and a list of keys to exclude.

    Args:
        arrays (dict): A dictionary containing arrays.
        component (int, optional): The component index to extract. Defaults to 3.
        keys_to_extract (list, optional): A list of keys specifying which arrays to
                                         process. If None, processes all keys.
                                         Defaults to None.
        exclude_keys (list, optional):  A list of keys to exclude from processing.
                                        Defaults to ['flair'].
    Returns:
        dict: The modified dictionary with components extracted.
    """

    if keys_to_extract:
        keys = keys_to_extract  # Use the specified list of keys
    else:
        keys = arrays.keys()  # Process all keys if none specified

    for key in keys:
        if key not in exclude_keys:
            arrays[key] = arrays[key][component]

    return arrays


import numpy as np


def find_roi_slice(arrays, key="gt_seg", padding=(10, 10, 10, 10), square=True):
    """Finds the region of interest (ROI) slice with flexible padding.

    Args:
        arrays (dict): A dictionary containing arrays.
        key (str, optional): The key to access the segmentation array. Defaults to 'gt_seg'.
        padding (tuple, optional): Padding in the format (top, bottom, left, right).
                                   Defaults to (10, 10, 10, 10).
        square (bool, optional): If True, returns a square ROI with padding. Defaults to True.

    Returns:
        tuple: A tuple of slices (col_slice, row_slice, slice_idx)
    """

    value = arrays[key]
    slice_areas = np.sum(value, axis=(0, 1))
    largest_slice_idx = np.argmax(slice_areas)
    value = value[:, :, largest_slice_idx]

    rows = np.any(value, axis=1)
    cols = np.any(value, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    top_pad, bottom_pad, left_pad, right_pad = padding

    rmin = max(rmin - top_pad, 0)
    rmax = min(rmax + bottom_pad, value.shape[0])
    cmin = max(cmin - left_pad, 0)
    cmax = min(cmax + right_pad, value.shape[1])

    if square:
        # Make the ROI square
        max_dim = max(rmax - rmin, cmax - cmin)  # Find the largest dimension
        center_r = (rmin + rmax) // 2  # Calculate center points
        center_c = (cmin + cmax) // 2

        half_side = max_dim // 2  # Size of the half-side of the square

        rmin = max(0, center_r - half_side)
        rmax = min(value.shape[0], center_r + half_side)
        cmin = max(0, center_c - half_side)
        cmax = min(value.shape[1], center_c + half_side)

    return slice(rmin, rmax, None), slice(cmin, cmax, None), largest_slice_idx


def crop_to_roi(arrays, roi):
    for key, value in arrays.items():
        arrays[key] = value[roi]
    return arrays


def calculate_and_plot_contours(
    prob_cropped,
    seg_cropped,
    gt_seg_cropped,
    min_size,
    cmap_prob,
    gamma_correction,
    legend_patches,
    TP_color,
    FP_color,
    FN_color,
    ax,
):

    im = ax.imshow(prob_cropped, cmap=cmap_prob)
    # im = ax.imshow(prob_cropped, cmap=cmap_prob, norm=gamma_correction)

    # Calculate TP, FP, FN, and remove small objects
    TP = np.logical_and(seg_cropped, gt_seg_cropped)
    FP = np.logical_and(seg_cropped, 1 - gt_seg_cropped)
    FN = np.logical_and(1 - seg_cropped, gt_seg_cropped)
    TP = remove_small_objects(TP, min_size=min_size)
    FP = remove_small_objects(FP, min_size=min_size)
    FN = remove_small_objects(FN, min_size=min_size)

    # Plot contours
    ax.contour(TP, colors=TP_color, linewidths=2, levels=[0.5])
    ax.contour(FP, colors=FP_color, linewidths=2, levels=[0.5])
    ax.contour(FN, colors=FN_color, linewidths=2, levels=[0.5])

    ax.legend(handles=legend_patches, loc="lower left")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return im


def plot_case_results(
    arrays,
    case,
    out_dir,
    min_size=3,
    cmap_seg="Set1",
    cmap_prob="YlOrRd",
    TP_color="green",
    FP_color="lightgray",
    FN_color="black",
):
    """Plots the segmentation results and probability maps with contours.

    Args:
        arrays (dict): Dictionary containing arrays ('flair', 'gt_seg', etc.)
        min_size (int): Minimum size for removing small objects.
        cmap_seg (str): Colormap for segmentation overlays.
        cmap_prob (str): Colormap for probability maps.
        TP_color (str): Color for true positive (TP) contours.
        FP_color (str): Color for false positive (FP) contours.
        FN_color (str): Color for false negative (FN) contours.
    """

    flair_cropped = arrays["flair"]
    gt_seg_cropped = arrays["gt_seg"]
    baseline_prob_cropped = arrays["baseline_prob"]
    l1ace_prob_cropped = arrays["l1ace_prob"]
    baseline_seg_cropped = arrays["baseline_seg"]
    l1ace_seg_cropped = arrays["l1ace_seg"]

    fig, axs = plt.subplots(
        1, 4, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
    )
    ax0, ax1, ax2, cax = axs

    # ... (Code for FLAIR image and segmentation overlay - Same as before) ...
    # Mask the areas where gt_seg_cropped is 0 to keep them transparent in the overlay
    masked_seg = np.ma.masked_where(gt_seg_cropped == 0, gt_seg_cropped)

    # Visualizing the FLAIR image with the binary segmentation overlay
    ax0.imshow(flair_cropped, cmap="gray", interpolation="nearest")
    # Apply the masked overlay with the 'Blues' colormap
    ax0.imshow(masked_seg, alpha=0.7, cmap=cmap_seg, interpolation="nearest")
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])

    gamma_correction = mcolors.PowerNorm(gamma=0.35)
    legend_patches = [
        Patch(color=TP_color, label="TP"),
        Patch(color=FP_color, label="FP"),
        Patch(color=FN_color, label="FN"),
    ]

    # Baseline Probability Map
    _ = calculate_and_plot_contours(
        baseline_prob_cropped,
        baseline_seg_cropped,
        gt_seg_cropped,
        min_size,
        cmap_prob,
        gamma_correction,
        legend_patches,
        TP_color,
        FP_color,
        FN_color,
        ax1,
    )

    # L1ACE Probability Map
    im2 = calculate_and_plot_contours(
        l1ace_prob_cropped,
        l1ace_seg_cropped,
        gt_seg_cropped,
        min_size,
        cmap_prob,
        gamma_correction,
        legend_patches,
        TP_color,
        FP_color,
        FN_color,
        ax2,
    )

    # ... (Code for adjusting colorbar - Same as before) ...
    # Adjust the colorbar position
    pos = ax2.get_position()
    cax.set_position([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(im2, cax=cax, orientation="vertical")
    # Ensure maximum data value is included
    if np.max(l1ace_prob_cropped) < 1.0:
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.savefig(os.path.join(out_dir, f"{case}.pdf"), bbox_inches="tight")
    plt.close(fig)


def process_case(
    case,
    out_dir,
    min_size,
    cmap_seg,
    cmap_prob,
    TP_color,
    FP_color,
    FN_color,
    component,
    border_size_2d,
):
    """Processes a single case (can be called in parallel)."""
    arrays = load_case(case)
    arrays = transform_to_brats_classes(arrays)
    arrays = extract_component(arrays, component=3)
    roi = find_roi_slice(arrays, key="gt_seg", padding=border_size_2d, square=True)
    arrays = crop_to_roi(arrays, roi)
    plot_case_results(
        arrays,
        case,
        out_dir,
        min_size,
        cmap_seg,
        cmap_prob,
        TP_color,
        FP_color,
        FN_color,
    )


if __name__ == "__main__":
    parent_dir = "../data/BraTS2021_TestingData"
    out_dir = "./bundle/runs/results_summary/seg_plots"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    entries = os.listdir(parent_dir)
    case_dirs = [
        entry for entry in entries if os.path.isdir(os.path.join(parent_dir, entry))
    ]

    # case_dirs = ['BraTS2021_00014',]

    cmap_seg = ListedColormap(
        [
            (0.0, 0.5, 0.0),
        ]
    )
    cmap_prob = plt.get_cmap("PRGn")
    min_size = 10
    TP_color = "blue"
    FP_color = "yellow"
    FN_color = "red"
    component = "wt"
    border_size_2d = (0, 10, 6, 8)  # (top, bottom, left, right)

    num_cores = 6  # Adjust based on your system

    with mp.Pool(processes=num_cores) as pool:
        args_list = [
            (
                case,
                out_dir,
                min_size,
                cmap_seg,
                cmap_prob,
                TP_color,
                FP_color,
                FN_color,
                component,
                border_size_2d,
            )
            for case in case_dirs
        ]
        results = pool.starmap(process_case, args_list)
