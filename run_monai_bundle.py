import argparse
from monai.bundle.scripts import run


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run a MONAI bundle script with specified configurations."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train",
            "inference_predict",
            "inference_evaluation",
            "temperature_scaling",
            "temperature_scaling_eval",
        ],
        help="Operation mode.",
    )
    parser.add_argument(
        "--sys",
        type=str,
        required=True,
        choices=["low", "med", "high"],
        help="System specification.",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Dataset name, e.g., brats_2021."
    )
    parser.add_argument(
        "--model", type=str, help="Model configuration name, e.g., baseline_ce."
    )
    return parser


def get_config_files(args):
    # Common configs:
    config_files = [
        "bundle/configs/common.yaml",
        f"bundle/configs/data/{args.data}.yaml",
        f"bundle/configs/sys/{args.sys}_spec.yaml",
    ]

    mode_specific_configs = {
        "train": ["train", "validation", f"train/{args.model}"],
        "inference_predict": ["inference_predict"],
        "inference_evaluation": ["inference_eval"],
        "temperature_scaling": ["temp_scale"],
        "temperature_scaling_eval": ["inference_eval", "temp_scale_eval"],
    }

    config_files.extend(
        [f"bundle/configs/{file}.yaml" for file in mode_specific_configs[args.mode]]
    )
    return config_files


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == "train" and not args.model:
        raise ValueError("Model configuration name is required for training.")

    config_files = get_config_files(args)
    model_name = f"{args.model}_{args.data}_{args.sys}"
    model_name += "_temp_scaled" if args.mode == "temperature_scaling_eval" else ""

    run(
        bundle_root="./bundle",
        meta_file="./bundle/configs/metadata.json",
        config_file=config_files,
        logging_file="./bundle/configs/logging.conf",
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
