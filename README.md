<div align="center">

# ACE-DLIRIS
# Average Calibration Error
# A Differentiable Loss for Improved Reliability in Image Segmentation
</div>

All trained models and evaluation metrics can be found in `bundle/runs`.

## Abstract:

Deep neural networks for medical image segmentation often produce overconfident results misaligned with empirical observations. Such miscalibration challenges their clinical translation. We propose to use marginal L1 average calibration error (mL1-ACE) as a novel auxiliary loss function to improve pixel-wise calibration without compromising segmentation quality. We show that this loss, despite using hard binning, is directly differentiable, bypassing the need for approximate but differentiable surrogate or soft binning approaches. Our work also introduces the concept of *dataset reliability histograms* which generalizes standard reliability diagrams for refined visual assessment of calibration in semantic segmentation aggregated at the dataset level. Using mL1-ACE, we reduce average and maximum calibration error by 45% and 55% respectively, maintaining a Dice score of 87% on the BraTS 2021 dataset.

[PAPER]

## Data Setup

To ensure smooth operation of the project, it's crucial to set up the data directory correctly. The data directory should be placed at the same level as the project directory, not inside it. Here's an example structure for clarity:

```plaintext
Documents/
├── data/
│   ├── BraTS2021_TestingData/
│   └── BraTS2021_TrainingValidationData/
└── ACE-DLIRIS/
```

Inside the `data` directory, there should be two main folders:

- `BraTS2021_TrainingValidationData`: Contains the data used for training and validation purposes.
- `BraTS2021_TestingData`: Contains the data used for testing the model's performance.

Additionally, the specific cases that are included for training, validation, and testing are listed in two text files:

- `brats21_train_val.txt`: Lists the cases used for training and validation.
- `brats21_test.txt`: Lists the cases used for testing.

Ensure these files are placed in an accessible location within the `data` directory and are correctly referenced in your project configuration or code. This setup is crucial for the proper functioning of the training, validation, and testing processes.

## Usage
### Docker

Using Docker ensures a consistent environment for running the project, regardless of the host system. Follow the steps below to set up and use Docker for this project:

1. **Build the Docker Container**

   Before running any commands, you need to build the Docker container. You can do this by executing the `docker_build.sh` script located in the `docker` directory. This script builds a Docker image based on the NVIDIA CUDA base image, setting up an environment suitable for running the project.

   ```bash
   ./docker/docker_build.sh
   ```

   This script uses the following parameters to build the Docker image:

   - `docker_tag`: A tag or name for the Docker image, set to `${USER}/dliris:latest` by default.
   - `docker_base`: The base Docker image, set to `nvidia/cuda:11.8.0-runtime-ubuntu22.04` for CUDA support.
   - User and group IDs to ensure file permissions match between the Docker container and the host system.

   After running the script, a Docker image will be created, which you can then use to run the project's Docker commands.

2. **Run the Project Using Docker**

   With the Docker image built, you can now run the project within the Docker container. Use the `docker_run.sh` script to start the container and execute the project's tasks. Replace the placeholder values in the command below with your specific configurations:

   ```bash
   scripts/docker_run.sh --mode <mode> --sys <system_setting> --data <dataset> --model <model_name> [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>]
   ```

   Refer to the [Command-Line Arguments](#command-line-arguments) section for details on each argument.

3. **Optional: Push the Docker Image to a Registry**

   If you wish to share your Docker image or use it across multiple machines, you can push the built image to a Docker registry such as Docker Hub. Uncomment and use the `docker push` command in the `docker_build.sh` script to upload your image.

   ```bash
   # docker push "${docker_tag}"
   ```

   Ensure that you have the necessary permissions and that you're logged in to the Docker registry before pushing the image.


This section guides users through building the Docker container necessary for running the project and includes optional steps for sharing the Docker image via a registry.

### Using Docker
Set up your environment using Docker to ensure consistency across different setups. Follow these instructions to get started:

```bash
# Training
scripts/docker_run.sh --mode train --sys <system_setting> --data <dataset> --model <model_name> [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>]

# Predicting
scripts/docker_run.sh --mode inference_predict --sys <system_setting> --data <dataset> [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>]

# Evaluating
scripts/docker_run.sh --mode inference_evaluation --sys <system_setting> --data <dataset> [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>]

# Temperature Scaling
scripts/docker_run.sh --mode temperature_scaling --sys <system_setting> --data <dataset> [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>]

# Temperature Scale Evaluation
scripts/docker_run.sh --mode temperature_scaling_eval --sys <system_setting> --data <dataset> [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>]
```

### Using `run_monai_bundle` Directly
If you prefer to run the MONAI bundle directly without Docker, use the following commands:

```bash
# Training
python run_monai_bundle.py --mode train --sys <system_setting> --data <dataset> --model <model_name>

# Predicting
python run_monai_bundle.py --mode inference_predict --sys <system_setting> --data <dataset>

# Evaluating
python run_monai_bundle.py --mode inference_evaluation --sys <system_setting> --data <dataset>

# Temperature Scaling
python run_monai_bundle.py --mode temperature_scaling --sys <system_setting> --data <dataset>

# Temperature Scale Evaluation
python run_monai_bundle.py --mode temperature_scaling_eval --sys <system_setting> --data <dataset>
```

Replace `<mode>`, `<system_setting>`, `<dataset>`, and `<model_name>` with your specific configurations. The `<gpu_number>`, `<cpus>`, and `<shm_size>` are optional parameters for Docker that you can adjust based on your hardware.

## Command-Line Arguments

When using the `run_monai_bundle.py` script or the Docker command, the following arguments can be specified to customize the execution:

- `--mode`: Specifies the operation mode of the script. The available modes are:
  - `train`: Train the model using the specified dataset and model configuration.
  - `inference_predict`: Run inference to generate predictions using a trained model on the specified dataset.
  - `inference_evaluation`: Evaluate the model's performance on the specified dataset by comparing predictions against ground truth.
  - `temperature_scaling`: Apply temperature scaling to calibrate the model's confidence on the validation set.
  - `temperature_scaling_eval`: Evaluate the temperature scaled model's performance on the specified dataset

- `--sys`: Defines the system specification. The values relating to training are:

         | System Spec | VRAM       | Cores       | RAM          |
         |-------------|------------|-------------|--------------|
         | Low         | 6 GB       | 2-6         | 16 GB        |
         | Medium      | 12 GB      | 4-8         | 32 GB        |
         | High        | 32 GB      | 16-32       | 128 GB       |

- `--data`: The name of the dataset to be used for training, inference, or evaluation. For example, specifying `brats_2021` would use the BraTS 2021 dataset for the operation.

- `--model`: The name of the model configuration (loss) to be used during training. This specifies the loss function. The options are:
   - `baseline_ce`: Model trained with cross-entropy
   - `baseline_dice`: Model trained with dice
   - `baseline_dice_ce`: Model trained with dice + cross-entropy (1:1)
   - `hardl1ace_ce`: Model trained with mL1-ACE + cross-entropy (1:1)
   - `hardl1ace_dice`: Model trained with mL1-ACE + dice loss (1:1)
   - `hardl1ace_dice_ce`: Model trained with mL1-ACE + dice + cross-entropy (1:1:1)

Note: The `--model` argument is required when the `--mode` is set to `train`.

Replace the placeholder values with your specific configurations to tailor the command to your project's needs.


## Running Unit Tests

We use `pytest` for unit testing to ensure the reliability and correctness of the code. You can run the tests to verify that changes or additions to the codebase maintain the expected functionality.

To run all unit tests, navigate to the root directory of the project and execute the following command:

```bash
pytest ace_dliris/tests
```

This command runs all test files located in the `ace_dliris/tests` directory, ensuring comprehensive coverage of the codebase.

If you wish to run tests for a specific module, you can specify the path to the test file. For example, to run tests for the BRATS transforms module, you would use:

```bash
pytest ace_dliris/tests/test_brats_transforms.py
```

Replace `test_brats_transforms.py` with the appropriate test file name to run tests for different modules.

### Test Organization

The tests are organized as follows:

- `test_brats_transforms.py`: Tests for BRATS dataset-specific transformations.
- `test_handlers.py`: Tests for various event handlers used during training and evaluation.
- `test_losses.py`: Tests for custom loss functions implemented for the model.
- `test_metrics.py`: Tests for evaluation metrics used to assess model performance.


## Using `.devcontainer` for Development

For those who use Visual Studio Code (VS Code), our project supports development inside a Docker container using a `.devcontainer` setup. This allows you to have a consistent, isolated, and reproducible development environment, which mirrors the configuration used for running the project.

Before you get started, ensure your system meets the following prerequisites:

- **NVIDIA Drivers and Containers:**
  - NVIDIA Driver -- With CUDA support
  - `nvidia-container-toolkit`
  - `nvidia-docker2`

  These packages are essential for leveraging NVIDIA GPUs within Docker containers. They allow Docker containers to access the full GPU capabilities for computation.

- **Docker:**
  Ensure Docker is installed and running on your system. Docker is used to create, manage, and run our containers.

- **Visual Studio Code:**
  Download and install Visual Studio Code (VS Code) from the [official website](https://code.visualstudio.com/). VS Code is a lightweight but powerful source code editor that runs on your desktop.

- **Remote - Containers extension for VS Code:**
  Install the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code. This extension lets you use a Docker container as a full-featured development environment.

With these prerequisites in place, you can proceed with setting up the project environment using Docker and the `.devcontainer` setup for a seamless development experience.
