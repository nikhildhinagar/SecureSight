# SecureSight: Image Classification with Differential Privacy

This project implements image classification on the CIFAR-10 dataset using various deep learning architectures (CNN, Vision Transformer, ConvNeXt). It features integration with **Opacus** for training with Differential Privacy (DP), ensuring that the trained models respect user privacy.

## Why Differential Privacy Matters?

In the era of large-scale AI, models are often trained on sensitive user data (e.g., medical records, financial data, personal images). **Differential Privacy (DP)** provides a mathematical guarantee that the output of a model (or the model itself) does not reveal whether any specific individual's data was included in the training set.

-   **Privacy Guarantee**: Ensures that an attacker cannot reverse-engineer the training data to extract private information.
-   **Regulatory Compliance**: Helps meet requirements for data protection regulations like GDPR and HIPAA.
-   **Trust**: Builds user trust by demonstrating a commitment to data privacy.

This project uses **Opacus**, a library that enables training PyTorch models with differential privacy with minimal code changes.

## Features

- **Multiple Architectures**: Support for CNN, Vision Transformer (ViT), and ConvNeXt Tiny.
- **Differential Privacy**: Optional DP training using `opacus` to protect training data privacy.
- **Mixed Precision Training**: Support for mixed precision to speed up training and reduce memory usage.
- **Configurable**: Easy configuration via `config.yaml` or command-line arguments.
- **MPS/CUDA Support**: Automatically detects and uses Apple Metal Performance Shaders (MPS) or CUDA if available.

## Project Structure

- `main.py`: Entry point for training and evaluation.
- `config.yaml`: Configuration file for default hyperparameters.
- `data.py`: Handles data loading and preprocessing for CIFAR-10.
- `utilities.py`: Contains training, evaluation, and model saving loops.
- `models/`: Directory containing model definitions (`cnn.py`, `vit.py`, `convnext.py`).
- `tests/`: Unit tests for the project.

## Installation

Ensure you have Python installed. Install the required dependencies:

```bash
pip install torch torchvision opacus pyyaml
```

## Usage

### Basic Usage

Run the training script with default settings (defined in `config.yaml`):

```bash
python main.py
```

### Custom Configuration

You can override default settings using command-line arguments:

```bash
python main.py --model_type vit --epochs 5 --batch_size 32 --learning_rate 0.0005
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Batch size for training and validation | `64` |
| `--learning_rate` | Learning rate for optimizer | `0.001` |
| `--epochs` | Number of training epochs | `2` |
| `--model_type` | Model architecture (`cnn`, `vit`, `convnext`) | `cnn` |
| `--use_mixed_precision` | Enable mixed precision training | `False` |
| `--save_weights` | Flag to save model weights after training | `False` |
| `--save_model_path` | Path to save the trained model | `trained_model.pth` |
| `--disable_dp` | Disable Differential Privacy training | `False` |

### Differential Privacy

Differential Privacy is enabled by default. To disable it (e.g., for standard training benchmarks), use the `--disable_dp` flag:

```bash
python main.py --disable_dp
```

## Expected Results

Upon running the script, you will see:
1.  **Device Information**: The script will print the device being used (e.g., `mps`, `cuda`, or `cpu`).
2.  **Training Logs**: Progress bars or logs showing the loss and accuracy for each epoch.
3.  **Privacy Budget**: If DP is enabled, it will print the privacy budget (epsilon) used.
4.  **Evaluation**: Final accuracy on the test set.
5.  **Execution Time**: Total time taken for the training process.

Example Output:
```text
Using device: mps
Files already downloaded and verified
Files already downloaded and verified
Using sigma=1.0 and C=1.2
Mixed Precision Training: Disabled
Epoch 1/2: 100%|...| Loss: 1.85 Acc: 32.5%
Epoch 2/2: 100%|...| Loss: 1.50 Acc: 45.2%
Training complete.
Total execution time: 2 minutes and 15.30 seconds
```

## License Notice

All rights reserved.
Please see the LICENSE file for terms and restrictions.

