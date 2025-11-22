import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from utilities import train, evaluate, save_model
from data import get_datasets
from models.cnn import CNN
from models.vit import ViT
from models.convnext import ConvNextTiny
import time
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

def new_forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = self.relu(out)

    return out

BasicBlock.forward = new_forward

def fix_inplace_modules(model):
    """
    Iterate through all modules of the model and set inplace=False for any module that has this attribute.
    This is necessary for Opacus compatibility.
    """
    print("Running fix_inplace_modules...")
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'inplace'):
            # print(f"Found module with inplace attribute: {name}, value: {module.inplace}")
            if module.inplace:
                module.inplace = False
                print(f"Disabled inplace for module: {name}")
                count += 1
    print(f"Total modules fixed: {count}")

def main():
    # Load config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Argument parser
    parser = argparse.ArgumentParser(description="CIFAR-10 Classification")
    
    # Flags for hyperparameters
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=config["learning_rate"], help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=config["epochs"], help="Number of training epochs")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Enable mixed precision training")    
    parser.add_argument("--model_type", type=str, default=config["model_type"], choices=["cnn", "vit", "convnext"], help="Model type to train (cnn or vit)")
    parser.add_argument("--save_weights", action="store_true", help="Save model weights")
    parser.add_argument("--save_model_path", type=str, default=config["save_model_path"], help="Path to save the trained model")
    parser.add_argument("--disable_dp", action="store_true", default=config.get("disable_dp", False), help="Disable Differential Privacy training")


    args = parser.parse_args()
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_data, test_data = get_datasets()
    # Optimization: Use num_workers and pin_memory
    num_workers = 4
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)


    if args.model_type == "cnn":
        model = CNN()
    elif args.model_type == "vit":
        model = ViT(num_classes=10)
    elif args.model_type == "convnext":
        model = ConvNextTiny(num_classes=10)
    else:
        raise ValueError("Invalid model type in config. Choose 'cnn' or 'vit'.")
    
    if not args.disable_dp:
        # Disable inplace operations BEFORE validation
        fix_inplace_modules(model)

        errors = ModuleValidator.validate(model, strict=False)
        print("ERRORS: ", errors[-5:])

        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
        
        # Disable inplace operations AGAIN after validation just in case
        fix_inplace_modules(model)
    
    model.to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    if not args.disable_dp:
        MAX_GRAD_NORM = 1.2
        EPSILON = 50.0
        DELTA = 1e-5

        privacy_engine = PrivacyEngine()

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
    else:
        print("Differential Privacy Disabled")


    # Mixed precision
    # Use torch.amp.GradScaler if available (PyTorch 1.10+), else fallback
    if args.use_mixed_precision:
        try:
            scaler = torch.cuda.amp.GradScaler()
        except:
            # Fallback or for MPS if supported in future versions
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            print("Warning: GradScaler not initialized properly, check PyTorch version/device support.")
    else:
        scaler = None

    mixed_precision_status = "Enabled" if scaler else "Disabled"
    print(f"Mixed Precision Training: {mixed_precision_status}")
    
    # Training loop
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, device, scaler, epoch)
        evaluate(model, test_loader, criterion, device)
    
    # Save model
    if(args.save_weights == True):
        save_model(model, args.save_model_path)
        print("Model saved.")

    print("Training complete.")

if __name__ == "__main__":

    # Record the start time
    start_time = time.time()

    main()

    # Record the end time
    end_time = time.time()
    
    # Calculate and display the total elapsed time
    total_time = end_time - start_time
    # print(f"Total execution time: {total_time:.2f} seconds")
    # print(f"Total execution time: {total_time:.2f/60} minutes")

    minutes, seconds = divmod(total_time, 60)
    print(f"Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds")
