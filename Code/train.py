"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.3):
    """Zero out the weakest weights in every Conv layer."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
    print(f"\n[GREEN] Pruning done: weakest {int(amount * 100)}% of weights zeroed out.")
    return model
    
def remove_pruning_masks(model):
    """Make the pruning permanent by baking the zeros into the weights."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    print("[GREEN] Pruning is now permanent.")
    return model

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    model_to_run = config.get("MODELS", [config.get("MODEL")])

    for model_name in model_to_run:
        print(f"\nTraining model: {model_name}")
        config["MODEL"] = model_name  # Update the model in the config for each iteration

        for dataset_config in config["DATASETS"]:
            data_name = dataset_config["DATA"]
            channels = dataset_config["CHANNELS"]
            num_classes = dataset_config["NUM_CLASSES"] # enhancement:added the configuration for the model to use.

            train_loader, val_loader, test_loader = get_loaders(data=data_name,  # fix: fixed syntax error for var test_loader #fix: corrected the data parameter
                                                        data_path=config["DATA_PATH"], 
                                                        batch_size=config["BATCH_SIZE"],
                                                        val_split=config.get("VAL_SPLIT", 0.1)) #added val_split

            model_class = getattr(models, config["MODEL"])
            model = model_class(in_channels=channels, num_classes=num_classes).to(device) #fix: removed the 0.99 dropout and intialized the model with the correct number of input channels and classes
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5,)  # fix: added a learning rate scheduler to reduce the learning rate on plateau


            trainer = Trainer(model, criterion, optimizer, device, scheduler)
            trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"], dataset_name=data_name)  # fix: added dataset_name parameter to the fit method for better logging

            # Pruning: only for GreenResNet18
            if model_name == "GreenResNet18":
                # Step 1: Prune the weakest 30% of weights
                model = apply_pruning(model, amount=config.get("PRUNE_AMOUNT", 0.3))
                # Step 2: Fine-tune to recover accuracy
                finetune_epochs = config.get("FINETUNE_EPOCHS", 5)
                print(f"[GREEN] Fine-tuning for {finetune_epochs} epochs to recover accuracy...")
                fine_optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"] * 0.1)
                fine_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fine_optimizer, mode='min', patience=3, factor=0.5)
                fine_trainer = Trainer(model, criterion, fine_optimizer, device, fine_scheduler)
                fine_trainer.fit(train_loader, val_loader, epochs=finetune_epochs, dataset_name=f"{data_name}_finetuned")
                # Step 3: Make pruning permanent
                model = remove_pruning_masks(model)

            test_loss, test_accuracy, precision, recall, f1_score, test_latency, test_inf_mem = trainer.evaluate(test_loader)
            print(f"\n{'='*50}")
            print(f"Model: {model_name} | Dataset: {data_name}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print(f"Precision: {precision:.2f}%")
            print(f"Recall: {recall:.2f}%")
            print(f"F1-Score: {f1_score:.2f}%")
            print(f" Latency: {test_latency:.4f} ms/ sample")
            print(f" Inference Memory: {test_inf_mem:.2f} MB")
            print(f"{'='*50}")

if __name__ == "__main__":
    main()
