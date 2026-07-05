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

    train_loader, val_loader, test_loader = get_loaders(data=config["DATA"],  # fix: fixed syntax error for var test_loader
                                                        data_path=config["DATA_PATH"], 
                                                        batch_size=config["BATCH_SIZE"])

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=0.99, activation_str=None).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

if __name__ == "__main__":
    main()