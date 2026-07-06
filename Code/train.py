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
            trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"], dataset_name=data_name, )  # fix: added dataset_name parameter to the fit method for better logging

            test_loss, test_accuracy, precision, recall, f1_score, test_latency = trainer.evaluate(test_loader)
            print(f"\n{'='*50}")
            print(f"Model: {model_name} | Dataset: {data_name}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print(f"Precision: {precision:.2f}%")
            print(f"Recall: {recall:.2f}%")
            print(f"F1-Score: {f1_score:.2f}%")
            print(f" Latency: {test_latency:.4f} ms/ sample")
            print(f"{'='*50}")

if __name__ == "__main__":
    main()