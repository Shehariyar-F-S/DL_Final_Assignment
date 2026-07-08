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

# =============================================================================
# PART 3: 3-Way Transfer Learning Benchmark Matrix
# =============================================================================
def run_organs_benchmark(config, device, train_loader, val_loader, test_loader, channels, num_classes):
    print("\n" + "="*70)
    print(" PART 3 — SCARCE-DATA BENCHMARK MATRIX | Dataset: organs")
    print("="*70)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    def train_and_eval(model, name):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["LEARNING_RATE"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        trainer = Trainer(model, criterion, optimizer, device, scheduler)
        trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"], dataset_name=name)
        test_loss, test_acc, prec, rec, f1 = trainer.evaluate(test_loader)
        return test_loss, test_acc, prec, rec, f1
        
    # --- Benchmark A: Scratch ---
    
    print("\n[PART 3 - A] Training ResNet18 from SCRATCH on 'organs'...")
    model_a = models.ResNet18(in_channels=channels, num_classes=num_classes).to(device)
    a_loss, a_acc, a_prec, a_rec, a_f1 = train_and_eval(model_a, "organs_scratch")
    
    # --- Benchmark B: All Frozen ---
    
    print("\n[PART 3 - B] Training Transfer Model (ALL FROZEN)...")
    model_b = models.FrozenTransferResNet18(in_channels=channels, num_classes=num_classes).to(device)
    b_loss, b_acc, b_prec, b_rec, b_f1 = train_and_eval(model_b, "organs_all_frozen")
    
    # --- Benchmark C: Stage 4 Unfrozen ---
    
    print("\n[PART 3 - C] Training Transfer Model (STAGE 4 UNFROZEN)...")
    model_c = models.FineTunedTransferResNet18(in_channels=channels, num_classes=num_classes).to(device)
    c_loss, c_acc, c_prec, c_rec, c_f1 = train_and_eval(model_c, "organs_partial_unfreeze")
    print("\n" + "="*70)
    print(" PART 3 — FINAL RESULTS MATRIX | Dataset: organs")
    print("="*70)
    print(f"{'Metric':<20} {'Scratch (A)':<15} {'All Frozen (B)':<20} {'Stage4 Unfrozen (C)'}")
    print("-" * 70)
    print(f"{'Test Accuracy':<20} {a_acc:>8.2f}% {b_acc:>15.2f}% {c_acc:>18.2f}%")
    print(f"{'Precision':<20} {a_prec:>8.2f}% {b_prec:>15.2f}% {c_prec:>18.2f}%")
    print(f"{'Recall':<20} {a_rec:>8.2f}% {b_rec:>15.2f}% {c_rec:>18.2f}%")
    print(f"{'F1-Score':<20} {a_f1:>8.2f}% {b_f1:>15.2f}% {c_f1:>18.2f}%")
    print(f"{'Test Loss':<20} {a_loss:>8.4f} {b_loss:>16.4f} {c_loss:>19.4f}")
    print("="*70)


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

            # PART 3 If dataset is organs, run the Matrix

            if data_name == "organs" and model_name == "ResNet18":
                run_organs_benchmark(config, device, train_loader, val_loader, test_loader, channels, num_classes)
                continue
            model_class = getattr(models, config["MODEL"])
            model = model_class(in_channels=channels, num_classes=num_classes).to(device) #fix: removed the 0.99 dropout and intialized the model with the correct number of input channels and classes
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # fix: added label smoothing to the loss function to improve generalization
            optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5,)  # fix: added a learning rate scheduler to reduce the learning rate on plateau


            trainer = Trainer(model, criterion, optimizer, device, scheduler)
            trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"], dataset_name=data_name, )  # fix: added dataset_name parameter to the fit method for better logging

            # PART 3 SAVER: Save orgs weights so organs can use them

            if data_name == "orgs" and model_name == "ResNet18":
                weights_path = "orgs_pretrained_resnet18.pth"
                torch.save(model.state_dict(), weights_path)
                print(f"\n[TRANSFER] Saved {weights_path} to be used by 'organs' dataset!")
            
            test_loss, test_accuracy, precision, recall, f1_score = trainer.evaluate(test_loader)
            print(f"\n{'='*50}")
            print(f"Model: {model_name} | Dataset: {data_name}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print(f"Precision: {precision:.2f}%")
            print(f"Recall: {recall:.2f}%")
            print(f"F1-Score: {f1_score:.2f}%")
            print(f"{'='*50}")

if __name__ == "__main__":
    main()
