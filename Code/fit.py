"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0 #fix: changed the variable name sum to avoid shadowing the built-in function names
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad() # fix: added zero_grad() to clear gradients before backpropagation
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, (correct / total) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0

        all_labels = [] # fix: added variables to store all labels and predictions for further analysis
        all_preds = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_labels.extend(labels.cpu().numpy())  # Extend the list with the true labels
                all_preds.extend(predicted.cpu().numpy())  # Extend the list with the predicted labels

        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0) # fix: added precision, recall, and f1_score calculations for better evaluation metrics

        return running_loss / total, (correct / total) * 100, precision * 100, recall * 100, f1_score * 100

                
        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, epochs, dataset_name = "dataset", patience = 7): # fix: added dataset_name parameter to the fit method for better logging
        print("\n Starting Training Routine...")
        print("-" * 50)

        best_val_loss = float('inf') # Track the best loss
        best_model_weights = None  # Track the best model weights

        train_losses, val_losses = [], []

        epochs_no_improve = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.evaluate(val_loader)

            train_losses.append(train_loss) #tracking the train loss for plotting
            val_losses.append(val_loss) #tracking the validation loss for plotting

            #save the best model weights based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = self.model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1 # as we are tracking the number of epochs without improvement in validation loss
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            
            # Early stopping condition
            if epochs_no_improve >= patience:  
                print(f"Early stopping Triggered after {patience} epochs without improvement.")
                break #break the training loop if no improvement in validation loss for 'patience' epochs

        print("-" * 50)
        print("Training Complete!")

    def plot_losses(self, train_losses, val_losses, dataset_name):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(train_losses, color='#005564', linewidth=2, label='Train Loss')
        ax.plot(val_losses, color='#FF6A00', linewidth=2, label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cross Entropy Loss')
        ax.set_title(f'Training Curve - {dataset_name}', color='#163C69', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_loss_curve.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
