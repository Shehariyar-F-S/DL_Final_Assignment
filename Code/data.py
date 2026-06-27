"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}.pt"  # path issue fixed.....
    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    #shuffling the data to ensure randomness before splitting into train and validation sets
    torch.manual_seed(42)  # for reproducibility
    indices = torch.randperm(total_samples)
    shuffled_images = data_dict['train_images'][indices]
    shuffled_labels = data_dict['train_labels'][indices]

    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    train_data = shuffled_images[:val_start]   # data leak issue fixed.....
    train_labels = shuffled_labels[:val_start]
    val_data = shuffled_images[val_start:]
    val_labels = shuffled_labels[val_start:]

    test_data = data_dict['test_images']
    test_labels = data_dict['test_labels']

    #Preprocess the data: Convert to float and normalize to [0, 1]
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader