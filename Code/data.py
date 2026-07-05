"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}.pt"  # path issue fixed.....
    data_dict = torch.load(d_path, map_location=torch.device('cpu'), weights_only=False)  # Ensure data is loaded on CPU

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
    def preprocess(x, y):
        x = x.float() 
        if x.max() > 1.0:  # Check if normalization is needed
            x /= 255.0  # Normalize to [0, 1]
        
        if len(x.shape) == 3:  # If single channel, add channel dimension
            x = x.unsqueeze(1)  # Add channel dimension for grayscale images

        if x.shape[-1] < 32 or x.shape[-2] < 32:  # If images are smaller than 32x32, resize them
            x = torch.nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

        y = y.squeeze().long()
        return x, y
    
    # Apply preprocessing to train, validation, and test datasets
    train_data, train_labels = preprocess(train_data, train_labels)
    val_data, val_labels = preprocess(val_data, val_labels)
    test_data, test_labels = preprocess(test_data, test_labels)

    class AugmentedDataset(torch.utils.data.Dataset):
    
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.transform(self.images[idx])
            return image, self.labels[idx]
    
    train_dataset = AugmentedDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader