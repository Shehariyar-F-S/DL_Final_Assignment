"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
import torch.nn as nn
import os

activation_str = "ReLU"  # Placeholder for activation function, can be replaced with "ReLU" or others as needed.


class VGGBlock(nn.Module):
    """Modular VGG block with configurable number of conv layers and channels.

    C configuration from Simonyan & Zisserman's VGG paper.
    """
    def __init__(self, in_channels, out_channels, num_convs, padding=1):
        super().__init__()
        layers = []
        current_in_channels = in_channels
        for i in range(num_convs):
            is_config_c_tail = (num_convs == 3 and i == 2)
            kernel_size = 1 if is_config_c_tail else 3
            actual_padding = 0 if kernel_size == 1 else padding
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, padding=actual_padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_in_channels = out_channels #fix: Update in_channels for the next layer
            
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """ResBlock with 3x3 convolutions (He et al., 2016)."""
    def __init__(self, in_channels, out_channels, activation, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If spatial size shrinks (stride > 1) or channels change, adjust the shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        out = self.activation(out)
        return out


class AlexNet(nn.Module):
    """AlexNet (Krizhevsky et al., 2012) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs): #fix:added the missing arguments in_channels and num_classes.
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3), # fix: changed the hardcoded input channels to in_channels for flexibility
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # fix: added adaptive average pooling to ensure consistent output size before the classifier
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(192 * 2 * 2, 1024), #fix: shape mispatch corrected to match the output of the avgpool layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)  # fix: changed the hardcoded output classes to num_classes for flexibility,
        )

    def forward(self, x):
        x = self.features(x)
        x= self.avgpool(x)  # fix: added adaptive average pooling to ensure consistent output size before the classifier
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG16(nn.Module):
    """VGG16 in C configuration of Simonyan & Zisserman, (2014) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)

        self.features = nn.Sequential(
            VGGBlock(in_channels, 64, num_convs=2),
            VGGBlock(64, 128, num_convs=2),
            VGGBlock(128, 256, num_convs=3),
            VGGBlock(256, 512, num_convs=3),
            VGGBlock(512, 512, num_convs=3)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # fix: added adaptive average pooling to ensure consistent output size before the classifier
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # fix: added adaptive average pooling to ensure consistent output size before the classifier
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet18(nn.Module):
    """ResNet18 (He et al., 2016) adapted for smaller inputs.
    
    activation - flexible activation function to allow experimentation (e.g., ReLU, LeakyReLU, etc.)
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        activation = getattr(nn, activation_str)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = activation(inplace=True)
        print("Using activation function:", self.activation)
        
        self.stage1 = nn.Sequential(
            ResBlock(64, 64, activation(inplace=True), stride=1),
            ResBlock(64, 64, activation(inplace=True), stride=1)
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, activation(inplace=True), stride=2),          
            ResBlock(128, 128, activation(inplace=True), stride=1)
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, activation(inplace=True), stride=2),
            ResBlock(256, 256, activation(inplace=True), stride=1)
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, 512, activation(inplace=True), stride=2),
            ResBlock(512, 512, activation(inplace=True), stride=1)
        )
        
        drop_rate = kwargs.get("drop_rate", 0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out) #fix: added return statement to ensure the output of the classifier is returned

# =============================================================================
# Part 3: Transfer Learning Architectures
# =============================================================================
class FrozenTransferResNet18(nn.Module):
    """Transfer Learning Benchmark B: All Convolutional Layers Frozen."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        
        # 1. Instantiate OUR custom base model with the ORIGINAL 'orgs' configuration (11 classes)
        self.base_model = ResNet18(in_channels=1, num_classes=11) 
        
        # 2. Load our custom medical weights!
        weights_path = "orgs_pretrained_resnet18.pth"
        if os.path.exists(weights_path):
            self.base_model.load_state_dict(torch.load(weights_path))
            print(f"[TRANSFER] Successfully loaded pre-trained weights from {weights_path}")
        else:
            print(f"[WARNING] Pre-trained weights {weights_path} not found! You must run 'orgs' first.")

        # 3. FREEZE the entire convolutional body
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 4. Swap the classification head for the new 'organs' dataset
        drop_rate = kwargs.get("drop_rate", 0.5)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

class FineTunedTransferResNet18(nn.Module):
    """Transfer Learning Benchmark C: Stage 4 Unfrozen (Partial Fine-Tuning)."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        
        self.base_model = ResNet18(in_channels=1, num_classes=11) 
        
        weights_path = "orgs_pretrained_resnet18.pth"
        if os.path.exists(weights_path):
            self.base_model.load_state_dict(torch.load(weights_path))
            print(f"[TRANSFER] Successfully loaded pre-trained weights from {weights_path}")

        # 1. Freeze everything first
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 2. UNFREEZE ONLY STAGE 4
        for param in self.base_model.stage4.parameters():
            param.requires_grad = True
            
        # 3. Swap the head
        drop_rate = kwargs.get("drop_rate", 0.5)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
