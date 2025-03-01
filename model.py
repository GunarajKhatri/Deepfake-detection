import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Define the DeepfakeDetector model
class DeepfakeDetector(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.5):
        super(DeepfakeDetector, self).__init__()

        # Load Pretrained ResNet with Correct Weights
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer

        # Unfreeze ResNet for Fine-Tuning
        for param in self.resnet.parameters():
            param.requires_grad = True  # Enable gradient updates

        # LSTM for Temporal Processing
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        self.final_dropout = nn.Dropout(p=dropout)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        # Resize transformation (Applied before entering the model)
        self.resize = transforms.Resize((224, 224))
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Reshape for ResNet Processing
        x = x.view(B * T, C, H, W)
        x = self.resize(x)
        
        # Feature Extraction
        with torch.no_grad():
            x = self.resnet(x)
        x = x.view(B, T, 512)
        
        # LSTM Processing
        lstm_out, _ = self.lstm(x)
        
        # Extract last frame's LSTM output
        lstm_out = self.final_dropout(lstm_out)
        last_out = lstm_out[:, -1, :]
        
        # Classification (Raw logits, no Softmax)
        return self.fc(last_out)
