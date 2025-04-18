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

        # Freeze Early Layers for Better Transfer Learning
        for param in list(self.resnet.children())[:6]:
            for p in param.parameters():
                p.requires_grad = False

        # LSTM for Temporal Processing
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        self.final_dropout = nn.Dropout(p=dropout)  # Dropout only in training mode

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)


    def forward(self, x):
        B, T, C, H, W = x.shape  # (Batch, Time, Channels, Height, Width)

        # Flatten batch & time
        x = x.view(B * T, C, H, W)

        # Feature Extraction via ResNet
        x = self.resnet(x)  # Output: (B*T, 512, 1, 1)
        x = x.view(B, T, 512)  # Reshape for LSTM

        # LSTM Processing
        lstm_out, _ = self.lstm(x)

        # Apply Dropout Only in Training Mode
        if self.training:
            lstm_out = self.final_dropout(lstm_out)

        # Use Mean Pooling Instead of Last Frame
        lstm_out = torch.mean(lstm_out, dim=1)
        
        # Classification (Raw logits, no Softmax)
        return self.fc(lstm_out)
