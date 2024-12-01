import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torchvision import models
from torch import optim
import tqdm
import numpy as np
class HybridHead(nn.Module):
    def __init__(self, embed_dim, num_heads, ssm_dim, output_dim, dropout=0.1):
        """
        Implements a hybrid head combining attention and state-space model (SSM) mechanisms.
        Args:
            embed_dim (int): Embedding dimension for input.
            num_heads (int): Number of attention heads.
            ssm_dim (int): Dimension of SSM hidden state.
            output_dim (int): Output dimension of the head.
            dropout (float): Dropout rate.
        """
        super(HybridHead, self).__init__()

        # Attention head
        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # State-space model (SSM) head
        self.ssm = nn.Sequential(
            nn.Linear(embed_dim, ssm_dim),
            nn.GELU(),
            nn.Linear(ssm_dim, embed_dim),
        )

        # Fully connected layer for combining attention and SSM outputs
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the hybrid head.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).
        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Attention head: Compute attention over the sequence
        attn_output, _ = self.attention(x, x, x)

        # State-space model (SSM) head
        ssm_output = self.ssm(x)

        # Concatenate attention and SSM outputs
        combined = torch.cat([attn_output, ssm_output], dim=-1)

        # Fully connected layer for final prediction
        output = self.fc(combined.mean(dim=1))  # Pool across sequence
        return output


class HibridMHCNN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_layers=3,
                 kernel_size=3,
                 dropout_rate=0.1,
                 sem_cls_output_dim=10,
                 poisonous_output_dim=1,
                 embed_dim=128,
                 num_heads=4,
                 ssm_dim=64,
                 resnet=True,
                 device='cuda'):
        """
        Implements MHCNN with hybrid heads for semantic class prediction and binary classification.
        Args:
            in_channels (int): Number of input channels.
            base_channels (int): Base number of convolutional channels.
            num_layers (int): Number of convolutional layers in the encoder.
            kernel_size (int): Kernel size for convolutions.
            dropout_rate (float): Dropout rate for regularization.
            sem_cls_output_dim (int): Number of semantic classes.
            poisonous_output_dim (int): Dimension for binary classification.
            embed_dim (int): Embedding dimension for hybrid heads.
            num_heads (int): Number of attention heads.
            ssm_dim (int): Hidden state dimension for the SSM.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        super(HibridMHCNN, self).__init__()
        self.device = device
        self.resnet = resnet
        # Encoder: Convolutional feature extractor
        if self.resnet:
            self.encoder = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
            current_channels = 2048

            # Global pooling to reduce spatial dimensions
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
        else:
            layers = []
            current_channels = in_channels
            for i in range(num_layers):
                layers.append(nn.Conv2d(current_channels, base_channels * (2 ** i), kernel_size, padding='same'))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm2d(base_channels * (2 ** i)))
                layers.append(nn.MaxPool2d(kernel_size=2))
                current_channels = base_channels * (2 ** i)
            self.encoder = nn.Sequential(*layers)

            # Global pooling to reduce spatial dimensions
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

        # Hybrid Heads
        self.sem_cls_head = HybridHead(embed_dim=embed_dim, num_heads=num_heads, ssm_dim=ssm_dim,
                                       output_dim=sem_cls_output_dim, dropout=dropout_rate)
        self.poisonous_head = HybridHead(embed_dim=embed_dim, num_heads=num_heads, ssm_dim=ssm_dim,
                                         output_dim=poisonous_output_dim, dropout=dropout_rate)

        # Fully connected layer to project encoder features to embedding dimension
        self.projector = nn.Linear(current_channels, embed_dim)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of the MHCNN with hybrid heads.
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            dict: Output containing semantic class and binary predictions.
        """
        x= x.permute(0, 3, 1, 2)
        # Encoder
        x = self.encoder(x)
        x = self.global_pool(x)
        x = self.flatten(x)

        # Project to embedding dimension
        x = self.projector(x).unsqueeze(1)  # Add sequence dimension

        # Hybrid heads
        sem_cls_output = self.sem_cls_head(x)
        poisonous_output = self.poisonous_head(x)

        return {
            "sem_cls": sem_cls_output,
            "poisonous": poisonous_output
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def build_model(in_channels, base_channels, num_layers, kernel_size, dropout_rate, num_semcls, device):
    """
    Build the MHCNN model with hybrid heads.
    Args:
        in_channels (int): Number of input channels.
        base_channels (int): Base number of convolutional channels.
        num_layers (int): Number of convolutional layers in the encoder.
        kernel_size (int): Kernel size for convolutions.
        dropout_rate (float): Dropout rate for regularization.
        num_semcls (int): Number of semantic classes.
        device (str): Device to run the model on ('cuda' or 'cpu').
    Returns:
        nn.Module: MHCNN model with hybrid heads.
    """
    model = HibridMHCNN(
        in_channels=in_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        sem_cls_output_dim=num_semcls,
        poisonous_output_dim=1,
        embed_dim=128,
        num_heads=4,
        ssm_dim=64,
        device=device
    )

    model.to(device)
    return model



import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
import tqdm
import numpy as np

def train_model(model, train_loader, valid_loader, num_epochs, device, save_path="./models/checkpoints/resnetHMH.pth", patience=5):
    """
    Train the model with advanced features like LR scheduling, early stopping, and checkpointing.
    
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        device (str): Device to train on ('cuda' or 'cpu').
        save_path (str): Path to save the best model checkpoint.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    # Define loss functions
    semantic_cls_criterion = nn.CrossEntropyLoss()
    poisonous_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            x = batch["image"].to(device)
            y_sem_cls = batch["target_sem_cls"].to(device)
            y_poisonous = batch["target_poisonous"].to(device)

            # Forward pass
            output = model(x)
            sem_cls_loss = semantic_cls_criterion(output["sem_cls"], y_sem_cls)
            poisonous_loss = poisonous_criterion(output["poisonous"].squeeze(), y_poisonous.float())
            loss = sem_cls_loss + poisonous_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Adjust learning rate
        scheduler.step()

        # Log training loss
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        sem_outputs, sem_targets = [], []
        poi_outputs, poi_targets = []

        with torch.no_grad():
            for batch in tqdm.tqdm(valid_loader, desc="Validating"):
                x = batch["image"].to(device)
                y_sem_cls = batch["target_sem_cls"].to(device)
                y_poisonous = batch["target_poisonous"].to(device)

                output = model(x)
                sem_cls_loss = semantic_cls_criterion(output["sem_cls"], y_sem_cls)
                poisonous_loss = poisonous_criterion(output["poisonous"].squeeze(), y_poisonous.float())
                loss = sem_cls_loss + poisonous_loss

                val_loss += loss.item()

                # Collect predictions and targets
                sem_outputs.append(output["sem_cls"].cpu().numpy())
                sem_targets.append(y_sem_cls.cpu().numpy())
                poi_outputs.append(output["poisonous"].cpu().numpy())
                poi_targets.append(y_poisonous.cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss:.4f}")

        # Flatten and process outputs for accuracy calculations
        sem_outputs = np.concatenate(sem_outputs, axis=0)
        sem_targets = np.concatenate(sem_targets, axis=0)
        poi_outputs = np.concatenate(poi_outputs, axis=0)
        poi_targets = np.concatenate(poi_targets, axis=0)

        if len(sem_targets.shape) > 1 and sem_targets.shape[1] > 1:
            sem_targets = np.argmax(sem_targets, axis=1)

        sem_acc = accuracy_score(sem_targets, np.argmax(sem_outputs, axis=1))
        poi_acc = accuracy_score(poi_targets, np.round(poi_outputs))

        print(f"Semantic Class Accuracy: {sem_acc:.4f}")
        print(f"Poisonous Accuracy: {poi_acc:.4f}")

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete. Best model saved to:", save_path)
    return model