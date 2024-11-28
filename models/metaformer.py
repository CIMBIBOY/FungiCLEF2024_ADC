import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Import MetaFormer components
from models.MetaFormer.models import MetaFG  # Adjust import path if necessary
from src.dataset import FungiDataset

def main():
    # Path to pretrained weights and data directory
    pretrained_weights = "model_data/metafg_0_21k_224.pth"
    data_dir = "./data/x_train"  # Update this to your dataset path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the MetaFormer model
    model = MetaFG(
        img_size=224,
        num_classes=100,  # Update to the number of classes in your dataset
    )
    model.to(device)

    # Load pretrained weights
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # Define data transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = FungiDataset(data_dir=os.path.join(data_dir, "train"), transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = FungiDataset(data_dir=os.path.join(data_dir, "val"), transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
        scheduler.step()

        # Validation loop
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save fine-tuned model
    torch.save(model.state_dict(), "fine_tuned_metaformer.pth")
    print("Fine-tuned model saved to fine_tuned_metaformer.pth")


if __name__ == "__main__":
    main()