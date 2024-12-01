import torch
from torch.utils.data import DataLoader
from models.HibridMHCNN import build_model , train_model
from datasets.fungidataset import build_dataset
from torch import nn
IMAGEDIR = "/Users/koksziszdave/Downloads/fungi_images"
LABELDIR = "/Users/koksziszdave/Downloads/fungi_train_metadata.csv"

args = {
    "image_dir": IMAGEDIR,
    "labels_path": LABELDIR,
    "pre_load": False,
    "batch_size": 32
}

train_loader, valid_loader = build_dataset(args)

num_semcls = 100
"""
study = optimize_hyperparameters(train_loader, valid_loader, num_semcls, n_trials=50, device='cpu')
print("Best hyperparameters:", study.best_params)

best_model = build_model(
    in_channels=3,
    base_channels=study.best_params["base_channels"],
    num_layers=study.best_params["num_layers"],
    kernel_size=study.best_params["kernel_size"],
    activation_fn=study.best_params["activation_fn"],
    dropout_rate=study.best_params["dropout_rate"],
    num_semcls=num_semcls,
    device='cpu'
)
"""
device = 'cpu'
model = build_model(in_channels=3, base_channels=16, num_layers=4, kernel_size=3, dropout_rate=0.5, num_semcls=num_semcls, device=device)


# Train the model
trained_model=train_model(model, train_loader, valid_loader, num_epochs=30, device=device)

# Save the trained model
#torch.save(trained_model.state_dict(), "hibrid_model.pth")

# Load the trained model
#model = build_model(in_channels=3, base_channels=16, num_layers=6, kernel_size=3, dropout_rate=0.5, num_semcls=num_semcls, device=device)
#model.load_state_dict(torch.load("hibrid_model.pth"))

# Evaluate the model

def evaluate_model(model, data_loader, device):
    """
    Evaluate the MHCNN model with hybrid heads.
    Args:
        model (nn.Module): MHCNN model with hybrid heads.
        data_loader (DataLoader): DataLoader for the dataset.
        device (str): Device to run the model on ('cuda' or 'cpu').
    Returns:
        float: Mean loss on the dataset.
        float: Mean accuracy on the dataset.
    """
    model.eval()
    semantic_cls_criterion = nn.CrossEntropyLoss()
    poisonous_criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_sem_cls_correct = 0
    total_poisonous_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch["image"].to(device)
            y_sem_cls = batch["target_sem_cls"].to(device)
            y_poisonous = batch["target_poisonous"].to(device)
            output = model(x)
            sem_cls_loss = semantic_cls_criterion(output["sem_cls"], y_sem_cls)
            poisonous_loss = poisonous_criterion(output["poisonous"].squeeze(), y_poisonous.float())
            loss = sem_cls_loss + poisonous_loss
            total_loss += loss.item()

            sem_cls_preds = output["sem_cls"].argmax(dim=1)
            poisonous_preds = (output["poisonous"] > 0.0).float()
            total_sem_cls_correct += (sem_cls_preds == y_sem_cls).sum().item()
            total_poisonous_correct += (poisonous_preds == y_poisonous).sum().item()
            total_samples += x.size(0)

    mean_loss = total_loss / len(data_loader)
    mean_sem_cls_accuracy = total_sem_cls_correct / total_samples
    mean_poisonous_accuracy = total_poisonous_correct / total_samples

    return mean_loss, mean_sem_cls_accuracy, mean_poisonous_accuracy

train_loss, train_sem_cls_accuracy, train_poisonous_accuracy = evaluate_model(model, train_loader, device)