import torch
from torch.utils.data import DataLoader
from models.HibridMHCNN import build_model , train_model
from src.fungidataset import build_dataset

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
model = build_model(in_channels=3, base_channels=16, num_layers=6, kernel_size=3, dropout_rate=0.5, num_semcls=num_semcls, device=device)


# Train the model
trained_model=train_model(model, train_loader, valid_loader, num_epochs=1, device=device)

# Save the trained model
torch.save(trained_model.state_dict(), "hibrid_model.pth")