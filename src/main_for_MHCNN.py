from torch.utils.data import DataLoader
from models.MHCNN import build_model, optimize_hyperparameters

train_loader = DataLoader(...)
valid_loader = DataLoader(...)
num_semcls = 100

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
    device='cuda'
)