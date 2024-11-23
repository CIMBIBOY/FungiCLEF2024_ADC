import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader

class MHCNN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_layers=3,
                 kernel_size=3,
                 activation_fn=nn.ReLU,
                 dropout_rate=0.5,
                 num_semcls=10,
                 device='cuda'):
        """
        Multi-Headed CNN with flexible architecture for hyperparameter optimization.
        Args:
            in_channels (int): Number of input channels.
            base_channels (int): Number of channels in the first convolutional layer. Increases per layer.
            num_layers (int): Number of convolutional layers in the encoder.
            kernel_size (int): Kernel size for convolutions.
            activation_fn (nn.Module): Activation function to use.
            dropout_rate (float): Dropout rate for regularization.
            num_semcls (int): Number of semantic classes for classification.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        super(MHCNN, self).__init__()

        self.device = device
        self.num_semcls = num_semcls

        # Define the encoder
        layers = []
        current_channels = in_channels
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(current_channels, base_channels * (2 ** i), kernel_size=kernel_size, padding='same')
            )
            layers.append(activation_fn())
            layers.append(nn.BatchNorm2d(base_channels * (2 ** i)))
            layers.append(nn.MaxPool2d(kernel_size=2))
            current_channels = base_channels * (2 ** i)
        layers.append(nn.Dropout(dropout_rate))

        self.encoder = nn.Sequential(*layers)

        # Global pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Define MLP heads
        self.mlp_heads = nn.ModuleDict({
            "sem_cls_head": self._create_mlp(current_channels, 128, num_semcls, dropout_rate, activation_fn),
            "poisonous_head": self._create_mlp(current_channels, 128, 1, dropout_rate, activation_fn)
        })

        # Initialize weights
        self._initialize_weights()

    def _create_mlp(self, input_dim, hidden_dim, output_dim, dropout_rate, activation_fn):
        """Helper to create an MLP with dropout and activation."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = self.flatten(x)

        # Forward pass through each MLP head
        outputs = {head_name: head(x) for head_name, head in self.mlp_heads.items()}
        return outputs

    def train_model(self, train_loader, valid_loader, early_stopper,
                    num_epochs=100, learning_rate=1e-4, weight_decay=1e-5):
        """
        Train the model with the given data loaders and early stopping.
        """
        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), {k: v.to(self.device) for k, v in targets.items()}

                optimizer.zero_grad()
                outputs = self(inputs)

                # Compute the multi-task loss
                loss = F.cross_entropy(outputs["sem_cls_head"], targets["sem_cls"]) + \
                       F.binary_cross_entropy_with_logits(outputs["poisonous_head"], targets["poisonous"].unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            valid_loss = self.evaluate(valid_loader)

            print(f"Epoch {epoch + 1}: Train Loss {train_loss:.5f}, Valid Loss {valid_loss:.5f}")

            if early_stopper.early_stop(valid_loss):
                print("Early stopping triggered.")
                break

    def evaluate(self, valid_loader):
        """
        Evaluate the model on the validation set.
        """
        self.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), {k: v.to(self.device) for k, v in targets.items()}
                outputs = self(inputs)

                # Compute the validation loss
                loss = F.cross_entropy(outputs["sem_cls_head"], targets["sem_cls"]) + \
                       F.binary_cross_entropy_with_logits(outputs["poisonous_head"], targets["poisonous"].unsqueeze(1))
                valid_loss += loss.item()
        return valid_loss

    def predict(self, test_loader):
        """
        Predict on the test set.
        """
        self.eval()
        predictions = {"sem_cls": [], "poisonous": []}
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)

                # Collect predictions for each head
                predictions["sem_cls"].append(torch.argmax(outputs["sem_cls_head"], dim=1).cpu())
                predictions["poisonous"].append(torch.sigmoid(outputs["poisonous_head"]).cpu())
        return predictions

    def _initialize_weights(self):
        """
        Initialize weights using Xavier uniform initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def build_model(in_channels=3,
                base_channels=64,
                num_layers=3,
                kernel_size=3,
                activation_fn=nn.ReLU,
                dropout_rate=0.5,
                num_semcls=10,
                device='cuda'):
    """
    Build and return an instance of the MHCNN model with the specified hyperparameters.

    Args:
        in_channels (int): Number of input channels.
        base_channels (int): Number of channels in the first convolutional layer. Increases per layer.
        num_layers (int): Number of convolutional layers in the encoder.
        kernel_size (int): Kernel size for convolutions.
        activation_fn (nn.Module): Activation function to use.
        dropout_rate (float): Dropout rate for regularization.
        num_semcls (int): Number of semantic classes for classification.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        MHCNN: An instance of the MHCNN model.
    """
    model = MHCNN(
        in_channels=in_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate,
        num_semcls=num_semcls,
        device=device
    )
    return model



def objective(trial, train_loader, valid_loader, num_semcls, device='cuda'):
    """
    Objective function for optimizing the hyperparameters of the MHCNN model.

    Args:
        trial (optuna.Trial): The Optuna trial object for sampling hyperparameters.
        train_loader (DataLoader): Training dataset loader.
        valid_loader (DataLoader): Validation dataset loader.
        num_semcls (int): Number of semantic classes.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        float: Validation loss for the trial.
    """
    # Sample hyperparameters
    base_channels = trial.suggest_int("base_channels", 32, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 2, 5)
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    activation_fn = trial.suggest_categorical("activation_fn", [nn.ReLU, nn.LeakyReLU])

    # Build model with sampled hyperparameters
    model = build_model(
        in_channels=3,
        base_channels=base_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate,
        num_semcls=num_semcls,
        device=device
    )

    # Define optimizer and early stopper
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

    # Training loop
    model.to(device)
    for epoch in range(20):  # Limit to 20 epochs for faster optimization
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = F.cross_entropy(outputs["sem_cls_head"], targets["sem_cls"]) + \
                   F.binary_cross_entropy_with_logits(outputs["poisonous_head"], targets["poisonous"].unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate on validation data
        valid_loss = model.evaluate(valid_loader)
        trial.report(valid_loss, epoch)

        # Handle early stopping
        if trial.should_prune() or early_stopper.early_stop(valid_loss):
            raise optuna.exceptions.TrialPruned()

    return valid_loss


def optimize_hyperparameters(train_loader, valid_loader, num_semcls, n_trials=50, device='cuda'):
    """
    Optimize hyperparameters for the MHCNN model using Optuna.

    Args:
        train_loader (DataLoader): Training dataset loader.
        valid_loader (DataLoader): Validation dataset loader.
        num_semcls (int): Number of semantic classes.
        n_trials (int): Number of trials to run.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        optuna.Study: The optimization study object.
    """
    study = optuna.create_study(direction="minimize", study_name="MHCNN Hyperparameter Optimization")
    study.optimize(
        lambda trial: objective(trial, train_loader, valid_loader, num_semcls, device=device),
        n_trials=n_trials
    )
    return study


