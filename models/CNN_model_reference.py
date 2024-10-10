import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

for i in range(torch.cuda.device_count()):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9,
                 hiddens=None, dropouts=None, kernel_sizes=None, maxpools=None, paddings=None, strides=None,
                 criterion=nn.CrossEntropyLoss(), activation=nn.ReLU(), output_activation=nn.Softmax(dim=1),
                 pre_process=None, post_process=None, dimensions=2, device='cuda'):
        super(UNet, self).__init__()

        if dimensions == 2:
            conv = nn.Conv2d
            batchnorm = nn.BatchNorm2d
            dropout = nn.Dropout2d
            maxpool = nn.MaxPool2d
        elif dimensions == 3:
            conv = nn.Conv3d
            batchnorm = nn.BatchNorm3d
            dropout = nn.Dropout3d
            maxpool = nn.MaxPool3d
        else:
            raise ValueError('Only 2D and 3D convolutions are supported')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.criterion = criterion
        self.pre_process = pre_process if pre_process is not None else lambda x: x
        self.post_process = post_process if post_process is not None else lambda x: x
        self.device = device

        # channels
        if hiddens is None:
            hiddens = [64]
        elif isinstance(hiddens, int):
            hiddens = [hiddens]
        elif isinstance(hiddens, list):
            assert len(hiddens) > 0, "Number of hiddens must be greater than 0"
        else:
            raise ValueError("Hiddens must be an integer or a list of integers")
        channels_downsample = [in_channels] + hiddens
        self.n_downsamples = len(channels_downsample) - 1
        channels_upsample = hiddens[::-1] + [out_channels]
        self.n_upsamples = len(channels_upsample) - 2

        # dropouts
        if dropouts is None:
            dropouts = [0] * self.n_downsamples
        elif isinstance(dropouts, float):
            dropouts = [dropouts] * self.n_downsamples
        elif isinstance(dropouts, list):
            assert len(dropouts) == self.n_downsamples, \
                "Number of dropouts must match the number of downsamples"
        else:
            raise ValueError("Dropouts must be a float or a list of floats")

        # maxpools
        if maxpools is None:
            maxpools = [2] * self.n_downsamples
        elif isinstance(maxpools, int):
            maxpools = [maxpools] * self.n_downsamples
        elif isinstance(maxpools, tuple):
            maxpools = [maxpools] * self.n_downsamples
        elif isinstance(maxpools, list):
            assert len(maxpools) == self.n_downsamples, \
                "Number of maxpools must match the number of downsamples"
        else:
            raise ValueError("Maxpools must be an integer, a tuple of integers or a list of integers")

        n_enc_convolutions = 2 * self.n_downsamples
        n_dec_convolutions = 3 * self.n_upsamples + 2
        # kernel_sizes
        if kernel_sizes is None:
            kernel_sizes = [[3] * n_enc_convolutions, [3] * n_dec_convolutions]
        elif isinstance(kernel_sizes, int):
            kernel_sizes = [[kernel_sizes] * n_enc_convolutions, [kernel_sizes] * n_dec_convolutions]
        elif isinstance(kernel_sizes, tuple):
            kernel_sizes = [[kernel_sizes] * n_enc_convolutions, [kernel_sizes] * n_dec_convolutions]
        elif isinstance(kernel_sizes, list):
            assert len(kernel_sizes) == 2, \
                "Number of kernel_sizes must be organized into encoder, decoder kernel sizes"
            assert len(kernel_sizes[0]) == n_enc_convolutions, \
                "Number of encoder kernel_sizes must match the number of encoder convolutions"
            assert len(kernel_sizes[1]) == n_dec_convolutions, \
                "Number of decoder kernel_sizes must match the number of decoder convolutions"
        else:
            raise ValueError("Kernel_sizes must be an integer, a tuple of integers or a list of integers")

        enc_kernel_sizes, dec_kernel_sizes = kernel_sizes

        if paddings is None:
            paddings = [['same'] * n_enc_convolutions, ['same'] * n_dec_convolutions]
        elif isinstance(paddings, str):
            paddings = [[paddings] * n_enc_convolutions, [paddings] * n_dec_convolutions]
        elif isinstance(paddings, tuple):
            paddings = [[paddings] * n_enc_convolutions, [paddings] * n_dec_convolutions]
        elif isinstance(paddings, int):
            paddings = [[paddings] * n_enc_convolutions, [paddings] * n_dec_convolutions]
        elif isinstance(paddings, list):
            assert len(paddings) == 2, \
                "Number of paddings must be organized into encoder, decoder paddings"
            assert len(paddings[0]) == n_enc_convolutions, \
                "Number of encoder paddings must match the number of encoder convolutions"
            assert len(paddings[1]) == n_dec_convolutions, \
                "Number of decoder paddings must match the number of decoder convolutions"
        else:
            raise ValueError("Paddings must be an integer, a tuple of integers or a list of integers")
        enc_paddings, dec_paddings = paddings

        if strides is None:
            strides = [[1] * n_enc_convolutions, [1] * n_dec_convolutions]
        elif isinstance(strides, int):
            strides = [[strides] * n_enc_convolutions, [strides] * n_dec_convolutions]
        elif isinstance(strides, tuple):
            strides = [[strides] * n_enc_convolutions, [strides] * n_dec_convolutions]
        elif isinstance(strides, list):
            assert len(strides) == 2, \
                "Number of strides must be organized into encoder, decoder strides"
            assert len(strides[0]) == n_enc_convolutions, \
                "Number of encoder strides must match the number of encoder convolutions"
            assert len(strides[1]) == n_dec_convolutions, \
                "Number of decoder strides must match the number of decoder convolutions"
        else:
            raise ValueError("Strides must be an integer, a tuple of integers or a list of integers")
        enc_strides, dec_strides = strides

        self.layers_upsample = nn.ModuleDict()
        self.layers_downsample = nn.ModuleDict()

        # Left side of the U-Net
        for i in range(self.n_downsamples):
            self.layers_downsample[f"conv_enc_{i}_1"] = \
                conv(channels_downsample[i], channels_downsample[i + 1],
                     kernel_size=enc_kernel_sizes[2 * i], padding=enc_paddings[2 * i], stride=enc_strides[2 * i])

            self.layers_downsample[f"conv_enc_{i}_2"] = \
                conv(channels_downsample[i + 1], channels_downsample[i + 1],
                     kernel_size=enc_kernel_sizes[2 * i + 1], padding=enc_paddings[2 * i + 1],
                     stride=enc_strides[2 * i + 1])

            self.layers_downsample[f"batchnorm_enc_{i}"] = \
                batchnorm(channels_downsample[i + 1])

            self.layers_downsample[f"dropout_{i}"] = \
                dropout(p=dropouts[i])

            if i != self.n_downsamples - 1:  # utsonak nem kell
                self.layers_downsample[f"maxpool_{i}"] = maxpool(maxpools[i])

        # Up-sampling starts, right side of the U-Net
        for i in range(self.n_upsamples):
            self.layers_upsample[f"upconv_dec_{i}"] = \
                conv(channels_upsample[i], channels_upsample[i + 1],
                     kernel_size=dec_kernel_sizes[3 * i], padding=dec_paddings[3 * i], stride=dec_strides[3 * i])

            self.layers_upsample[f"conv_dec_{i}_1"] = \
                conv(channels_upsample[i], channels_upsample[i + 1],
                     kernel_size=dec_kernel_sizes[3 * i + 1], padding=dec_paddings[3 * i + 1],
                     stride=dec_strides[3 * i + 1])

            self.layers_upsample[f"conv_dec_{i}_2"] = \
                conv(channels_upsample[i + 1], channels_upsample[i + 1],
                     kernel_size=dec_kernel_sizes[3 * i + 2], padding=dec_paddings[3 * i + 2],
                     stride=dec_strides[3 * i + 2])

            if i == self.n_upsamples - 1:
                self.layers_upsample[f"conv_dec_{i}_3"] = \
                    conv(channels_upsample[i + 1], channels_upsample[i + 1],
                         kernel_size=dec_kernel_sizes[3 * i + 3], padding=dec_paddings[3 * i + 3],
                         stride=dec_strides[3 * i + 3])
            self.layers_upsample[f"batchnorm_dec_{i}"] = batchnorm(channels_upsample[i + 1])
            #else:
            #    self.layers_upsample[f"batchnorm_dec_{i}"] = batchnorm(channels_upsample[i + 1])

        # Output layer of the U-Net with a softmax activation
        self.layers_upsample["conv_out"] = \
            conv(channels_upsample[-2], channels_upsample[-1],
                 kernel_size=dec_kernel_sizes[-1], padding=dec_paddings[-1], stride=dec_strides[-1])
        self.output_activation = output_activation
        self.activation = activation

        # initialize all parameters with xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        skip_connections = []
        # Left side of the U-Net
        for i in range(self.n_downsamples):
            x = self.activation(self.layers_downsample[f"conv_enc_{i}_1"](x))
            x = self.activation(self.layers_downsample[f"conv_enc_{i}_2"](x))
            x = self.layers_downsample[f"batchnorm_enc_{i}"](x)
            x = self.layers_downsample[f"dropout_{i}"](x)

            if i != self.n_downsamples - 1:
                skip_connections.append(x)
                x = self.layers_downsample[f"maxpool_{i}"](x)

        # Upsampling Starts, right side of the U-Net
        for i in range(self.n_upsamples):
            skip = skip_connections.pop()
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = self.activation(self.layers_upsample[f"upconv_dec_{i}"](x))
            x = torch.cat([skip, x], dim=1)
            x = self.activation(self.layers_upsample[f"conv_dec_{i}_1"](x))
            x = self.activation(self.layers_upsample[f"conv_dec_{i}_2"](x))

            if i == self.n_upsamples - 1:
                x = self.activation(self.layers_upsample[f"conv_dec_{i}_3"](x))
            x = self.layers_upsample[f"batchnorm_dec_{i}"](x)

        # Output layer of the U-Net with a softmax activation
        out_conv = self.layers_upsample["conv_out"](x)
        y = self.output_activation(out_conv)
        return y

    def train_model(self, train_loader, valid_loader, early_stopper,
                    num_epochs=100, learning_rate=1e-4, weight_decay=1e-5, device='cuda'):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_loss = valid_loss = []
        for epoch in range(num_epochs):
            epoch_train_loss = correct_train = total_train = 0
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

                c, t = self._accuracy_score(targets, outputs)
                correct_train += c
                total_train += t

                outputs.detach().cpu()
                targets.detach().cpu()

            train_accuracy = 100 * correct_train / total_train

            epoch_valid_loss, valid_accuracy = self.evaluate(valid_loader, device=device)

            epoch_train_loss /= len(train_loader)
            epoch_valid_loss /= len(valid_loader)
            metrics = {
                "epoch_train_loss": epoch_train_loss,
                "epoch_valid_loss": epoch_valid_loss
            }
            train.report(metrics)
            train_loss.append(epoch_train_loss)
            valid_loss.append(epoch_valid_loss)
            print(
                f'Epoch {epoch + 1:03}: | Train Loss: {epoch_train_loss:.5f} | Validation Loss: {epoch_valid_loss:.5f}'
                + f' | Train Acc: {train_accuracy:.2f}% | Valid Acc: {valid_accuracy:.2f}%')

            if early_stopper.early_stop(epoch_valid_loss):
                break
        return train_loss, valid_loss

    def predict(self, test_loader, device='cuda'):
        self.eval()
        raw_predictions = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, _ = batch
                inputs = self.pre_process(inputs).to(device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                raw_predictions.append(preds.detach().cpu().numpy())
        results = []
        for batch in raw_predictions:
            for img in batch:
                results.append(self.post_process(img))
        return results, raw_predictions

    def evaluate(self, test_loader, device='cuda'):
        epoch_valid_loss = 0
        self.eval()
        correct_valid = 0
        total_valid = 0
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            epoch_valid_loss += loss.item()

            c, t = self._accuracy_score(targets, outputs)
            correct_valid += c
            total_valid += t

            outputs.detach().cpu()
            targets.detach().cpu()

        valid_accuracy = 100 * correct_valid / total_valid
        return epoch_valid_loss, valid_accuracy

    def summary(self, input_shape):
        from torchinfo import summary
        return summary(self, input_size=input_shape)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        if y_true.shape[1] == 1:
            total = np.prod(y_true.size())
            correct = torch.sum(y_pred.round() == y_true).item()
            return correct, total
        else:
            _, acc_predictions = torch.max(y_pred, 1)
            _, acc_targets = torch.max(y_true, 1)
            total = np.prod(acc_targets.size())
            correct = (acc_predictions == acc_targets).sum().item()
            return correct, total




class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False