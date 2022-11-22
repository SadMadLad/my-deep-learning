import torch
from torch import nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    def training_step(self, batch):
        X, y = batch
        loss = F.cross_entropy(self(X), y)
        return loss

    @torch.no_grad()
    def validation_step(self, val_loader):
        """Accuracy + Loss + Class-wise Correct + Wrong Predictions"""
        self.eval()
        
        accuracies = []
        losses = []
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        predictions_dict = {0: {'correct': 0, 'wrong': 0}, 1: {'correct': 0, 'wrong': 0}, 2: {'correct': 0, 'wrong': 0}, 3: {'correct': 0, 'wrong': 0}, 4: {'correct': 0, 'wrong': 0},
                            5: {'correct': 0, 'wrong': 0}, 6: {'correct': 0, 'wrong': 0}, 7: {'correct': 0, 'wrong': 0}, 8: {'correct': 0, 'wrong': 0}, 9: {'correct': 0, 'wrong': 0}}

        for batch in val_loader:
            loss = self.training_step(batch)
            accuracy, predicted_classes = self.accuracy(batch)

            losses.append(loss.item())
            accuracies.append(accuracy.item())

            for i in range(len(class_names)):
                indices = list((batch[1] == i).nonzero(
                    as_tuple=True)[0].detach().numpy())
                correct_predicted = (
                    predicted_classes[indices] == batch[1][indices]).sum()
                wrong_predicted = (
                    predicted_classes[indices] != batch[1][indices]).sum()

                predictions_dict[i]['correct'] += correct_predicted.item()
                predictions_dict[i]['wrong'] += wrong_predicted.item()

        avg_accuracy = torch.tensor(accuracies, dtype=torch.float16)
        avg_loss = torch.tensor(losses, dtype=torch.float16)

        avg_accuracy = avg_accuracy.sum() / avg_accuracy.shape[0]
        avg_loss = avg_loss.sum() / avg_loss.shape[0]

        for i, label in enumerate(class_names):
            predictions_dict[label] = predictions_dict[i]
            del predictions_dict[i]

        return {
            "val_accuracy": avg_accuracy.item(),
            "val_loss": avg_loss.item(),
            "class_wise": predictions_dict,
        }

    def accuracy(self, batch):
        X, y = batch
        predictions = self(X)
        proba = F.softmax(predictions, dim=1)
        _, predicted_classes = torch.max(proba, 1)
        acc = torch.sum(predicted_classes == y) / predicted_classes.shape[0]

        return acc, predicted_classes


class CifarModel(BaseModule):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        # Starting: 3 x 32 x 32
        self.conv_1 = self.conv_block(in_channels, 64)  # 64 x 32 x 32
        self.conv_2 = self.conv_block(64, 128, True)  # 128 x 16 x 16
        self.res_1 = nn.Sequential(self.conv_block(
            128, 128), self.conv_block(128, 128))  # 128 x 16 x 16

        self.conv_3 = self.conv_block(128, 256, True)  # 256 x 8 x 8
        self.conv_4 = self.conv_block(256, 512, True)  # 512 x 4 x 4
        self.res_2 = nn.Sequential(self.conv_block(
            512, 512), self.conv_block(512, 512))  # 512 x 4 x 4

        self.dense = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def conv_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, stride=1),
            # nn.LayerNorm(normalized_shape=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = self.conv_1(inputs)
        inputs = self.conv_2(inputs)
        inputs = self.res_1(inputs) + inputs

        inputs = self.conv_3(inputs)
        inputs = self.conv_4(inputs)
        inputs = self.res_2(inputs) + inputs

        outputs = self.dense(inputs)
        return outputs
