import torch
from torch import nn
import torch.nn.functional as F


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)

    def forward(self, inputs):
        inputs = inputs.reshape(-1, 784)
        inputs = self.hidden(inputs)
        relu = self.relu(inputs)
        outputs = self.output(relu)

        return outputs

    def training_step(self, batch):
        X, y = batch
        predictions = self(X)
        loss = F.cross_entropy(predictions, y)

        return loss

    def accuracy(self, batch):
        X, y = batch
        predictions = self(X)
        proba = F.softmax(predictions, dim=1)
        _, predicted_classes = torch.max(proba, 1)
        acc = torch.sum(predicted_classes == y) / predicted_classes.shape[0]

        return acc

    def validation_step(self, valid_loader):
        accuracies = []
        losses = []

        for batch in valid_loader:
            loss = self.training_step(batch)
            acc = self.accuracy(batch)

            losses.append(loss.item())
            accuracies.append(acc.item())

        avg_accuracy = torch.tensor(accuracies, dtype=torch.float16)
        avg_loss = torch.tensor(losses, dtype=torch.float16)

        avg_accuracy = avg_accuracy.sum() / avg_accuracy.shape[0]
        avg_loss = avg_loss.sum() / avg_loss.shape[0]

        return {
            "val_loss": avg_loss.item(),
            "val_accuracy": avg_accuracy.item(),
        }
