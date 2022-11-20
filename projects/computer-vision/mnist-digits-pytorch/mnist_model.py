import torch
from torch import nn
import torch.nn.functional as F


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, inputs):
        inputs = inputs.reshape(-1, 784)
        outputs = self.linear(inputs)

        return outputs

    def training_step(self, batch):
        X, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_classes = torch.max(probabilities, 1)
        accuracy = torch.sum(predicted_classes == y) / \
            predicted_classes.shape[0]
        return accuracy, loss

    def validation_step(self, validation_loader):
        accuracies = []
        losses = []
        for X, y in validation_loader:
            predictions = self(X)
            probabilities = F.softmax(predictions, dim=1)
            _, predicted_classes = torch.max(probabilities, 1)
            accuracy = torch.sum(predicted_classes == y) / \
                predicted_classes.shape[0]

            loss = F.cross_entropy(predictions, y)

            accuracies.append(accuracy)
            losses.append(loss)
        print("Validation Accuracy: ", torch.sum(torch.tensor(accuracies)).item() /
              len(accuracies), " | Loss: ", torch.sum(torch.tensor(losses)).item() / len(losses))

    def on_epoch_end(self, epoch, accuracy, loss):
        print(f"Epoch: {epoch} | Accuracy: {accuracy.item()} | Loss: {loss.item()}")
