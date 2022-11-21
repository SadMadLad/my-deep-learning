import torch
from torch import nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    def training_step(self, batch):
        X, y = batch
        loss = F.cross_entropy(self(X), y)
        return loss

    def validation_step(self, val_loader):
        """Accuracy + Loss + Class-wise Correct + Wrong Predictions"""
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
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(2, 2)

        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu_3 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.dense_1 = nn.Linear(2048, 256)
        self.relu_4 = nn.ReLU()
        self.final = nn.Linear(256, 10)

    def forward(self, inputs):
        inputs = self.conv_1(inputs)
        inputs = self.relu_1(inputs)
        inputs = self.maxpool_1(inputs)

        inputs = self.conv_2(inputs)
        inputs = self.relu_2(inputs)
        inputs = self.maxpool_2(inputs)

        inputs = self.conv_3(inputs)
        inputs = self.relu_3(inputs)
        inputs = self.maxpool_3(inputs)

        inputs = self.flatten(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.relu_4(inputs)

        outputs = self.final(inputs)

        return outputs

# model = CifarModel()
# print(model)