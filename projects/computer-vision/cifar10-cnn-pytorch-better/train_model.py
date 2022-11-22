from dataloader import Loader
from cifar_model import CifarModel

from torch.optim import SGD
import pprint
import torch


def train():
    model = CifarModel(3)
    train_loader, validation_loader = Loader()

    def fit(model, epochs, train_loader, validation_loader, lr):
        optimizer = SGD(model.parameters(), lr=lr)
        history = []

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            epoch_summary = model.validation_step(validation_loader)
            print(
                f"\nEpoch: {epoch} | Validation Accuracy: {epoch_summary['val_accuracy']} | Validation Loss: {epoch_summary['val_loss']}")
            pprint.pprint(epoch_summary['class_wise'])
            history.append(epoch_summary)

        return history

    def save_model(model):
        print("Saving Model...")
        torch.save(model.state_dict(), "./models/cifar10-model.pth")

    _ = fit(model, 10, train_loader, validation_loader, 0.1)
    save_model(model)
    _ = fit(model, 10, train_loader, validation_loader, 0.03)
    save_model(model)
    _ = fit(model, 10, train_loader, validation_loader, 0.01)
    save_model(model)
    _ = fit(model, 15, train_loader, validation_loader, 0.003)
    save_model(model)
    _ = fit(model, 15, train_loader, validation_loader, 0.001)
    save_model(model)


train()
