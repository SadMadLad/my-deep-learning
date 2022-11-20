from mnist_model import MnistModel
from dataloaders import Loaders

import torch


def main():
    train_loader, validation_loader, test_loader = Loaders()
    model = MnistModel()

    # Hyperparameters

    epochs = 100
    lr = 1e-4

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def fit(model, optimizer, train_loader, validation_loader, epochs=100):
        history = []

        for epoch in range(epochs):
            for batch in train_loader:
                accuracy, loss = model.training_step(batch)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                history.append(loss.item())

            model.validation_step(validation_loader)
            model.on_epoch_end(epoch, accuracy, loss)

        return history

    fit(model, optimizer, train_loader, validation_loader, epochs)


main()
