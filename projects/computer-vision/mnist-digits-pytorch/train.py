from mnist_model import MnistModel
from dataloaders import Loaders

import torch


def Train():
    train_loader, validation_loader, test_loader = Loaders()
    model = MnistModel()

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

    def save_model(model):
        print("Saving Model...")
        torch.save(model.state_dict(), "./models/mnist-model.pth")

    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2)
    fit(model, optimizer, train_loader, validation_loader, epochs=5)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    fit(model, optimizer, train_loader, validation_loader, epochs=5)

    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
    fit(model, optimizer, train_loader, validation_loader, epochs=5)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    fit(model, optimizer, train_loader, validation_loader, epochs=5)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    fit(model, optimizer, train_loader, validation_loader, epochs=5)

    save_model(model=model)


Train()
