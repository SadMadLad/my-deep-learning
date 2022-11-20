import torch
from torch.optim import SGD

from mnist_model import MnistModel
from dataloaders import Loader, GetSingleBatch


def train():
    model = MnistModel()
    train_loader, valid_loader = Loader()

    def fit(model, epochs, train_loader, valid_loader, lr):
        optimizer = SGD(model.parameters(), lr=lr)
        history = []

        for epoch in range(epochs):
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            step_summary = model.validation_step(valid_loader)
            print(f"Epoch: {epoch} | {step_summary}")
            history.append(step_summary)
        
        return history

    def save_model(model):
        print("Saving Model...")
        torch.save(model.state_dict(), "./models/mnist-model.pth")
    
    _ = fit(model, 2, train_loader, valid_loader, 0.5)
    _ = fit(model, 3, train_loader, valid_loader, 0.3)
    _ = fit(model, 4, train_loader, valid_loader, 0.1)
    # _ = fit(model, 2, train_loader, valid_loader, 1e-5)
    # _ = fit(model, 2, train_loader, valid_loader, 1e-6)

    save_model(model)

train()
