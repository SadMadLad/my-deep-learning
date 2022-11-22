import torch
from torch import nn
from torch.optim import Adam
import pprint

from devicer import get_default_device, to_device, DeviceDataLoader
from cifar_model import CifarModel
from dataloader import Loader


def deep_train():
    device = get_default_device()
    print(device)
    model = to_device(CifarModel(3, 10), device)

    train_loader, validation_loader = Loader()

    train_loader = DeviceDataLoader(train_loader, device)
    validation_loader = DeviceDataLoader(validation_loader, device)

    def fit(model, epochs, max_lr, train_loader, validation_loader, weight_decay=0, grad_clip=None):
        torch.cuda.empty_cache()

        optimizer = Adam(model.parameters(), lr=max_lr,
                         weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

        history = []

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_summary = model.validation_step(validation_loader)
            print(
                f"\nEpoch: {epoch} | Validation Accuracy: {epoch_summary['val_accuracy']} | Validation Loss: {epoch_summary['val_loss']}")
            pprint.pprint(epoch_summary['class_wise'])
            history.append(epoch_summary)

        return history

    def save_model(model):
        print("Saving Model...")
        torch.save(model.state_dict(), "./models/cifar10-model.pth")

    _ = fit(model, 50, 0.01, train_loader, validation_loader,
            weight_decay=1e-4, grad_clip=0.1)
