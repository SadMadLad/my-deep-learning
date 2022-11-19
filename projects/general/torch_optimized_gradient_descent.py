import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F


def OptimizedGradientDescent():
    inputs = np.array([[73, 67, 43],
                       [91, 88, 64],
                       [87, 134, 58],
                       [102, 43, 37],
                       [69, 96, 70]], dtype='float32')
    targets = np.array([[56, 70],
                        [81, 101],
                        [119, 133],
                        [22, 37],
                        [103, 119]], dtype='float32')

    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    # Initializing DataLoader
    train_ds = TensorDataset(inputs, targets)
    dataloader = DataLoader(train_ds, batch_size=4, shuffle=True)

    for X, y in dataloader:
        print("A Random Batch: ")
        print("\tX: ", X)
        print("\ty: ", y)
        break

    # Model
    model = Linear(in_features=3, out_features=2)

    print("\nTest Predictions: ", model(inputs).detach().numpy())
    print("Model Weights: ", model.weight)
    print("Model Bias: ", model.bias)

    # Loss Functions
    loss_fn = F.mse_loss
    loss = loss_fn(model(inputs), targets)

    print("\nTest Loss: ", loss.detach().numpy())

    # Optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-5)

    # Fit function for training
    def fit(model, loss_function, optimizer, epochs, loader):
        for epoch in range(epochs):
            for X, y in loader:
                # 1. Making predictions
                predictions = model(X)
                # 2. Calculating loss
                loss = loss_function(predictions, y)
                # 3. Computing gradient
                loss.backward()
                # 4. Update parameters
                optimizer.step()
                # 5. Reset gradients
                optimizer.zero_grad()
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss.detach().numpy()}")

        return model

    model = fit(model, loss_fn, optim, 100, dataloader)
    print("Final Predictions: ")
    final_predictions = model(inputs)
    print(final_predictions)
    print("Targets: ")
    print(targets)

    return


OptimizedGradientDescent()
