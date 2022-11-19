import torch
import numpy as np


def vanilla_gradient_descent():
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

    W = torch.randn(size=(2, 3), requires_grad=True)
    b = torch.randn(size=(2, ), requires_grad=True)

    def mse(predictions, truths):
        diff = truths - predictions
        return torch.sum(diff*diff) / diff.numel()

    def model(inputs):
        return inputs @ W.t() + b

    epochs = 50000
    lr = 1e-5
    prev_loss = float('inf')
    for epoch in range(epochs):
        predictions = model(inputs)
        loss = mse(predictions, targets)
        loss.backward()

        with torch.no_grad():
            W.sub_(W*lr)
            b.sub_(b*lr)

            W.grad.zero_()
            b.grad.zero_()
        if prev_loss < loss:
            print("Stop! Loss increasing now")
            break
        prev_loss = loss


        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.detach().numpy()}")
        

vanilla_gradient_descent()
