from engine.tensor import Tensor


def mse_loss(predictions, targets):
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)

    diff = predictions - targets
    squared = diff ** 2
    n = predictions.data.size

    return squared.sum() * (1.0 / n)