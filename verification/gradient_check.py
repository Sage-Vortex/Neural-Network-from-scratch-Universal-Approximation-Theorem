import numpy as np
from engine.tensor import Tensor
from layers.dense import Dense
from layers.activations import Tanh
from layers.loss import mse_loss
from network.model import Sequential


def compute_numerical_gradient(model, X_data, y_data, param, h=1e-4):
    numerical_grad = np.zeros_like(param.data)
    it = np.nditer(param.data, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index

        original = param.data[idx]

        # f(x + h)
        param.data[idx] = original + h
        pred_plus = model.forward(Tensor(X_data))
        loss_plus = mse_loss(pred_plus, y_data).data

        # f(x - h)
        param.data[idx] = original - h
        pred_minus = model.forward(Tensor(X_data))
        loss_minus = mse_loss(pred_minus, y_data).data

        # central difference formula
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * h)

        # restore original value
        param.data[idx] = original
        it.iternext()

    return numerical_grad


def relative_error(analytical, numerical):
    return np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)


def gradient_check(model, X_data, y_data, tolerance=1e-4):
    # forward + backward to populate all .grad fields
    pred = model.forward(Tensor(X_data))
    loss = mse_loss(pred, y_data)
    loss.backward()

    all_passed = True

    for i, param in enumerate(model.parameters()):
        analytical = param.grad.copy()
        numerical  = compute_numerical_gradient(model, X_data, y_data, param)

        err = relative_error(analytical, numerical)
        max_err = np.max(err)

        status = "PASS" if max_err < tolerance else "FAIL"
        if status == "FAIL":
            all_passed = False

        label = "W" if i % 2 == 0 else "b"
        layer = i // 2 + 1
        print(f"Layer {layer} {label} | max relative error: {max_err:.2e} | {status}")

    return all_passed


def main():
    np.random.seed(0)

    X_data = np.linspace(-np.pi, np.pi, 10).reshape(-1, 1)
    y_data = np.sin(X_data)

    model = Sequential([
        Dense(1, 4),
        Tanh(),
        Dense(4, 1)
    ])

    print("Running gradient check...")
    print("-" * 50)
    passed = gradient_check(model, X_data, y_data)
    print("-" * 50)

    if passed:
        print("All gradients verified. Autograd engine is correct.")
    else:
        print("Gradient check FAILED. Review backward rules.")


if __name__ == "__main__":
    main()