# Neural Network implementing Universal Approximation Theorem from Scratch Using Only NumPy
```
More advanced 5 expereiments on this Neural Network will be added soon
```
This project builds a feedforward neural network entirely from scratch using only NumPy — no PyTorch, no TensorFlow — to experimentally verify the Universal Approximation Theorem (UAT). The UAT states that a neural network with sufficient width can approximate any continuous function to arbitrary precision. The goal was to learn sin(x) with no trigonometry built in, just data and gradient descent. The project includes a full reverse-mode automatic differentiation engine, Xavier-initialized dense layers, ReLU/Sigmoid/Tanh activations, MSE loss, SGD and Adam optimizers, and four experiments of increasing difficulty. Gradient correctness is verified numerically (relative error < 1e-10 for weights) and confirmed against PyTorch (agreement to 3e-8). The project took approximately 2 months to fully understand and implement.

## Repository Structure

```
NN_for_UAT/
    engine/
        tensor.py           # Tensor class, autograd graph, backward pass
        ops.py              # NumPy forward/backward math primitives
    layers/
        base.py             # Abstract Layer base class
        dense.py            # Fully connected layer with Xavier initialization
        activations.py      # ReLU, Sigmoid, Tanh
        loss.py             # MSE loss
    network/
        model.py            # Sequential container
        optimizer.py        # SGD and Adam
    experiments/
        01_simple_sin.py    # sin(x) approximation, SGD vs Adam comparison
        02_width_study.py   # Approximation error vs neuron count
        03_hard_function.py # Four target functions of increasing complexity
    verification/
        gradient_check.py   # Numerical gradient verification
        compare_pytorch.py  # PyTorch agreement check
    visualization/
        loss_landscape_3d.py  # Filter-normalized 3D loss surface
    proofs/
        main.tex            # LaTeX paper with derivations and results
```

## Requirements

```
numpy
matplotlib
torch
```

Install with:
```
pip install numpy matplotlib torch
```

## Running the Experiments

Run each file directly from the repo root:

```
python experiments/01_simple_sin.py
python experiments/02_width_study.py
python experiments/03_hard_function.py
python verification/gradient_check.py
python verification/compare_pytorch.py
python visualization/loss_landscape_3d.py
```

Results (plots as `.png`) are saved in the same folder as each script.
