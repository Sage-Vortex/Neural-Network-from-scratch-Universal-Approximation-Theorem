
# Neural Network Implementing the Universal Approximation Theorem from Scratch Using Only NumPy

This project builds a feedforward neural network entirely from scratch using only **NumPy** — no PyTorch or TensorFlow for the implementation — to experimentally study the **Universal Approximation Theorem (UAT)**.

The UAT states that a neural network with sufficient width can approximate any continuous function to arbitrary precision. The original goal of the project was to learn `sin(x)` without using any built-in trigonometric operation in the network itself, relying only on data and gradient descent.

The project includes:

- A complete reverse-mode automatic differentiation engine
- Xavier-initialized dense layers
- ReLU, Sigmoid, and Tanh activations
- Mean Squared Error (MSE) loss
- SGD and Adam optimizers
- Numerical gradient verification
- Verification against PyTorch
- Experiments studying width, depth, optimizer choice, function complexity, training budget, and random-seed robustness

Gradient correctness is verified numerically with relative error below `1e-10` for weights and agrees with PyTorch to approximately `3 × 10^-8`.

The project took approximately two months to fully understand and implement.

---

## Repository Structure

```text
NN_for_UAT/
│
├── engine/
│   ├── tensor.py
│   └── ops.py
│
├── layers/
│   ├── base.py
│   ├── dense.py
│   ├── activations.py
│   └── loss.py
│
├── network/
│   ├── model.py
│   └── optimizer.py
│
├── experiments/
│   ├── experiments/
│   │   └── 04_width_vs_depth.py
│   │
│   └── Experiments_Workshop/
│       ├── A/
│       │   ├── 04_width_vs_depth.py
│       │   └── 04b_visualize_failure.py
│       │
│       ├── B/
│       │   └── 05_optimizer_comparison.py
│       │
│       ├── C/
│       │   └── 06_min_width_per_function.py
│       │
│       └── D/
│           ├── 07_training_budget.py
│           └── 08_seed_robustness.py
│
├── results/
│   └── noprop_blocks3_cifar10.json
│
├── verification/
│   ├── gradient_check.py
│   └── compare_pytorch.py
│
├── visualization/
│   └── loss_landscape_3d.py
│
├── proofs/
│   └── main.tex
│
├── Summary_of_Experiments.pdf
└── .gitignore
````

---

## Requirements

```text
numpy
matplotlib
torch
```

Install the required packages with:

```bash
pip install numpy matplotlib torch
```

---

## Running the Experiments

Run the experiments from the repository root.

### Width vs. Depth

```bash
python experiments/experiments/04_width_vs_depth.py
```

### Workshop Experiments

#### Experiment A — Width, Depth, and Failure Visualization

```bash
python experiments/Experiments_Workshop/A/04_width_vs_depth.py
python experiments/Experiments_Workshop/A/04b_visualize_failure.py
```

#### Experiment B — Optimizer Comparison

```bash
python experiments/Experiments_Workshop/B/05_optimizer_comparison.py
```

#### Experiment C — Minimum Width per Function

```bash
python experiments/Experiments_Workshop/C/06_min_width_per_function.py
```

#### Experiment D — Training Budget and Seed Robustness

```bash
python experiments/Experiments_Workshop/D/07_training_budget.py
python experiments/Experiments_Workshop/D/08_seed_robustness.py
```

### Verification

```bash
python verification/gradient_check.py
python verification/compare_pytorch.py
```

### Visualization

```bash
python visualization/loss_landscape_3d.py
```

Generated experimental plots are saved alongside the corresponding experiment scripts.

---

## Experimental Study

The Universal Approximation Theorem guarantees that a sufficiently wide single-hidden-layer neural network can approximate any continuous function to arbitrary precision. However, the theorem does **not** guarantee that gradient descent will successfully find suitable weights within a realistic training budget.

This project studies that gap empirically using a from-scratch NumPy automatic differentiation framework.

The experiments compare neural network architectures with approximately equal parameter counts of around **500 parameters**, ranging from wide-shallow to deep-narrow configurations.

The experimental study examines:

* Optimizer choice
* Width versus depth
* Function complexity
* Minimum width required for different functions
* Training budget
* Random-seed robustness

---

## Key Finding

A particularly important result is observed on a **dual-frequency oscillatory target**.

The wide-shallow network exhibits a deterministic optimization failure:

```text
MSE ≈ 0.487
```

This corresponds approximately to predicting the mean of the target function.

The failure persists across:

* 5 random seeds
* 3 optimizers
* Training budgets up to 15,000 epochs

In contrast, a deep-narrow architecture with approximately the same number of parameters reaches:

```text
MSE < 5.1 × 10^-5
```

This demonstrates that the failure is not simply caused by insufficient parameter capacity.

---

## Width Sweep

A width sweep from **2 to 512 neurons** further investigates whether increasing the width of a single-hidden-layer network can overcome the failure.

The results show that no tested width crosses the success threshold for this particular function.

This suggests that the observed failure is better explained as a persistent optimization attractor associated with single-hidden-layer architectures rather than a lack of representational capacity.

---

## Main Conclusion

Under a fixed parameter budget, **depth can provide a practical optimization advantage that width alone does not**.

The experiments suggest that:

> The Universal Approximation Theorem establishes an existence guarantee, but does not guarantee that gradient descent will successfully realize the corresponding approximation within a finite and practical training budget.

The results provide experimental evidence of a gap between:

1. **Representational capacity** — whether an architecture can theoretically represent the target function.
2. **Optimization accessibility** — whether gradient-based training can actually find suitable parameters.

In the studied dual-frequency example, depth allows the network to escape a deterministic optimization failure that persists in wide single-hidden-layer networks.

---

## Verification

The implementation was independently checked in two ways.

### Numerical Gradient Checking

Relative error for the computed weight gradients is below:

```text
1 × 10^-10
```

### PyTorch Comparison

The from-scratch NumPy implementation agrees with PyTorch to approximately:

```text
3 × 10^-8
```

These checks provide confidence that the automatic differentiation and optimization components are implemented correctly.

---

## Project Motivation

The purpose of this project was not simply to build a neural network library, but to understand what happens underneath modern deep-learning frameworks.

Everything used for the core network implementation was built from first principles using NumPy, including:

* Tensor operations
* Computational graphs
* Reverse-mode automatic differentiation
* Backpropagation
* Dense layers
* Activation functions
* Loss functions
* Parameter initialization
* Optimizers

The experimental work then uses this framework to investigate questions about neural-network approximation and optimization that are not answered by the Universal Approximation Theorem itself.

---

## Experimental Questions

The project investigates several practical questions that are not answered directly by the UAT:

* Does increasing width always improve optimization?
* Can depth outperform width when the parameter budget is fixed?
* Does optimizer choice eliminate architecture-specific failures?
* How does function complexity affect approximation?
* Is there a minimum width required for particular target functions?
* Can simply increasing the training budget overcome optimization failures?
* Are observed failures reproducible across random seeds?

---

## Results and Reproducibility

The repository contains the generated experimental figures and result files used to analyze the experiments.

The main experimental results include:

```text
results_04_width_vs_depth_2.png
results_04b_failure_visualization.png
results_05_optimizer_comparison.png
results_06_min_width_per_function.png
results_07_training_budget.png
results_08_seed_robustness.png
```

A consolidated summary of the experimental work is also provided in:

```text
Summary_of_Experiments.pdf
```

---

## Important Distinction: UAT vs. Optimization

The Universal Approximation Theorem is fundamentally an **existence theorem**.

It tells us that, under appropriate conditions, there exists a neural network capable of approximating a target continuous function arbitrarily well.

It does **not** guarantee:

* A particular architecture will train successfully
* Gradient descent will find the required parameters
* A finite training budget will be sufficient
* A specific optimizer will reach the desired solution
* Greater width will necessarily eliminate optimization difficulties

This project experimentally investigates this distinction between what a neural network **can represent** and what gradient-based optimization can **actually discover**.

---

## What This Project Demonstrates

The project demonstrates that a neural network can be constructed without relying on high-level deep-learning frameworks while still supporting:

```text
Forward propagation
        ↓
Computational graph construction
        ↓
Reverse-mode automatic differentiation
        ↓
Gradient computation
        ↓
Parameter optimization
        ↓
Function approximation
```

The experimental framework then provides a controlled environment for studying how architectural choices affect optimization behavior.

---

## Project Motivation

Modern frameworks such as PyTorch and TensorFlow hide most of the mechanics involved in neural-network training.

This project was developed to understand those mechanics directly by implementing the underlying components from scratch.

Rather than treating neural networks as a black box, the project builds the system layer by layer and then uses it to study a deeper question:

> If a neural network has enough capacity to represent a function, under what conditions can gradient descent actually find that representation?

---

## Author

This project was developed as an extended study of:

* Neural networks
* Automatic differentiation
* Backpropagation
* Optimization
* Network architecture
* Function approximation
* The Universal Approximation Theorem


