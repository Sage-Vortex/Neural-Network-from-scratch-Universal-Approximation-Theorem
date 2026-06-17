import numpy as np


# ---------- ADD ----------
def add_forward(a, b):
    return a + b

def add_backward(grad_output, a, b):
    grad_a = unbroadcast(grad_output, a.shape)
    grad_b = unbroadcast(grad_output, b.shape)
    return grad_a, grad_b


# ---------- MULTIPLY ----------
def mul_forward(a, b):
    return a * b

def mul_backward(grad_output, a, b):
    grad_a = unbroadcast(b * grad_output, a.shape)
    grad_b = unbroadcast(a * grad_output, b.shape)
    return grad_a, grad_b


# ---------- MATRIX MULTIPLY ----------
def matmul_forward(a, b):
    return a @ b

def matmul_backward(grad_output, a, b):
    grad_a = grad_output @ b.T
    grad_b = a.T @ grad_output
    return grad_a, grad_b


# ---------- POWER ----------
def pow_forward(a, exponent):
    return a ** exponent

def pow_backward(grad_output, a, exponent):
    return exponent * (a ** (exponent - 1)) * grad_output


# ---------- RELU ----------
def relu_forward(a):
    return np.maximum(0, a)

def relu_backward(grad_output, out_data):
    return (out_data > 0) * grad_output


# ---------- SUM ----------
def sum_forward(a):
    return a.sum()

def sum_backward(grad_output, a):
    return np.ones_like(a) * grad_output

# ---------- EXPONENTIAL ----------
def exp_forward(a):
    return np.exp(a)

def exp_backward(grad_output, out_data):
    return out_data * grad_output


# ---------- NATURAL LOG ----------
def log_forward(a):
    return np.log(a)

def log_backward(grad_output, a):
    return (1.0 / a) * grad_output


def unbroadcast(grad, shape):
    """
    Sum-reduce `grad` so it matches `shape`.
    Needed because broadcasting in forward pass means
    the gradient comes back in the *output* shape, not
    the original tensor's shape.
    """
    # Step 1: remove extra leading dimensions
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Step 2: sum over dimensions that were size 1 (broadcasted)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad