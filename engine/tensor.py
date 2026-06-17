import numpy as np
from engine import ops


class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # ---------- ADD ----------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(ops.add_forward(self.data, other.data), (self, other), '+')

        def _backward():
            da, db = ops.add_backward(out.grad, self.data, other.data)
            self.grad += da
            other.grad += db

        out._backward = _backward
        return out

    # ---------- MULTIPLY ----------
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(ops.mul_forward(self.data, other.data), (self, other), '*')

        def _backward():
            da, db = ops.mul_backward(out.grad, self.data, other.data)
            self.grad += da
            other.grad += db

        out._backward = _backward
        return out

    # ---------- MATMUL ----------
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(ops.matmul_forward(self.data, other.data), (self, other), '@')

        def _backward():
            da, db = ops.matmul_backward(out.grad, self.data, other.data)
            self.grad += da
            other.grad += db

        out._backward = _backward
        return out

    # ---------- POWER ----------
    def __pow__(self, exponent):
        out = Tensor(ops.pow_forward(self.data, exponent), (self,), f'**{exponent}')

        def _backward():
            self.grad += ops.pow_backward(out.grad, self.data, exponent)

        out._backward = _backward
        return out

    # ---------- RELU ----------
    def relu(self):
        out = Tensor(ops.relu_forward(self.data), (self,), 'relu')

        def _backward():
            self.grad += ops.relu_backward(out.grad, out.data)

        out._backward = _backward
        return out

    # ---------- SUM ----------
    def sum(self):
        out = Tensor(ops.sum_forward(self.data), (self,), 'sum')

        def _backward():
            self.grad += ops.sum_backward(out.grad, self.data)

        out._backward = _backward
        return out

    # ---------- BACKWARD PASS ----------
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    # ---------- CONVENIENCE OPERATORS ----------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return other + (-self)
# ---------- EXP ----------
    def exp(self):
        out = Tensor(ops.exp_forward(self.data), (self,), 'exp')

        def _backward():
            self.grad += ops.exp_backward(out.grad, out.data)

        out._backward = _backward
        return out

    # ---------- LOG ----------
    def log(self):
        out = Tensor(ops.log_forward(self.data), (self,), 'log')

        def _backward():
            self.grad += ops.log_backward(out.grad, self.data)

        out._backward = _backward
        return out


