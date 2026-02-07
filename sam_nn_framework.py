#!/usr/bin/env python3
"""
SAM Neural Network Framework - Custom Implementation
Replaces PyTorch with our own neural network library for AGI autonomy
"""

import numpy as np
import math
import random
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import copy
import time


class Tensor:
    """Custom tensor implementation for SAM neural networks"""

    def __init__(self, data: Union[list, np.ndarray, float, int], requires_grad: bool = False):
        if isinstance(data, (float, int)):
            data = np.array([data])
        elif isinstance(data, list):
            data = np.array(data)

        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.size = self.data.size

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)
            result._backward = lambda: self._add_grad(other, result)
            return result
        else:
            result = Tensor(self.data + other, self.requires_grad)
            result._backward = lambda: self._add_grad_scalar(other, result)
            return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)
            result._backward = lambda: self._mul_grad(other, result)
            return result
        else:
            result = Tensor(self.data * other, self.requires_grad)
            result._backward = lambda: self._mul_grad_scalar(other, result)
            return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)
            result._backward = lambda: self._matmul_grad(other, result)
            return result
        else:
            raise ValueError("Matrix multiplication requires Tensor")

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __neg__(self):
        return self.__mul__(-1.0)

    def __pow__(self, power):
        result = Tensor(self.data ** power, self.requires_grad)
        result._backward = lambda: self._pow_grad(power, result)
        return result

    def sum(self, axis=None):
        result = Tensor(np.sum(self.data, axis=axis), self.requires_grad)
        result._backward = lambda: self._sum_grad(axis, result)
        return result

    def mean(self, dim=None, keepdim=False):
        """Compute mean along specified axis (dim parameter for PyTorch compatibility)"""
        axis = dim  # PyTorch uses 'dim', our framework uses 'axis'
        result_data = np.mean(self.data, axis=axis, keepdims=keepdim)
        result = Tensor(result_data, self.requires_grad)
        result._backward = lambda: self._mean_grad(axis, keepdim, result)
        return result

    def exp(self):
        result = Tensor(np.exp(self.data), self.requires_grad)
        result._backward = lambda: self._exp_grad(result)
        return result

    def log(self):
        result = Tensor(np.log(self.data), self.requires_grad)
        result._backward = lambda: self._log_grad(result)
        return result

    def tanh(self):
        result = Tensor(np.tanh(self.data), self.requires_grad)
        result._backward = lambda: self._tanh_grad(result)
        return result

    def relu(self):
        result = Tensor(np.maximum(0, self.data), self.requires_grad)
        result._backward = lambda: self._relu_grad(result)
        return result

    def sigmoid(self):
        result = Tensor(1 / (1 + np.exp(-self.data)), self.requires_grad)
        result._backward = lambda: self._sigmoid_grad(result)
        return result

    def sqrt(self):
        result = Tensor(np.sqrt(self.data), self.requires_grad)
        result._backward = lambda: self._sqrt_grad(result)
        return result

    def transpose(self, axes=None):
        result = Tensor(np.transpose(self.data, axes), self.requires_grad)
        result._backward = lambda: self._transpose_grad(axes, result)
        return result

    def reshape(self, shape):
        result = Tensor(self.data.reshape(shape), self.requires_grad)
        result._backward = lambda: self._reshape_grad(result)
        return result

    @property
    def T(self):
        return self.transpose()

    def backward(self):
        """Compute gradients using reverse-mode autodiff"""
        if self.grad is None:
            raise RuntimeError("Tensor does not require gradients")

        # Initialize gradient
        self.grad = np.ones_like(self.data)

        # Build computation graph and compute gradients
        visited = set()
        topo_order = []

        def build_topo(node):
            if id(node) in visited:
                return
            visited.add(id(node))

            if hasattr(node, '_backward'):
                for child in getattr(node, '_children', []):
                    build_topo(child)

            topo_order.append(node)

        build_topo(self)

        # Execute backward pass
        for node in reversed(topo_order):
            if hasattr(node, '_backward') and node._backward is not None:
                node._backward()

    def zero_grad(self):
        """Zero out gradients"""
        if self.grad is not None:
            self.grad.fill(0)

    def detach(self):
        """Detach tensor from computation graph"""
        result = Tensor(self.data, requires_grad=False)
        return result

    # Gradient computation methods
    def _add_grad(self, other, out):
        """Accumulate gradients with proper broadcasting"""
        if self.requires_grad:
            if self.grad.shape != out.grad.shape:
                # Handle broadcasting issues
                if out.grad.shape == () and self.grad.shape == (1,):
                    self.grad += np.array([out.grad.item()])
                elif self.grad.shape == () and out.grad.shape == (1,):
                    self.grad = np.array([self.grad.item() + out.grad.item()])
                else:
                    self.grad += out.grad
            else:
                self.grad += out.grad

        if other.requires_grad:
            if other.grad.shape != out.grad.shape:
                # Handle broadcasting issues
                if out.grad.shape == () and other.grad.shape == (1,):
                    other.grad += np.array([out.grad.item()])
                elif other.grad.shape == () and out.grad.shape == (1,):
                    other.grad = np.array([other.grad.item() + out.grad.item()])
                else:
                    other.grad += out.grad
            else:
                other.grad += out.grad

    def _add_grad_scalar(self, scalar, out):
        if self.requires_grad:
            self.grad += out.grad

    def _mul_grad(self, other, out):
        if self.requires_grad:
            self.grad += other.data * out.grad
        if other.requires_grad:
            other.grad += self.data * out.grad

    def _mul_grad_scalar(self, scalar, out):
        if self.requires_grad:
            self.grad += scalar * out.grad

    def _matmul_grad(self, other, out):
        if self.requires_grad:
            self.grad += out.grad @ other.data.T
        if other.requires_grad:
            other.grad += self.data.T @ out.grad

    def _pow_grad(self, power, out):
        if self.requires_grad:
            self.grad += power * (self.data ** (power - 1)) * out.grad

    def _sum_grad(self, axis, out):
        if self.requires_grad:
            if axis is None:
                self.grad += out.grad
            else:
                self.grad += np.expand_dims(out.grad, axis=axis)

    def _mean_grad(self, axis, keepdims, out):
        if self.requires_grad:
            if axis is None:
                grad = out.grad / self.size
                self.grad += grad
            else:
                # Expand gradients back to original shape
                grad_shape = np.ones(self.ndim, dtype=int)
                grad_shape[axis] = self.shape[axis]
                grad = np.reshape(out.grad, grad_shape) / self.shape[axis]
                self.grad += grad

    def _exp_grad(self, out):
        if self.requires_grad:
            self.grad += out.data * out.grad

    def _log_grad(self, out):
        if self.requires_grad:
            self.grad += out.grad / self.data

    def _tanh_grad(self, out):
        if self.requires_grad:
            self.grad += (1 - out.data ** 2) * out.grad

    def _relu_grad(self, out):
        if self.requires_grad:
            self.grad += (self.data > 0).astype(float) * out.grad

    def _sigmoid_grad(self, out):
        if self.requires_grad:
            self.grad += out.data * (1 - out.data) * out.grad

    def _sqrt_grad(self, out):
        if self.requires_grad:
            self.grad += 0.5 * (self.data ** -0.5) * out.grad

    def _transpose_grad(self, axes, out):
        if self.requires_grad:
            self.grad += np.transpose(out.grad, axes)

    def _reshape_grad(self, out):
        if self.requires_grad:
            self.grad += out.grad.reshape(self.shape)


class Parameter(Tensor):
    """Parameter tensor that can be optimized"""

    def __init__(self, data: Union[list, np.ndarray, float, int]):
        super().__init__(data, requires_grad=True)


class Module(ABC):
    """Base class for neural network modules"""

    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        pass

    def parameters(self):
        """Get all parameters"""
        params = []
        for param in self._parameters.values():
            params.append(param)
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True):
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)

    def eval(self):
        """Set evaluation mode"""
        self.train(False)


class Linear(Module):
    """Linear layer"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights using Xavier initialization
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(np.random.randn(out_features, in_features) * scale)

        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Sequential(Module):
    """Sequential container for modules"""

    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class ReLU(Module):
    """ReLU activation"""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    """Tanh activation"""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    """Sigmoid activation"""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class MSELoss(Module):
    """Mean Squared Error Loss"""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target) ** 2).mean()


class BCELoss(Module):
    """Binary Cross Entropy Loss"""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Clamp predictions to avoid log(0)
        pred_clamped = Tensor(np.clip(pred.data, 1e-7, 1 - 1e-7))
        loss = -(target * pred_clamped.log() + (1 - target) * (1 - pred_clamped).log())
        return loss.mean()


class CrossEntropyLoss(Module):
    """Cross Entropy Loss"""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Assume pred is logits, target is class indices
        pred_exp = pred.exp()
        pred_sum = pred_exp.sum(axis=-1, keepdims=True)
        pred_softmax = pred_exp / pred_sum

        # Convert target to one-hot (simplified for integer targets)
        batch_size, num_classes = pred.data.shape
        target_onehot = np.zeros((batch_size, num_classes))
        target_onehot[np.arange(batch_size), target.data.astype(int).flatten()] = 1
        target_tensor = Tensor(target_onehot)

        loss = -(target_tensor * pred_softmax.log()).sum(axis=-1).mean()
        return loss


class Optimizer(ABC):
    """Base optimizer class"""

    def __init__(self, parameters: List[Tensor], lr: float = 1e-3):
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        """Update parameters"""
        pass

    def zero_grad(self):
        """Zero gradients"""
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def __init__(self, parameters: List[Tensor], lr: float = 1e-3, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.velocity[i] = self.momentum * self.velocity[i] + self.lr * param.grad
                param.data -= self.velocity[i]


class Adam(Optimizer):
    """Adam optimizer"""

    def __init__(self, parameters: List[Tensor], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                # Update biased second moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

                # Correct bias
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Update parameters
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Utility functions (similar to torch.nn.functional)
def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return MSELoss()(input, target)


def relu(input: Tensor) -> Tensor:
    return input.relu()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    # Subtract max for numerical stability
    input_max = input.data.max(axis=dim, keepdims=True)
    input_stable = input.data - input_max
    exp_input = np.exp(input_stable)
    exp_sum = exp_input.sum(axis=dim, keepdims=True)
    softmax_output = exp_input / exp_sum
    return Tensor(softmax_output, input.requires_grad)


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors"""
    data_list = [t.data for t in tensors]
    concat_data = np.concatenate(data_list, axis=dim)

    requires_grad = any(t.requires_grad for t in tensors)
    result = Tensor(concat_data, requires_grad)

    # Store references for gradient computation
    result._children = tensors
    result._concat_dim = dim

    def _concat_grad():
        # Split gradients back to original tensors
        splits = [t.data.shape[dim] for t in tensors]
        grad_splits = np.split(result.grad, np.cumsum(splits[:-1]), axis=dim)

        for i, tensor in enumerate(tensors):
            if tensor.requires_grad:
                tensor.grad += grad_splits[i]

    result._backward = _concat_grad
    return result


def randn(*shape, requires_grad: bool = False) -> Tensor:
    """Random tensor with normal distribution"""
    data = np.random.randn(*shape)
    return Tensor(data, requires_grad)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    """Zero tensor"""
    data = np.zeros(shape)
    return Tensor(data, requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    """One tensor"""
    data = np.ones(shape)
    return Tensor(data, requires_grad)


def eye(n: int, requires_grad: bool = False) -> Tensor:
    """Identity matrix"""
    data = np.eye(n)
    return Tensor(data, requires_grad)


def normal(mean: float = 0.0, std: float = 1.0, size=None, requires_grad: bool = False) -> Tensor:
    """Normal distribution tensor"""
    data = np.random.normal(mean, std, size)
    return Tensor(data, requires_grad)


# DataLoader implementation
class DataLoader:
    """Simple data loader for training"""

    def __init__(self, dataset: List[tuple], batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            # Separate inputs and targets
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]

            yield Tensor(np.array(inputs)), Tensor(np.array(targets))


# Training utilities
def train_one_epoch(model: Module, dataloader: DataLoader, optimizer: Optimizer, loss_fn):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model: Module, dataloader: DataLoader, loss_fn):
    """Evaluate model on validation/test data"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.data.item()
            num_batches += 1

    return total_loss / num_batches


# Export the framework
__all__ = [
    'Tensor', 'Parameter', 'Module', 'Linear', 'Sequential',
    'ReLU', 'Tanh', 'Sigmoid', 'MSELoss', 'BCELoss', 'CrossEntropyLoss',
    'Optimizer', 'SGD', 'Adam',
    'mse_loss', 'relu', 'tanh', 'sigmoid', 'softmax', 'cat',
    'randn', 'zeros', 'ones', 'eye', 'normal',
    'DataLoader', 'train_one_epoch', 'evaluate'
]

print("✅ SAM Neural Network Framework loaded - PyTorch replacement ready!")
