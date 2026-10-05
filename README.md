# NeuroGrad

A scalar-valued autograd engine and neural network library built from scratch in Python.

NeuroGrad implements the core mechanics behind automatic differentiation and backpropagation using a small, readable computation engine. The project is built from first principles to develop a deeper understanding of how neural networks learn beneath high-level frameworks such as PyTorch.

## Features

- Scalar-valued automatic differentiation
- Dynamic computation graphs
- Operator overloading for mathematical operations
- Reverse-mode autodiff
- Backpropagation using the chain rule
- Gradient propagation through computation graphs
- Basic neural network primitives
- Multi-layer perceptrons (MLPs)

## How It Works

NeuroGrad represents each scalar value as an object that stores both its numerical value and its relationship to the values that produced it.

For example:

```python
a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + a

d.backward()

print(a.grad)
print(b.grad)
```

Operations between `Value` objects dynamically construct a computation graph.

Calling `backward()` traverses this graph in reverse topological order and applies the chain rule to calculate the gradient of the final output with respect to every value involved in the computation.

## Tech

- Python
- NumPy *(where applicable)*
- pytest *(for testing)*

## Status

**Work in progress**
