# rax

A toy deep learning library written in rust, very loosely inspired by the functional patterns of [Jax](https://github.com/google/jax), and also partly by [PyTorch](https://github.com/pytorch/pytorch) and [tinygrad](https://github.com/tinygrad/tinygrad).

The current api looks something like this (if slightly idealized):

```
model = (params, x, y) => loss # model is a pure function

loss = model(params, x, y) # `loss` is a computation graph is built

gradients_graph = loss.backward() # calling `backward` on the graph produces an isomorphic graph of gradients

new_param = update(params, gradients_graph)
```

## Features
- Tensors + a handful of ops (Binary/Unary/Reduce framing completely stolen from tinygrad)
    - Binary Ops: Matmuls, elementise arithmetic, max, etc.
    - Unary Ops: ReLU, square, etc.
    - Reduce Ops: mean, etc.
- Gradient computation.
    - `Node::backward` takes a computational graph and returns a trace of that graph with the parameters swapped for their gradients with respect to the head of the graph
- An SGD optimizer

## What I'm probably not going to do:
- Make it fast
    - It's ridiculously slow. for example broadcasting is implemented as a (in my opinion pretty elegant, but extremely slow) recursive tree traversal.
    - I'm fine with this, for now I'm interested in different challenges.
- Add helpful training constructs
    - Maybe I'll add a few more optimizers, but I'm not trying to actually train models with this.

## What's maybe next:
- Implementing more operations
    - I've implemented a few basic operations, but there are many more to go.
- Enabling less opinionated structure for storing parameters
    - currently, parameters are stored in HashMaps,
- Make the forward pass lazy
    - It could be cool to split the forward pass into 1) constructing the computation graph and 2) evaluating the graph.
    - This would allow for some cool things like:
        - Compilation to different backends (instead of my primitive tensor implementation)
        - Caching the graph for multiple evaluations ???
