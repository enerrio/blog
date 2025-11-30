# Jax 101

Recently, I’ve been exploring Jax by going through the documentation, watching YouTube videos, and experimenting in a Jupyter notebook. I’m used to PyTorch with its object-oriented focus, but Jax is a completely different beast. Jax puts an emphasis on functional programming which means functions should be pure and free of side effects. In this post, I’ll explain the basics of Jax and provide some exercises to help you get started on your journey learning Jax. This post assumes you have some familiarity with NumPy and machine learning.

## What is Jax?
From the Jax documentation[^1]:

> Jax is a python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.

Although Jax is popular in machine learning applications, it is a general-purpose library with many other use cases, such as in optimization and probabilistic programming. Developed by Google and open-sourced in 2018, Jax is used extensively internally[^2] at Google and has since become popular among practitioners.

![jax-star-history](https://i.ibb.co/34mDwcF/star-history-2024102.png)

## Why Jax
You might have raised an eyebrow at the “accelerator-oriented” line. Jax was built to be highly performant on specialized hardware like GPUs and TPUs. Arrays are automatically placed on available devices, so you won’t have to place them yourself like with PyTorch’s `to(device)` syntax. Jax makes it easy to run your code on different types of hardware without changing your program.

Another compelling reason to use Jax is its Just-In-Time (JIT) compilation capability. JIT-compiling your code compiles it using accelerated linear algebra (XLA), enabling it to run faster. XLA[^3] is a popular open source machine learning compiler for hardware accelerators. Later we’ll see how easy it is to use `jit` and also show the performance gains it provides. PyTorch recently added easy to use JIT capabilities with their `compile` function[^4] in version 2.X.

Two more powerful Jax tools are `grad` and `vmap`. `grad` computes the gradients of any given function,  similar to PyTorch’s `backward` function but more flexible. `vmap` can be applied to any function to automatically vectorize it. For example, you can have a function that computes the mean squared error loss between two scalars, and after applying `vmap`, it will be able to work with vectors without any code changes to the function itself! Before exploring those, let’s first look at the basics of Jax.

## Basics
![jnp-np](https://i.ibb.co/4J3r4ZT/jnp-np-logo.png)

Jax has a NumPy-like syntax so if you know NumPy then you are already familiar with a lot of Jax’s API! In fact, Jax provides an API available via `jax.numpy` that includes many of the same functions as NumPy. Here are some examples of building arrays in NumPy and in Jax:
```python
import numpy as np
import jax.numpy as jnp

arange_np = np.arange(5)
arange_jnp = jnp.arange(5)

linspace_np = np.linspace(-3, 3, 100)
linspace_jnp = jnp.linspace(-3, 3, 100)

zeros_np = np.zeros((10, 10), dtype=np.float16)
zeros_jnp = jnp.zeros((10, 10), dtype=jnp.float16)
```

They are exactly the same! While Jax and NumPy have similar APIs, if you look at the data types for the `arange` and `linspace` examples, you’ll notice that NumPy uses double precision, whereas Jax uses single precision. 
```python
print(f"NumPy dtype: {arange_np.dtype}")
print(f"Jax dtype: {arange_jnp.dtype}")
```
> NumPy dtype: int64  
Jax dtype: int32

This is a design choice by Jax because in many machine learning applications it is preferable to use single precision. It is one of a several key differences between NumPy and Jax. Jax’s sharp bits[^5] documentation page has more information on these key differences.

We can also plot the data created above using something like matplotlib without having to convert to NumPy.
```python
import matplotlib.pyplot as plt

plt.plot(linspace_jnp)
plt.title("Simple plot of jnp data");
```
![simple-plot](https://i.ibb.co/99xvjmv/linspaceplot.png)

## Randomness
So we’ve messed around a bit with the Jax API but what about creating arrays of pseudo-random numbers? Well this is another area where Jax differs from NumPy. You have much more control over random number generation in Jax through explicit random state management. Instead of defining a global state through the use of a random seed like in NumPy e.g. `np.random.seed(21)`, we explicitly have control over creating and updating the random state. 

In Jax we can create a pseudo-random number generator key and give it to a random function to generate an array. First let’s create the key.
```python
from jax import random

key = random.key(21)
print(key)
```
> Array((), dtype=key<fry>) overlaying:
[ 0 21]

Now we can generate some random data.
```python
x1 = random.normal(key, 3)
print(x1)
```
> [-2.6825788  -0.7566388  -0.29570565]

That looks great, but what happens if we were to use the same key again?
```python
x2 = random.normal(key, 3)
print(x2)
```
> [-2.6825788  -0.7566388  -0.29570565]

The exact same values were returned! Since the key represents our random state and remains unchanged, the returned values are identical. In Jax the user is responsible for ensuring that new keys are generated and used. Here’s an example of splitting our original key into two new keys.
```python
newkey1, newkey2 = random.split(key, 2)
print(newkey1)
print(newkey2)
```
> Array((), dtype=key<fry>) overlaying:
[  15689208 2943087094]  
Array((), dtype=key<fry>) overlaying:
[1648097174  339021355]

There’s much more to explore with the random module, check out the docs[^6]! Now let’s check out three of Jax’s biggest highlights.

## grad
At its core, `grad` is a function that takes another function as input and computes the gradient of its output with respect to its input. Jax provides automatic differentiation to make this process seamless. To see this in action, let’s define a ReLU function and use `grad` to create a new function that computes its derivative:
```python
from jax import grad

def relu(x):
    return jnp.maximum(0, x)

relu_grad = grad(relu)
```
`relu_grad` is now a function that will return the gradients of the original `relu` function. Note that `grad` will work on a function that outputs a scalar so we need to pass each element in `xs` to the `relu_grad` function separately like below.
```python
xs = jnp.linspace(-3, 3, 200)
ys_grad = [relu_grad(x) for x in xs]
```			
We can fix that later with `vmap`.

What if you have multiple inputs and want to get a partial derivative? You can control that as well with the `static_argnums` parameter (by default `grad` will take the derivative with respect to the first function parameter).
```python
def f(a, b):
    return 2 * a**3 - b**2

f_grad_0 = grad(f, argnums=0) # Derivative wrt `a`
f_grad_1 = grad(f, argnums=1) # Derivative wrt `b`
```

## vmap
`vmap` is also a function that takes another function as input. `vmap` will automatically vectorize the function to work with batches of data. To demonstrate, let’s look at the last example with `grad`. When we applied `grad` to the ReLU function it only worked on scalar outputs so we had to pass each element of our array to `relu_grad` separately. But if we apply `vmap` then we can transform the function to be able to handle multiple inputs at the same time.
```python
from jax import vmap

relu_vmap_grad = vmap(grad(relu))
ys_grad = relu_vmap_grad(xs) # can handle batches of data now
```

## jit
The last feature we’ll look at is `jit`. When you `jit` a function it gets compiled into optimized code that runs more efficiently. The function is compiled the first time it runs so when you call the jitted function on some input, Jax will “trace” it which just means extracting all the operations that act on the input. The result is a sequence of primitives (AKA fundamental units of computation) that are compiled using XLA.

Let’s see how easy it is to `jit` a function.
```python
from jax import jit

def f(x):
    x = x + 2
    x = x**2 - 4
    return jnp.sum(x)

f_jit = jit(f)
```
And now let’s look at the speedups gained. Note: This was run on a CPU but you should see similar gains on a GPU/TPU.
```python
xs = jnp.linspace(-10, 10, 1_000_000)
# warm up jitted function (i.e. it compiles 1st time it runs)
_ = f_jit(xs)

%timeit f(xs)
%timeit f_jit(xs).block_until_ready()
```
> 1.1 ms ± 44.3 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)  
346 μs ± 19.5 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

The first call to `f_jit` includes compilation time which is why we “warm up” the function so the measurement of execution time is accurate. Also note that we use `block_until_ready` to force Jax to wait until the function returns a value before continuing with the program. Don’t forget this! Otherwise Jax’s asynchronous dispatch[^7] will let the program run ahead before the computation completes and render the measurement inaccurate.

We can see these primitives using the `make_jaxpr` function.
```python
from jax import make_jaxpr

make_jaxpr(f)(xs)
```
> { lambda ; a:f32[1000000]. let  
    b:f32[1000000] = add a 2.0  
    c:f32[1000000] = integer_pow[y=2] b  
    d:f32[1000000] = sub c 4.0  
    e:f32[] = reduce_sum[axes=(0,)] d  
  in (e,) }

We can see the input is described as a variable `a` which is an array of one million 32-bit floating point numbers which matches the `xs` array we passed in. Then the following lines describe each operation that happens to `a`, resulting in the final value stored in `e`. This syntax can take some getting used to, so I recommend you play around with the input array’s data type and shape to see how the Jaxpr changes. This is a useful tool to see the inner workings of your function and making sure transformations are mapped to low level operations correctly. The full syntax has its own lengthy page in the documentation[^8].

But don’t `jit` everything! It may not compile properly if you have branches in your function that depend on runtime values. Try rewriting your code to avoid conditioning based on values, using special Jax-specific control flow operators, or `jit` other computationally heavy parts of the function. Generally, you want to `jit` functions that are computationally heavy or called repeatedly.

## Conclusion
In short here’s a summary of what makes Jax great:
* Easy autodiff with `grad`
* Faster execution time with `jit`
* Simple vectorization with `vmap`
* NumPy-like API

Finally, if you want to improve on your newly acquired skills, I’ve created a [Jupyter notebook](https://github.com/enerrio/jax-101/blob/main/exercises.ipynb) with some exercises for you to complete.
That’s it for this post. Next time we’ll go over applying what we learned here to train a simple neural network. Now try out those exercises and become a Jax ninja!

## Resources
[^1]: Jax documentation: https://jax.readthedocs.io/en/latest/index.html
[^2]: Google Internal use of Jax: https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/
[^3]: XLA: https://github.com/openxla/xla
[^4]: PyTorch Compile: https://pytorch.org/docs/stable/torch.compiler.html
[^5]: Jax Gotchas: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
[^6]: Jax PRNG: https://jax.readthedocs.io/en/latest/jep/263-prng.html#prng-design-jep
[^7]: Jax Asynchronous Dispatch: https://jax.readthedocs.io/en/latest/async_dispatch.html
[^8]: Jax Expression: https://jax.readthedocs.io/en/latest/jaxpr.html#jax-internals-jaxpr
