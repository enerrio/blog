# Equinox and Friends

In this post I’ll introduce some new libraries that will make training neural networks in Jax much easier and more reliable. We’ll take the setup from the previous blog post[^1] and rewrite it using these libraries to showcase the advantages. This will all be a preamble to the next post where we’ll use what we learn here to create and train a GPT-2 model. Let’s begin!

## Equinox
There are a few libraries that make writing neural networks in Jax easy such as Flax and Haiku. But we’re going to be using Equinox[^2]. A great advantage to using Equinox is that everything is a PyTree - a nested container of arrays and other PyTrees, which Jax can traverse efficiently. See my [previous blog post](https://enerrio.bearblog.dev/training-a-neural-network-with-jax/#pytrees) for a primer on PyTrees. Other libraries will often create abstractions that make it difficult to work with and debug. But having entire machine learning models and layers represented as PyTrees makes building and debugging much easier.

The [documentation](https://docs.kidger.site/equinox/examples/mnist/) has several good examples on building neural networks with Equinox. I’ll show a brief example here and go over some important tips and tricks to keep in mind.

Let’s start by reusing some of the code from the previous blog post [Training a Neural Network with Jax](https://enerrio.bearblog.dev/training-a-neural-network-with-jax/) where we trained a neural network on a polynomial regression task using pure Jax. First we’ll create the polynomial data:
```python
import jax.numpy as jnp
from jax import random

def generate_polynomial_data(
    key,
    coefficients = jnp.array([1, -2, 3]),
    n_samples = 100,
):
    """Generate polynomial data."""
    key, subkey = random.split(key, 2)
    X = jnp.linspace(-10, 10, n_samples)
    y = jnp.polyval(coefficients, X) + random.normal(subkey, shape=(n_samples,))
    return X.reshape(-1, 1), y.reshape(-1, 1), key

key = random.key(21)
X, y, key = generate_polynomial_data(
    key, coefficients=jnp.array([1, -2, 1, -2]), n_samples=1_000
)
# standardize data
X_norm = (X - X.mean()) / X.std()
y_norm = (y - y.mean()) / y.std()

print(f"X shape: {X.shape} - X dtype: {X.dtype}")
print(f"y shape: {y.shape} - y dtype: {y.dtype}")
```
> X shape: (1000, 1) - X dtype: float32   
y shape: (1000, 1) - y dtype: float32

Now recall that last time our model was simply a pure function that performed the forward pass through a couple of dense layers.
```python
import jax

def forward(params, x):
    """Forward pass of function."""
    x = jax.nn.tanh(jnp.dot(x, params["W1"]) + params["b1"])
    return jnp.dot(x, params["W2"]) + params["b2"]
```
The `params` argument is a PyTree AKA a dictionary containing our layer weights. Now let’s create the same model, but using Equinox!
```python
import equinox as eqx

class Model(eqx.Module):
    layers: list[Any]

    def __init__(self, in_dim, out_dim, key):
        key1, key2 = random.split(key, 2)
        self.layers = [
            eqx.nn.Linear(
                in_dim, out_dim, use_bias=True, key=key1
            ),
            jax.nn.tanh,
            eqx.nn.Linear(out_dim, 1, use_bias=True, key=key2),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

If you’ve used PyTorch before then this should look familiar. We create a `Model` class which is subclassed from `eqx.Module` and implement an `__init__` method and a `__call__` method, the latter of which represents the forward pass of the model. With this one class we’re taking care of the forward pass **and** the weight initialization. Also the entire model itself is a PyTree!

Equinox has a bunch of built-in layers (also PyTrees) available through their API. In the above example we’re using the Linear layer. When we create the layers we also supply a unique random key to each layer to handle random weight initialization. We also store each layer in a list so we can iterate through it in the forward pass. Finally we define the model parameters as class attributes (i.e. outside the `__init__` and `__call__` methods) so that Equinox can later identify which parameters in the `Model` PyTree are trainable.

From here we’ll just need two more things: a loss function and the training loop. The loss function is the same one we used last time, mean squared error.
```python
def mse(model, x, y):
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)
```

In the loss function we’re running `x` through the forward pass of the model and then calculating the mean squared error. Here’s an important thing to point out: We’re wrapping [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap) around the model in order to vectorize it. Why do we do this? When we write the forward pass for our Equinox model (and more generally any function in Jax) we design it to operate on a single data point rather than a batch of data. So if our `x` input data has a batch dimension of `(batch_size, 1)` then our forward pass is written to operate on just a single data point with a shape of `(1,)`. So if we want to pass a batch of data through our model then we first need to vectorize it with `vmap`. 

By default `vmap` will vectorize a function across the first dimension of each input argument, which happens to work for our problem since the first dimension of `x` is the batch dimension. But in a more realistic setting (which we will see later) we sometimes pass multiple inputs to the model’s forward pass, some of which are not batched. So we will have to specify the `in_axes` argument of `vmap` to state if an argument should be vectorized and which dimension should be vectorized. Here’s an example of what I mean:
```python
class Model(eqx.Module):
    def __init__(self):
        # layers initialization…

    def __call__(self, x, inference=False, key=None):
        # x is of shape (num_features,)
        x = self.layer(x)
        x = self.dropout(x, inference=inference, key=key)
        return x

# Create sample input with size (batch_size, num_features)
x_batch = jnp.ones((32, 5))
# Create a random key for each data sample in our batch
key = random.key(21)
keys_batch = random.split(key, 32)
keys_batch = jnp.array(keys_batch)

model = Model()
# Vectorize over the x and keys arguments, but not the inference argument
predicted_batch = jax.vmap(model, in_axes=(0, None, 0))(x_batch, False, keys_batch)
```

For `in_axes` we used integers to specify which axis to vectorize over for **all** inputs to the model’s forward pass. The zero in the 1st position means we want to vectorize `x_batch` over the 1st dimension (which is the batch dimension in this case because it has shape `(batch_size, num_features)`). The `None` in the 2nd position means we do not want to vectorize over the boolean we are passing to the model. And the zero in the 3rd position means we want to vectorize `keys_batch` over the 1st dimension (also the batch dimension in this case). We created a random key for every sample in the batch so that the dropout layer behaves differently for each data point.

This can be a tricky thing to get used to. I encourage curious readers to explore the examples in the Equinox documentation for some more advanced usages. Later I’ll recap all the important things to keep in mind when training models.

We’re almost ready to write the training loop for our model. Let’s first discuss the Optax library which we’ll use to create an optimizer and update our model’s weights during training.


## Optax
Optax[^3] is a library for gradient processing and optimization in Jax. It works very well with Equinox and can be used to create optimizers that can update a model’s weights during training. The main object in Optax is `optax.GradientTransformation` which represents an operation to be applied to gradients. See this simple example where three gradient transformations are chained together to form one optimizer:
```python
max_norm = 100.
learning_rate = 1e-3

my_optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm),
    optax.scale_by_adam(eps=1e-4),
    optax.scale(-learning_rate))
```

The simplest example is [`optax.sgd`](https://optax.readthedocs.io/en/latest/api/optimizers.html#sgd) which is just plain gradient descent. A powerful feature of Optax is that it lets you chain together various transformations (like above) in order to build custom optimizers. There are a lot of other optimizers available within Optax like Adam and RMSProp but for our example we’ll stick with regular stochastic gradient descent.

We can create an optimizer like so:
```python
import optax

optim = optax.sgd(learning_rate=0.01)
```

Then we want to initialize the optimizer and store our optimizer’s current state in a variable.
```python
opt_state = optim.init(eqx.filter(model, eqx.is_array))
```

The optimizer state stores necessary statistics (gradients, momentum, weight decay, state of a learning rate scheduler, etc) that the optimizer needs to perform updates to the model’s parameters. The optimizer is initialized on the model but we want to initialize it only on the model’s trainable parameters so we filter out every PyTree that is not an array. In our example, the model has static parameters like the `jax.nn.tanh` activation function in its list of layers. We need to filter this out which we do with the helper function `eqx.filter` otherwise we’ll encounter an error. Here’s how we can actually use the optimizer to get new weight update values:
```python
updates, opt_state = optim.update(
    grad, opt_state, eqx.filter(model, eqx.is_array)
)
model = eqx.apply_updates(model, updates)
```

The update method will give us the updates to apply to the model as well as an updated optimizer state. There’s much more to explore with Optax but for now we have everything we need to write out training loop.

```python
def train(model, optim, x, y, num_epochs, log_rate):
    """Train the model."""
    losses = []
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, x, y, opt_state):
        """Single training step."""
        loss, grad = eqx.filter_value_and_grad(mse)(model, x, y)
        # simple way
        # model = jax.tree.map(lambda p, g: p - 0.01 * g if g is not None else p, model, grad)
        # using equinox
        # updates = jax.tree.map(lambda g: -0.01 * g, grad)
        # model = eqx.apply_updates(model, updates)
        # using optax
        updates, opt_state = optim.update(
            grad, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, loss

    for i in range(num_epochs):
        model, loss = train_step(model, x, y, opt_state)
        if (i % log_rate) == 0:
            print(f"Epoch [{i}/{num_epochs}] | Train Loss: {loss:.3f}")
        losses.append(loss)
    return model, losses
``` 

The above function has our training loop defined as well as a single train step which is jitted. Note that we’re using another Equinox helper function `eqx.filter_jit` to filter out non-trainable parameters and jit the train step for so our code runs faster. There are also three different ways for performing gradient descent included in the above code snippet: a simple way using pure Jax, another way using only Equinox, and a third way that uses Optax and Equinox. All three are equivalent but the last option is the easiest when dealing with more complex optimizers. Once you have created the model and the optimizer, you can use this function to train your model.

That just about wraps up our introduction to Equinox and Optax. Let’s take a brief look at one more library that will help you write clean code.

## jaxtyping
jaxtyping[^4] is a library that provides type hints and runtime type checking for Jax arrays and PyTrees. You can type hint your Jax arrays with their actual shapes so you can know at a glance what the data type and shape of a given array is. The runtime checker also helps enforce this shape by throwing an error if you try to pass an array that has a different shape or data type than what the hint states. By using type hints with jaxtyping we can catch shape and type errors early, making our code more robust and easier to debug.

Let’s look at a version of our loss function from earlier but with type hints.
```python
from jaxtyping import Array, Float, Scalar, jaxtyped
from typeguard import typechecked as typechecker

@jaxtyped(typechecker=typechecker)
def mse(model: Model, x: Float[Array, "batch_size 1"], y: Float[Array, "batch_size 1"]) -> Scalar:
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)
```

Now our `x` and `y` arguments are type hinted to say that they are Jax arrays of shape `(batch_size, 1)` and contain floating point values. We used a string and an integer in our type hint. The `batch_size` string is just to represent a variable-sized axis and is not hardcoded to a specific batch size. But the integer 1 means that the second axis should be of size 1. Anything else would cause an error. Our dataset has a size `(1000, 1)` so this is the correct array shape for our dataset. We can change the batch size to be something other than 1000 and the type hint would still be valid. The function’s return type is not an array, but just a single floating point loss value which we can type hint with `Scalar`. Check out the [documentation](https://docs.kidger.site/jaxtyping/api/array/) for a deeper look at these hints and how they can be used. Finally, we decorate the function with `jaxtyped` to ensure that the function actually gets checked during runtime and raise an error if the type hints don’t match what is passed to the function.

I’m using the typeguard[^5] library but you can use any type checker you want, like beartype[^6]. 


## Summary of Best Practices
Let’s revisit all the important points to keep in mind when you are training neural networks in Jax:
1. Write the forward pass for a **single** data sample. In other words, pretend that the batch dimension doesn’t exist. `vmap` will automatically vectorize your forward pass across the batch dimension for you.
2. Use `vmap` when running batches of data through the forward pass of your Equinox model.
3. When using `vmap` for the forward pass, make use of the `in_axes` argument to properly vectorize over the batch dimension for **all** forward pass inputs.
4. Use Optax to manage your optimizers.
5. Use Equinox’s filter functions to properly filter out non-Jax arrays from your model PyTree before applying weight updates or jit.
6. Use jaxtyping to type hint your Jax arrays and make sure to use the runtime checker to enforce compliance.

With these tools at our disposal we are well-equipped to tackle more complex models. In the next post, we’ll apply what we’ve learned here to create and train a GPT-2 model. To see the full code check out the repo[^7]. Try running the code and [reach out](https://enerrio.bearblog.dev/about-me/) to me if you have any questions or comments. See you next time 👻

## Resources

[^1]: Previous blog post: https://enerrio.bearblog.dev/training-a-neural-network-with-jax/
[^2]: Equinox: https://docs.kidger.site/equinox/
[^3]: Optax: https://optax.readthedocs.io/en/latest/index.html
[^4]: jaxtyping: https://docs.kidger.site/jaxtyping/
[^5]: Typeguard: https://github.com/agronholm/typeguard
[^6]: Beartype: https://github.com/beartype/beartype
[^7]: Code: https://github.com/enerrio/jax-transformer/blob/main/equinox_test.py
