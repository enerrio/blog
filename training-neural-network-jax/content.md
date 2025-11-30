# Training a Neural Network with Jax

In my last [post](https://enerrio.bearblog.dev/jax-101/), we went over the basics of Jax. Did you complete the exercises 🤔? In this post, we will build on the skills learned so far and walk through training a neural network using Jax. We’ll also learn some new things along the way. You can follow along with the code [here](https://github.com/enerrio/jax-101/blob/main/nn.ipynb)[^1]. Let’s start with what every machine learning model needs: data.

## Creating a Dataset
Our objective will be to train a simple neural network on a polynomial regression task. With that in mind, let’s generate some synthetic data to work with. We’ll create data points that map to a polynomial function, which should be simple enough for our model to learn in a reasonable number of steps.
```python
import jax.numpy as jnp
from jax import random

def generate_polynomial_data(key, coefficients=[1, -2, 3], n_samples=100):
    key, subkey = random.split(key, 2)
    X = jnp.linspace(-10, 10, n_samples)
    y = jnp.polyval(coefficients, X) + random.normal(subkey, shape=(n_samples,))
    return X.reshape(-1, 1), y.reshape(-1, 1), key

key = random.key(21)
X, y, key = generate_polynomial_data(key, coefficients=jnp.array([1, -2, 1, -2]), n_samples=1_000)
X_norm = (X - X.mean()) / X.std()
y_norm = (y - y.mean()) / y.std()

print(f"X shape: {X.shape} - X dtype: {X.dtype}")
print(f"y shape: {y.shape} - y dtype: {y.dtype}")
```
> X shape: (1000, 1) - X dtype: float32  
y shape: (1000, 1) - y dtype: float32

Notice how we overwrite the random key inside the function and then return it. We do this because later we’ll want to randomly initialize the weights of the neural network, but we want to use a different key than the one we used to generate the synthetic data. In this simple example it will not matter much, but it’s good to keep in mind for real-world use cases. We also standardize the data to prevent the model from having to deal with values of a large magnitude. Large values would make training unstable.

Let’s visualize our data.
```python
import matplotlib.pyplot as plt
 
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Training data")
plt.grid(True);
```
![synthetic-data](https://i.ibb.co/th93hZn/traindata.png)

## Building the model
Now we can create the functions that are responsible for all aspects of training the neural network. Let’s start with the weights.
```python
def init_params(key):
    """Initialize model weights w/ Kaiming init."""
    W1_key, W2_key = random.split(key, 2)
    W1 = random.normal(W1_key, (1, 128), dtype=jnp.float32) * (5/3) / jnp.sqrt(1.)
    b1 = jnp.zeros(128, dtype=jnp.float32)
    W2 = random.normal(W2_key, (128, 1), dtype=jnp.float32) * (5/3) / jnp.sqrt(128.)
    b2 = jnp.zeros(1, dtype=jnp.float32)
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params
```
In the above code we are creating the weights for a 2-layer multi-layer perceptron with Kaiming initialization. One thing to note: We split the random key into 2 new keys, one for each of the layer weight initializations. Another thing to note: Kaiming what? As a refresher, Kaiming initialization[^2] is a weight initialization scheme that scales the weights by some value in order to help with model convergence by preserving the variance of activation values as they pass through layers. By activation values we mean the values right after they pass through a nonlinear function. The scaling factor we use is defined by $\frac{gain}{\sqrt{fan_{mode}}}$ where gain is a constant that depends on the nonlinearity being used and $fan_{mode}$ can be either `fan_in` or `fan_out`. `fan_in` and `fan_out` just mean the number of neurons in the input or output of the current layer.

In our case we use `fan_in` which is the number of **input** neurons in the layer. And later, we’ll use a tanh nonlinear function in the forward pass of our model so we’ll set the gain equal to $\frac{5}{3}$. We use the gain recommended for tanh according to PyTorch’s documentation[^3], which has a good overview of different weight initialization functions.

P.S. Try removing this scaling to see how it affects training.

Next, we’ll define three key functions:
1. Forward pass
2. Loss function
3. Weight updates

Remember that with Jax we practice functional programming and will need to write pure functions so we can JIT it later. For this reason, we will include everything a function needs in the signature and not reference variables that exist outside the function scope.
Let’s look at the forward pass first:
```python
def forward(params, x):
    """Forward pass of function."""
    x = jax.nn.tanh(jnp.dot(x, params["W1"]) + params["b1"])
    return jnp.dot(x, params["W2"]) + params["b2"]
```
This should look familiar: It’s simply defining the matrix multiplications that make up the forward pass of the model. The inputs are stored in `x` and are passed through each layer. This is where we use the tanh nonlinearity, right after the first layer. Since we’re doing a regression task we don’t apply a nonlinear function to the output layer. If we were training a classification model then we could apply a softmax function to the output layer to get a probability distribution over the class labels.

Next let’s check out the loss function, which for our use case will be mean squared error.
```python
def mse(params, x, y):
    """Mean squared error loss function."""
    y_pred = forward(params, x)
    return jnp.mean((y - y_pred) ** 2)
```
The function needs the model parameters and the features to make a prediction, just like the `forward` pass function. But we’ll also supply the ground truth values stored in `y` in order to compute mean squared error.

Finally, let’s define the function that is responsible for updating the model parameters.
```python
def update_params(params, gradients, lr=0.01):
    """Update model parameters."""
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, gradients)
    return new_params
```
Now we have a couple things to go over here. The `gradients` parameter is a dictionary that has an identical structure to the `params` dictionary but it will contain gradient values that we’ll compute later with `grad`. We’ll use the gradients to update the model parameters using a familiar equation: $W_{new} = W_{old} - \alpha * \frac{dL}{dW}$ where $\alpha$ is the learning rate to scale down the parameter updates. We don’t want to update our weights too far in one direction or our model might never converge!

You might notice that we are updating the model parameters and storing them in a new variable called `new_params`. Why don’t we just modify the parameters in place? Here we run into a unique aspect of Jax: Arrays are immutable. We can’t update arrays in-place[^4] so if you want to update a JAX array you must do it in the following way:
```python
arr = jnp.zeros((3, 3))
new_arr = arr.at[0, 0].set(1)
```

Now one more thing about `update_params`. We could loop through the dictionary keys and update the arrays that way, but instead we utilize the Jax function `jax.tree.map` to do this step more efficiently.

`jax.tree.map` is a utility function provided by Jax that applies a function to each element within a PyTree, returning a new PyTree with the transformed values. But what is a PyTree? To understand this let’s take a brief interlude to introduce the concept.

## PyTrees
In Jax, PyTrees are a tree-like data structure made up of nested Python containers. You can think of a PyTree as being made up of branches and leaves. A branch would be a container like a list, tuple, or dictionary. And a leaf would be the elements inside these containers, usually numerical values or arrays. Since branches themselves can be PyTrees, they have a recursive nature, where branches can be composed of other branches. Jax provides a bunch of built-in helper functions to easily manipulate PyTrees. Let’s take a look at some examples and what we can do with them in Jax.
```python
import jax
import operator

tree1 = [1, 2, 3]
tree2 = [(1, -2), (3, 4), 5]
tree3 = {"w1": 1.0, "w2": jnp.ones((3, 3)), "inner": {"w3": jnp.zeros((3, 3))}}

for i, tree in enumerate([tree1, tree2, tree3], 1):
    flattened_tree, _ = jax.tree.flatten(tree)
    print(f"Number of leaves in tree{i}: {len(flattened_tree)}")
```
> Number of leaves in tree1: 3  
Number of leaves in tree2: 5  
Number of leaves in tree3: 3

We can see that our branches in the above examples are simply common Python containers and the leaves are integers, floating point numbers, and Jax arrays. Once we flatten each tree we’re left with just the leaves. Let’s try out some of the other utility functions.
```python
vals, treedef = jax.tree.flatten(tree2)
reduced_val = jax.tree.reduce(operator.mul, tree2)
mapped_tree = jax.tree.map(lambda x: -x, tree2)
print(f"Original PyTree: {tree2}")
print(f"Reduced PyTree (multiplication): {reduced_val}")
print(f"Mapped PyTree (invert vals): {mapped_tree}")
print(f"Flattened PyTree: {vals}")
print(f"Tree definition: {treedef}")
```
> Original PyTree: [(1, -2), (3, 4), 5]  
Reduced PyTree (multiplication): -120  
Mapped PyTree (invert vals): [(-1, 2), (-3, -4), -5]  
Flattened PyTree: [1, -2, 3, 4, 5]  
Tree definition: PyTreeDef([(*, *), (*, *), *])

We can do things like apply a function to each leaf in a PyTree, reduce the leaves into a single value, and even see how Jax interprets the tree where `*` represents the leaves in the original PyTree. The most useful one for us is the `jax.tree.map` function which we can use to update every parameter value in our `params` dictionary.

PyTrees can be a little confusing at first, but try to think of them as simply containers of values. More generally, a PyTree leaf is anything that is **not registered** in the PyTree registry while a branch is anything that **is** registered. For more details, you can refer to the PyTree documentation page[^5]. With that out of the way, let’s get back to training the model!

## Model Training
So we left off with defining our `update_params` function which returns our modified model weights. Now we can define the training loop which will calculate a loss, compute gradients, and perform gradient descent.
```python
def train_loop(params, x, y, lr=0.01):
    loss, gradients = jax.value_and_grad(mse, argnums=0)(params, x, y)
    params = update_params(params, gradients, lr=lr)
    return params, loss
```
Instead of using `grad` directly we use the `jax.value_and_grad` function so we can get both the gradients and the actual loss value returned by `mse`. Remember that our `mse` function is doing the forward pass of the model for us so we don’t have to call it separately. Then we update the parameters and get new ones which we’ll use in the next forward pass.

Let’s initialize the parameters and then visualize the untrained model’s predictions.
```python
params = init_params(key)

# Plot untrained model's predictions
untrained_params = copy.deepcopy(params)
untrained_out = forward(untrained_params, X_norm)
untrained_out = (untrained_out * y.std()) + y.mean()

# Plot ground truth (y) vs input (X)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Ground Truth", color='blue', s=10)

# Plot predictions (untrained_out) vs input (X)
plt.scatter(X, untrained_out, label="Untrained Predictions", color='green', s=10)

# Add vertical error bars (red dotted lines) between predictions and ground truth
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i, 0], untrained_out[i, 0]], 'r--', linewidth=0.5)

plt.title("Ground Truth vs Untrained Predictions with Error Bars")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.legend();
```
![untrained-preds](https://i.ibb.co/QFtRL4m/untrained-Preds.png)

We can see the ground truth values are in blue, the model’s predictions are in green, and the red vertical lines represent the error between the predicted value and the true value.

Let’s actually train the model! Since we’re dealing with a small synthetic dataset we’ll skip splitting our training dataset into a validation/testing dataset or into smaller batches and just run full batch gradient descent.
```python
num_epochs = 1000
log_rate = 100
lr = 0.01
losses = []

start_time = time.time()
for i in range(num_epochs):
    params, loss = train_loop(params, X_norm, y_norm, lr)
    if (i % log_rate) == 0:
        print(f"Epoch [{i}/{num_epochs}] | Train Loss: {loss:.3f}")
    losses.append(loss)

end_time = time.time()
print(f"Total train time: {end_time-start_time:.2f} seconds")
```
> Epoch [0/1000] | Train Loss: 2.650  
Epoch [100/1000] | Train Loss: 0.100  
Epoch [200/1000] | Train Loss: 0.079  
Epoch [300/1000] | Train Loss: 0.070  
Epoch [400/1000] | Train Loss: 0.062  
Epoch [500/1000] | Train Loss: 0.057  
Epoch [600/1000] | Train Loss: 0.052  
Epoch [700/1000] | Train Loss: 0.047  
Epoch [800/1000] | Train Loss: 0.044  
Epoch [900/1000] | Train Loss: 0.040  
Total train time: 11.15 seconds

This looks reasonable! Our loss is gradually decreasing. Let’s use the same visualization from earlier but use the fully trained model to get the predicted values.
```python
# Plotting code is mostly the same except we use `params` instead of `untrained_params`
out = forward(params, X_norm)
out = (out * y.std()) + y.mean()
```
![trained-preds](https://i.ibb.co/nkVBkQz/trained-Preds.png)

That looks great! Our Jax model is learning the distribution of the training data. If we were to continue training we would likely dramatically overfit. In a real-world scenario we would use a validation data split to help monitor for that.

To dive deeper we can take a look at the Jaxpr of the `forward` function to see the step by step operations that are happening under the hood.
```python
jax.make_jaxpr(forward)(untrained_params, X)
```
> { lambda ; a:f32[1,128] b:f32[128,1] c:f32[128] d:f32[1] e:f32[1000,1]. let  
    f:f32[1000,128] = dot_general[  
      dimension_numbers=(([1], [0]), ([], []))  
      preferred_element_type=float32  
    ] e a  
    g:f32[1,128] = broadcast_in_dim[broadcast_dimensions=(1,) shape=(1, 128)] c  
    h:f32[1000,128] = add f g  
    i:f32[1000,128] = tanh h  
    j:f32[1000,1] = dot_general[  
      dimension_numbers=(([1], [0]), ([], []))  
      preferred_element_type=float32  
    ] i b  
    k:f32[1,1] = broadcast_in_dim[broadcast_dimensions=(1,) shape=(1, 1)] d  
    l:f32[1000,1] = add j k  
  in (l,) }  

But wait, what about JIT??

## The Power of JIT
Remember in the last post we learned about Just-In-Time (JIT) compilation and how it can make our code run faster? Why didn’t we use it during the training of our model? I wanted to first walk through the bare bones Jax training process before adding this extra complexity. We’ve trained our model already, and with this small dataset, it trained quickly. Now let’s see how much we can speed it up with JIT compilation.

A general rule of thumb is that you want to `jit` the outermost layer of your computation. Jax will trace the inputs through all the functions that get called so there’s no need to JIT the forward pass and the update step separately. If you did, then you’d be compiling some parts of the code twice which would be inefficient and could lead to slower execution time.
```python
train_loop_jit = jit(train_loop)

start_time = time.time()
for i in range(num_epochs):
    params, loss = train_loop_jit(params, X_norm, y_norm, lr) # train w/ jitted loop
    if (i % log_rate) == 0:
        print(f"Epoch [{i}/{num_epochs}] | Train Loss: {loss:.3f}")
    losses.append(loss)

end_time = time.time()
# Total train time: 11.15 seconds w/o JIT
# Total train time: 0.48 seconds w/ JIT
print(f"Total train time: {end_time-start_time:.2f} seconds")
```
> Epoch [0/1000] | Train Loss: 2.650  
Epoch [100/1000] | Train Loss: …  
…  
Epoch [900/1000] | Train Loss: …  
Total train time: 0.48 seconds

For the same number of epochs, we brought our training time down by a whole order of magnitude! On this simple setup it doesn’t mean much, but this will make a big difference on a real-world problem. We would see bigger gains by training on a hardware accelerator and taking advantage of modern training techniques like data sharding.

## Conclusion
I hope you enjoyed this post and learned something new! In the next Jax post I’d like to ramp up the difficulty and implement a GPT-2 model and go through the pretraining stage on a small-ish text dataset. See you then!

## Resources
[^1]: Code: https://github.com/enerrio/jax-101/blob/main/nn.ipynb
[^2]: Kaiming init: https://arxiv.org/pdf/1502.01852
[^3]: Recommended Gains: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
[^4]: In-place Updates: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates
[^5]: PyTrees: https://jax.readthedocs.io/en/latest/pytrees.html
