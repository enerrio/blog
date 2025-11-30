# GPT-2 in Jax

In this post we’ll be walking through creating and training a GPT-2 style model using Jax. In previous posts we learned the basics of Jax[^1], how to build simple neural networks[^2], and introduced libraries within the Jax ecosystem that make training neural networks easy[^3]. We’ll assume some familiarity with the Jax concepts covered in previous blog posts and with PyTorch DataLoaders[^4] which we'll use to help batch our data. This post is heavily inspired by the first half of Sebastian Raschka’s book "Build a Large Language Model."[^5] We’ll divide this post into four main components: data loading, model architecture, training, and text generation.

We’ll also organize our repo so it has the following structure:

```
├── config.py - Contains model hyperparameter configuration
├── entry_point.py - Our main script for running either training or inference
├── run_inference.py - Script to load a pertained model and generate text
├── run_train.py - Script to train our model
├── the-verdict.txt - Dataset to train on
├── tests/ - Unit tests
├── transformer/
│   ├── __init__.py
│   ├── data.py - Data preprocessing and create dataloaders
│   ├── infer.py - Functions with text sampling strategies
│   ├── model.py - GPT-2 model code
│   ├── train.py - Training loop
│   └── utils.py - Utility functions for model loading/saving and plotting
```

Let’s begin!

## Preparing the Dataset
For our dataset we’ll follow Sebastian’s lead and use The Verdict[^6] by Edith Wharton as our dataset. The dataset is very small, only 20 Kb and 20,479 characters. Let’s create a [map-style](https://pytorch.org/docs/stable/data.html#map-style-datasets) `Dataset` which will contain logic to tokenize our data and create data samples from raw text. Everything we cover here can go inside the `data.py` file.
```python
import tiktoken
from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    def __init__(
        self, text: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int
    ) -> None:
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        # Create dataset
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + 1 + max_length]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.input_ids[index], self.target_ids[index]

    def __len__(self) -> int:
        return len(self.input_ids)
```

We are using OpenAI’s tiktoken[^7] library to tokenize our text dataset using [byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding). Tiktoken has built-in support for a GPT-2 style tokenizer and all we have to do is load it and then we can convert raw text to token IDs and back. The `max_length` argument determines the context length we want our model to have i.e. the total number of tokens that the model takes as input. And the `stride` parameter determines how much we should shift each data sample. Here are a few examples to illustrate how the `stride` works:
```python
# stride of 1
input[0] = [0, 1, 2, 3]
input[1] = [1, 2, 3, 4]

# stride of 2
input[0] = [0, 1, 2, 3]
input[1] = [2, 3, 4, 5]  # shift the second sample by 2 positions

# stride of 3
input[0] = [0, 1, 2, 3]
input[1] = [3, 4, 5, 6]
```

Now that we have our dataset let’s create a function to create our DataLoader which will handle batching the data for us.
```python
def create_dataloader(
    text: str,
    tokenizer: tiktoken.Encoding,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Tokenize raw text data and create dataloader."""
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader
```

This is pretty simple. Why would we create a function for this if we have only one dataset? Like in any typical machine learning project we’ll want to split our data into a training set and a validation set that we can use to evaluate performance. We'll skip the test set in this case. We can re-use this function to create both a training DataLoader and a validation DataLoader. 

Wait, what’s with that `collate_fn` thing? The `collate_fn` parameter can be set to a callable that will take as input the result of our dataset’s `__getitem__` method and allows you to do some custom preprocessing before feeding data to your model. Think of it as an intermediate step between your data loader and the model. By default PyTorch’s DataLoader will convert data into PyTorch tensors but we want to deal with Jax arrays. So we’ll need a custom function to handle converting from PyTorch tensors to Jax arrays.
```python
from jaxtyping import Array, Int
import jax.numpy as jnp

def collate_fn(
    batch: list[tuple[list[int], list[int]]]
) -> tuple[Int[Array, "batch seq_len"], Int[Array, "batch seq_len"]]:
    """Convert tensors to Jax arrays."""
    input_batch, target_batch = zip(*batch)
    input_array = jnp.array(input_batch)
    target_array = jnp.array(target_batch)
    return input_array, target_array
```

Check out [this post](https://discuss.pytorch.org/t/how-to-use-collate-fn/27181) from the PyTorch forum for some more info on `collate_fn`. Also note how we’re starting to use one of the libraries from the previous blog post: jaxtyping. This helps us keep track of the data type and shape of our Jax arrays. In this case we’re returning arrays that have the same data type (Int) and shape `(batch, seq_len)`.

OK, let’s put it all together into one function that will give us our training and validation DataLoaders:
```python
from rich import print as rprint

def load_data(
    data_path: str,
    config: dict[str, int | float | bool],
    batch_size: int,
    train_ratio: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    """Load data, tokenize, and create dataloaders."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(raw_text))
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    rprint(f"Length of raw text file: {len(raw_text):,}")
    rprint(f"Total number of tokens: {total_tokens:,}")
    rprint(f"Total number of train characters: {len(train_data):,}")
    rprint(f"Total number of val characters: {len(val_data):,}")

    train_dataloader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    return train_dataloader, val_dataloader
```

There are some extra customizable parameters like controlling what percentage of the data should be allocated to the training set, the number of workers, batch size, and some helpful print statements using the rich[^8] library. The stride is set to be the same as the context length so that there are no overlapping sequences in the dataset. One reason for doing this is it ensures a given subsequence is seen only once in a single epoch which can be viewed as a form of regularization. On the flip side however, it can also lead to underutilization of the model by yielding fewer data samples than is possible with a smaller stride. This hyperparameter can be tweaked based on the use case and experimentation.

## Creating the Model
This part of the post will get to the meat of GPT-2: the architecture! We’ll cover implementing the entire model and take special consideration of Jax-specific additions. Everything in this section will go in the `model.py` file. We’ll implement four separate classes, each being an Equinox PyTree:

1. MultiHeadedAttention
2. MLP
3. TransformerBlock
4. GPTModel

In case you are unfamiliar with the GPT-2 model architecture here is a simplistic diagram:
![gpt-diagram](https://i.ibb.co/qpSWTrr/gpt2-drawio.png)

We’ll have multiple transformer blocks in our model and some other components such as layer normalization layers, dropout layers, etc. But from the diagram we can see that the `MultiHeadedAttention` and `MLP` class are nested within the `TransformerBlock` so let’s start with building those two classes.

### Attention
For the `MultiHeadedAttention` class we’ll be implementing the multi-headed attention logic that is common to modern language models. Since this will be an Equinox module let's subclass `eqx.Module` to make our attention class a PyTree that jells with Jax. We’ll also use `jax.vmap` to handle automatically vectorizing our forward pass and the einops[^9] library to make array reshaping much easier. Let’s start with the initialization.
```python
import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Int, Float, Bool, Key, PRNGKeyArray


class MultiHeadedAttention(eqx.Module):
    W_q: eqx.nn.Linear
    W_k: eqx.nn.Linear
    W_v: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    drop: eqx.nn.Dropout
    n_heads: int

    def __init__(self, cfg: dict[str, int | float | bool], key: PRNGKeyArray):
        key1, key2, key3, key4 = jr.split(key, 4)
        self.n_heads = cfg["n_heads"]
        assert (
            cfg["emb_dim"] % self.n_heads
        ) == 0, "Embedding dimension must be divisible by n_heads"
        self.W_q = eqx.nn.Linear(
            cfg["emb_dim"], cfg["emb_dim"], use_bias=cfg["qkv_bias"], key=key1
        )
        self.W_k = eqx.nn.Linear(
            cfg["emb_dim"], cfg["emb_dim"], use_bias=cfg["qkv_bias"], key=key2
        )
        self.W_v = eqx.nn.Linear(
            cfg["emb_dim"], cfg["emb_dim"], use_bias=cfg["qkv_bias"], key=key3
        )
        self.out_proj = eqx.nn.Linear(cfg["emb_dim"], cfg["emb_dim"], key=key4)
        self.drop = eqx.nn.Dropout(cfg["drop_rate"])
```

Each layer is given its own random key to initialize the weights and biases. The trainable weights should be defined as class attributes so that later, Equinox will know which values to modify during gradient descent. During training we’ll make use of a helper function called `eqx.filter_value_and_grad` which is similar to Jax’s built-in `jax.value_and_grad` function: Both take in a callable and will return the gradient and the callable’s return value. But the Equinox version will also filter out any non-trainable elements (i.e. everything that is not a floating point Jax/NumPy array). Let’s see what this looks like using a simpler version of the filter function: `eqx.partition`.
```python
mha = MultiHeadedAttention({"n_heads": 4, "emb_dim": 12, "qkv_bias": False, "drop_rate": 0.4}, jr.key(21))
params, static = eqx.partition(mha, eqx.is_array)

rprint(f"out_proj layer: {static.out_proj}\nn_heads attribute: {static.n_heads}")
rprint(f"out_proj layer: {params.out_proj}\nn_heads attribute: {params.n_heads}")
```
> out_proj layer: Linear(weight=None, bias=None, in_features=12, out_features=12, use_bias=True)  
n_heads attribute: 4

> out_proj layer: Linear(weight=f32[12,12], bias=f32[12], in_features=12, out_features=12, use_bias=True)  
n_heads attribute: None

The partition function will filter out the trainable and non-trainable parameters into two variables: `params` and `static`. When we look at the `out_proj` layer and the `n_heads` attribute for each variable we can see that the `params` will keep the weight and bias values intact and set `n_heads` to `None` while the `static` variable will do the opposite: set weights and biases to `None` and keep the `n_heads` value set to 4. The `eqx.filter_value_and_grad` function is a version of `partition` that includes calculating the gradient. Essentially during training this will tell Equinox to apply weight updates to only the floating point arrays in `params`.

Let’s look at a helper method that is responsible for creating the causal mask that will prevent past tokens from attending to future tokens:
```python
    def _create_causal_mask(self, seq_length: int) -> Bool[Array, "seq_len seq_len"]:
        """Creates a (seq_length, seq_length) boolean mask."""
        mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=jnp.bool))
        return mask
```

This function simply creates a boolean matrix with the upper triangular portion set to `False`. We’re using jaxtyping again to annotate our array shapes and data types.
```python
seq_length = 5
jnp.tril(jnp.ones((seq_length, seq_length), dtype=jnp.bool))
```
> Array([[ True, False, False, False, False],  
       [ True,  True, False, False, False],  
       [ True,  True,  True, False, False],  
       [ True,  True,  True,  True, False],  
       [ True,  True,  True,  True,  True]], dtype=bool)

Ok, now let’s take a look at the forward pass of our attention class:
```python
    def __call__(
        self,
        x: Float[Array, "seq_len emb_dim"],
        inference: bool = False,
        key: PRNGKeyArray = None,
    ) -> Float[Array, "seq_len emb_dim"]:
        queries = jax.vmap(self.W_q)(x)
        keys = jax.vmap(self.W_k)(x)
        values = jax.vmap(self.W_v)(x)
        queries = einops.rearrange(
            queries,
            "seq_len (heads head_dim) -> heads seq_len head_dim",
            heads=self.n_heads,
        )
        keys = einops.rearrange(
            keys,
            "seq_len (heads head_dim) -> heads seq_len head_dim",
            heads=self.n_heads,
        )
        values = einops.rearrange(
            values,
            "seq_len (heads head_dim) -> heads seq_len head_dim",
            heads=self.n_heads,
        )
        queries = queries / jnp.sqrt(keys.shape[-1])
        attention_scores = queries @ einops.rearrange(
            keys, "heads seq_len head_dim -> heads head_dim seq_len"
        )
        mask = self._create_causal_mask(x.shape[0])
        attention_scores = jnp.where(mask, attention_scores, -jnp.inf)
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.drop(attention_weights, inference=inference, key=key)
        context_weights = attention_weights @ values
        context_weights = einops.rearrange(
            context_weights,
            "heads seq_len head_dim -> seq_len (heads head_dim)",
        )
        out_proj = jax.vmap(self.out_proj)(context_weights)
        return out_proj
```

Instead of this becoming another self attention tutorial I’ll be focusing on the key differences that make all this work with Jax. So if any of the above is confusing or you just want a refresher, see [here](https://jalammar.github.io/illustrated-transformer/), [here](https://machinelearningmastery.com/the-transformer-attention-mechanism/), [here](https://www.youtube.com/watch?v=eMlx5fFNoYc), and chapter 3 of Sebastian’s [book](https://www.manning.com/books/build-a-large-language-model-from-scratch) for some more background on the attention mechanism.

The key things to talk about here have to do with `vmap` and inference. `vmap` is applied to the forward pass of our linear layers. Why do we do this? Let’s take the query projection `W_q` as an example. `W_q` has a weights matrix with shape `(emb_dim, emb_dim)`. So it is built to operate on a single embedding vector of shape `(emb_dim,)`. But our input to the forward pass `x` has a shape `(seq_len, emb_dim)` which means we have an embedding vector for each token in our context window. So in order to apply the query projection to every embedding vector in our input `x` we must vectorize that projection with `vmap`. When we apply `vmap` to `W_q` we allow that projection to now be able to work with a matrix of shape `(seq_len, emb_dim)`.

The next thing to talk about has to do with inference. We have a dropout layer in our attention class and during training we want to apply dropout randomly based on some dropout rate, which is a hyperparameter. But during inference we don’t want to apply dropout. Equinox has a built-in support for distinguishing between training mode and inference mode, we just have to set a boolean flag to the forward pass of the dropout layer to "switch" it on and off. We have an `inference` flag which should be `True` when running inference and also a `key` parameter which is set to `None` during inference. During training it is set to a Jax random key which will handle the random selection of parameters to drop. Any layer that has a different behavior between training mode and inference mode will have to be handled in this way so keep that in mind when building your own models with Equinox.

### MLP
Next up will be the multi-layer perceptron class which is very simple. Just two linear layers with a GeLU activation function in between.
```python
class MLP(eqx.Module):
    layers: list

    def __init__(self, cfg: dict[str, int | float | bool], key: PRNGKeyArray):
        key1, key2 = jr.split(key)
        self.layers = [
            eqx.nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4, key=key1),
            jax.nn.gelu,
            eqx.nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"], key=key2),
        ]

    def __call__(
        self, x: Float[Array, "seq_len emb_dim"]
    ) -> Float[Array, "seq_len emb_dim"]:
        for layer in self.layers:
            x = jax.vmap(layer)(x)
        return x
```

Like before we need separate random keys for each layer to handle weight initialization and in the forward pass we also apply `vmap` to handle vectorizing across the `seq_len` dimension of our input `x`. The layers and activation function are organized in a list to make iterating through them compact in the forward pass. Later in the training loop, Equinox will be able to handle filtering out the trainable elements from the non-trainable elements just like in the attention class. You might notice that when we iterate through the layers we are also applying `vmap` to the activation function. Although this is not necessary since the activation function will operate element-wise on the input, there is no harm to applying `vmap` and it makes the implementation look cleaner.

### Transformer Block
Now we can combine the attention class and the MLP class into a single transformer block.
```python
class TransformerBlock(eqx.Module):
    attn: MultiHeadedAttention
    mlp: MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    drop: eqx.nn.Dropout

    def __init__(self, cfg: dict[str, int | float | bool], key: PRNGKeyArray):
        key1, key2 = jr.split(key)
        self.attn = MultiHeadedAttention(cfg, key1)
        self.mlp = MLP(cfg, key2)
        self.ln1 = eqx.nn.LayerNorm(cfg["emb_dim"])
        self.ln2 = eqx.nn.LayerNorm(cfg["emb_dim"])
        self.drop = eqx.nn.Dropout(cfg["drop_rate"])

    def __call__(
        self,
        x: Float[Array, " seq_len emb_dim"],
        *,
        inference: bool = False,
        key: PRNGKeyArray = None,
    ) -> Float[Array, " seq_len emb_dim"]:
        if key is not None:
            key_attn, key_drop1, key_drop2 = jr.split(key, 3)
        else:
            key_attn = key_drop1 = key_drop2 = None
        shortcut = x
        x = jax.vmap(self.ln1)(x)
        x = self.attn(x, inference=inference, key=key_attn)
        x = self.drop(x, inference=inference, key=key_drop1) + shortcut

        shortcut = x
        x = jax.vmap(self.ln2)(x)
        x = self.mlp(x)
        return self.drop(x, inference=inference, key=key_drop2) + shortcut
```

We’re introducing a couple layer normalization layers and residual connections but there’s nothing too tricky here. One notable change is the random key creation in the forward pass. We need multiple keys to handle the layers with non-deterministic behavior. After all, we don’t want the second dropout application to behave **exactly** the same as the first dropout application. We want different connections to be dropped so that there is no correlation between those two dropout calls. So when we create those random keys we split them from the original `key` passed in. But during inference mode we don’t want to apply dropout so we have a section in the beginning of the forward pass to either create a key or set it to `None`. When a key is equal to `None` it is like we are not using a key at all i.e. dropout is not applied.

Another thing to note is the application of `vmap`. We don’t need to apply `vmap` to the MLP or to the attention class because we already use `vmap` directly inside the forward pass of those classes. But we do need to apply `vmap` to the layer normalization layer for the same reason we applied it to the linear layers: It is built to work on arrays of shape `(emb_dim,)` and we want to vectorize it to work on arrays of shape `(seq_len, emb_dim)`.

Almost there! One more section to go and then we’ll have the full model ready!

### Putting it all together
Now that we have all our building blocks ready we’re able to combine it.
```python
class GPTModel(eqx.Module):
    tok_embed: eqx.nn.Embedding
    pos_embed: Float[Array, "seq_len emb_dim"]
    drop_emb: eqx.nn.Dropout
    trf_blocks: list[TransformerBlock]
    final_norm: eqx.nn.LayerNorm
    out_head: eqx.nn.Linear

    def __init__(self, cfg: dict[str, int | float | bool], key: PRNGKeyArray):
        key1, key2, key3, key4 = jr.split(key, 4)
        self.tok_embed = eqx.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], key=key1)
        self.pos_embed = eqx.nn.Embedding(
            cfg["context_length"], cfg["emb_dim"], key=key2
        ).weight
        self.drop_emb = eqx.nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = [
            TransformerBlock(cfg, keyn) for keyn in jr.split(key3, cfg["n_layers"])
        ]
        self.final_norm = eqx.nn.LayerNorm(cfg["emb_dim"])
        self.out_head = eqx.nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], use_bias=False, key=key4
        )

    def __call__(
        self,
        x: Int[Array, " seq_len"],
        inference: bool = False,
        key: PRNGKeyArray = None,
    ) -> Float[Array, "seq_len vocab_size"]:
        if key is not None:
            key_drop, key_trf = jr.split(key, 2)
        else:
            key_drop = key_trf = None
        seq_len = x.shape[0]
        tok_embeds = jax.vmap(self.tok_embed)(x)
        x = tok_embeds + self.pos_embed[:seq_len, :]
        x = self.drop_emb(x, inference=inference, key=key_drop)
        for block in self.trf_blocks:
            key_trf, subkey_trf = key_split_allowing_none(key_trf)
            x = block(x, inference=inference, key=subkey_trf)
        x = jax.vmap(self.final_norm)(x)
        return jax.vmap(self.out_head)(x)
```

This should look pretty familiar by now. We’re creating an embedding layer to handle converting token IDs into embedding vectors, multiple transformer blocks, and an output projection that creates a probability distribution over the vocabulary which can be used to select the next token. We also use [absolute positional embeddings](https://stackoverflow.com/questions/73113261/the-essence-of-learnable-positional-embedding-does-embedding-improve-outcomes-b) to encode the position of a token into its vector representation. Since the token IDs for the positional embeddings are fixed (they are just integers from 0 to sequence length) we can work directly with the weight matrix which is why `self.pos_embed` is set to the embedding layer’s weight matrix. But when the positional embeddings are added to the token embeddings we index by the length of the input sequence `self.pos_embed[:seq_len, :]`. This isn’t necessary for training but during inference this will allow us to use variable-length sequences as input. Without it the model would always expect the input sequence to have the maximum number of tokens that the context window allows.

We also have a helper function `key_split_allowing_none` to help split random keys inside a for loop. This allows for a much more compact way of creating a random key inside the forward pass of each transformer block.
```python
from typing import Optional


def key_split_allowing_none(
    key: Optional[PRNGKeyArray],
) -> Key[Array, "2"] | tuple[None, None]:
    """Split key, if passed None then return original key."""
    if key is None:
        return key, None
    else:
        return jr.split(key)
```

You may have noticed that the function signature for some of the forward passes have a star in them. This is a feature of PEP 3102[^10] which makes every argument after the star a keyword-only argument. There is no need to include this in your models. I included it to make it more explicit when calling forward passes and because I noticed its prevalence in the Equinox source code.

## Training our Language Model
Now we can code the training loop in `train.py`. I’ll go over the main parts here but for the full implementation please refer to the [training script](https://github.com/enerrio/jax-transformer/blob/main/transformer/train.py).

First we will instantiate our GPT-2 model and also create the training and validation DataLoaders. We’re using the small architecture described [here](https://openai.com/index/better-language-models/) which has a size of 124 million parameters. The original GPT-2 architecture is doing weight tying (where the input embedding layer and the output linear layer share the same weights). But to make this implementation a little simpler I'm omitting this so the total number of parameters in our model will be slightly larger than the original GPT-2 small model.
```python
import optax
from config import GPT_CONFIG


# Create dataloaders
data_path = "the-verdict.txt"
train_dataloader, val_dataloader = load_data(
    data_path, model_config, batch_size=2
)

# Create model
key = jr.key(21)
model_key, train_key = jr.split(key)
model_config = GPT_CONFIG["small"]
model = GPTModel(model_config, model_key)

# Create optimizer
optim = optax.adamw(learning_rate=0.0004, weight_decay=0.1)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

# Calculate total number of trainable parameters
leaves, _ = jax.tree.flatten(model)
num_params = sum([leaf.size for leaf in leaves if eqx.is_array(leaf)])
rprint(f"Total number of model parameters (small): {num_params:,}")
```
> Total number of model parameters (small): 162,419,712

We have a configuration for the model's hyperparameters defined in `config.py` [here](https://github.com/enerrio/jax-transformer/blob/main/config.py). An Adam optimizer with weight decay is created and initialized as well, using the Optax library. This will give us our starting optimizer state which Optax can use to keep track of necessary statistics. See my [last post](https://enerrio.bearblog.dev/equinox-and-friends/#optax) for some more background. Let’s create a function for a single training step.
```python
from jaxtyping import PyTree, Scalar


@eqx.filter_jit
def train_step(
    model: eqx.Module,
    opt_state: PyTree,
    x: Int[Array, "batch seq_len"],
    y: Int[Array, "batch seq_len"],
    keys: Key[Array, " batch"],
) -> tuple[eqx.Module, PyTree, Scalar]:
    """Single training step for a batch of data. Forward pass, compute loss/grads, update weights."""

    def loss_fn(
        model: eqx.Module,
        x: Int[Array, "batch seq_len"],
        y: Int[Array, "batch seq_len"],
        keys: Key[Array, " batch"],
    ) -> Scalar:
        """Forward pass of model and compute loss."""
        logits = jax.vmap(model, in_axes=(0, None, 0))(x, False, keys)
        loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, y)
        return loss.mean()

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, keys)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
```

First, let’s talk about the `eqx.filter_jit` function that decorates `train_step`. This will just-in-time compile our function so that it can run fast. The filter part will prevent [tracing](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables) static arguments if they are not Jax arrays. This is similar to the `eqx.filter_value_and_grad` function that separates values into trainable and non-trainable parameters. Next, see how `keys` is also a Jax array? We want a unique random key for every data sample in our batch so we need to pass in a batch of random keys to use. Just like the dropout layers we do this so that the non-deterministic layers do not behave **exactly** the same way for every sample in the batch. 

Next is the loss function which does not need to be jitted because it is called as part of the `train_step` function. Remember that you **only need to jit the outermost layer of computation** to get a performance boost. Jitting the same layers twice is inefficient and could lead to slower execution times.

When we run our input `x` through the forward pass of the model we apply `vmap` to vectorize the forward pass across the batch dimension. Remember that our model’s forward pass is designed to operate on a single data sample, not a batch. This is a key part of Jax. `vmap` will handle vectorizing any given function so you don't have to deal with the batch dimension directly. Also note how we’re using `in_axes` to specify that we want to vectorize over the batch dimension (0th axis) of the input `x` and the `keys` array. The inference flag is set to False and since it’s a boolean we don’t want to vectorize over that parameter so we set the second element of `in_axis` equal to `None`. 

Next we can compute the loss using a built-in loss function from Optax. And finally, we can apply gradient descent and update our model’s parameters using our Optax optimizer and Equinox’s `apply_updates` function.

Let’s also create a similar function but for validation:
```python
def validate_step(
    inference_model: eqx.Module,
    x: Int[Array, "batch seq_len"],
    y: Int[Array, "batch seq_len"],
) -> Scalar:
    def validation_loss_fn(
        model: eqx.Module,
        x: Int[Array, "batch seq_len"],
        y: Int[Array, "batch seq_len"],
    ) -> Scalar:
        logits = jax.vmap(model, in_axes=(0, None, None))(x, True, None)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return loss.mean()

    loss = validation_loss_fn(inference_model, x, y)
    return loss
```

Looks pretty similar except there’s no need for an optimizer since we’re not updating the model’s parameters. Also since we’re just running inference we don’t need random keys and can set the inference argument to `True`. 

Now let’s actually write the training loop!
```python
train_stats = {"train_loss": [], "val_loss": [], "tokens_seen": []}

# Iterate over all epochs
for i in range(num_epochs):
    train_epoch_loss = val_epoch_loss = tokens_seen = 0.0
    # train phase
    for x_batch, y_batch in train_dataloader:
        key, *subkeys = jr.split(key, train_dataloader.batch_size + 1)
        subkeys = jnp.array(subkeys)
        model, opt_state, loss = train_step(
            model, opt_state, x_batch, y_batch, subkeys
        )
        train_epoch_loss += loss
        tokens_seen += x_batch.size
    # validation phase
    inference_model = eqx.nn.inference_mode(model)
    for x_val, y_val in val_dataloader:
        val_loss = validate_step(inference_model, x_val, y_val)
        val_epoch_loss += val_loss
    # Average and store loss
    train_epoch_loss /= len(train_dataloader)
    train_stats["train_loss"].append(train_epoch_loss)
    val_epoch_loss /= len(val_dataloader)
    train_stats["val_loss"].append(val_epoch_loss)
    train_stats["tokens_seen"].append(tokens_seen)
    rprint(f"Epoch [{i+1}/{num_epochs}] | Train Loss: {train_epoch_loss:.3f} | Val Loss: {val_epoch_loss:.3f}")
```

We’re going to keep track of the training/validation loss for each epoch as well as the total number of tokens we’ve trained on thus far so we can plot these later. Then we iterate over the total number of epochs we want to train for and switch between training and evaluating. During training we’ll need those keys we talked about earlier. We’ll want a unique random key for every data sample in our batch and in order for it to work well with `vmap` we’ll also convert it into an array of keys with `subkeys = jnp.array(subkeys)`. Also when we are evaluating our model we need to create a copy of it with all the non-deterministic layers like dropout switched to inference mode. We can easily do that with the `eqx.nn.inference_mode` function. We use that version of the model in the validation phase. Now the model should train as expected and we’ll get some training statistics that we can plot. Here’s an example of a model that trained for 30 epochs:
![loss](https://i.ibb.co/r4mp9NL/train-loss30nb-Custom.png)

Also we can save the model to disk using `eqx.tree_serialise_leaves`.
```python
with open("gpt-2.eqx", "wb") as f:
    eqx.tree_serialise_leaves(f, model)
```

Make sure to check out the [`train.py`](https://github.com/enerrio/jax-transformer/blob/main/transformer/train.py) script to see the full implementation. The only thing missing is some code for setting up a nice looking progress bar.

## Running Inference
Whew! That was a lot. We’re almost done. Let’s talk about inference because what’s the point of all that training if you can’t generate new text? Getting started with inference is pretty simple, we first need to load our model.
```python
model_key = jr.key(21)
skeleton = GPTModel(GPT_CONFIG["small"], model_key)
rprint("Loading model...")
with open("gpt-2.eqx", "rb") as f:
    model = eqx.tree_deserialise_leaves(f, skeleton)
rprint("Model loaded!")

model = eqx.nn.inference_mode(model)
tokenizer = tiktoken.get_encoding("gpt2")
prompt_tokens = text_to_token_ids("My name is chat and I am", tokenizer)
```

When we load in a model we first create a "skeleton" which is just a randomly initialized version of the model i.e. it is untrained. Then we can replace the weights with the ones from our trained model and create an inference mode copy of it. We'll use our tokenizer from earlier to convert a prompt into a sequence of token IDs.

Let’s create two more functions: `text_to_token_ids` which will convert text into token IDs and `token_ids_to_text` which will convert token IDs back to text.
```python
def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding):
    """Convert text to array of token IDs."""
    return jnp.array(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))


def token_ids_to_text(token_ids: Int[Array, " seq_len"], tokenizer: tiktoken.Encoding):
    """Convert sequence of token IDs to text."""
    return tokenizer.decode(token_ids)
```

The above two functions are pretty self-explanatory. The `allowed_special` argument will treat the text "<|endoftext|>" as its own special token. It is not strictly necessary for our inference use case but it is useful when, for example, you are training on multiple text sources concatenated together and want to distinguish between each source so you put this special token between them. That way the model can learn that although these texts are part of the same sequence, they are distinct from each other.

The `generate_text` function will run multiple forward passes of the model in order to sample new text.
```python
def generate_text(
    inference_model: eqx.Module,
    context: Int[Array, " seq_len"],
    max_new_tokens: int,
    context_size: int,
    key: PRNGKeyArray,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> Int[Array, " out_seq_len"]:
    """Run inference on some context using temperature scaling and top-k decoding strategies."""
    for _ in range(max_new_tokens):
        key, subkey = jr.split(key)
        idx_cond = context[-context_size:]
        logits = inference_model(idx_cond, inference=True)
        logits = logits[-1, :]
        # Apply top k filtering
        if top_k is not None:
            top_logits, _ = jax.lax.top_k(logits, top_k)
            min_val = top_logits[-1]
            logits = jnp.where(
                logits < min_val,
                jnp.full_like(logits, -jnp.inf),
                logits,
            )
        if temperature > 0.0:
            # Apply temperature scaling
            scaled_logits = logits / temperature
            idx_next = jr.categorical(subkey, scaled_logits, shape=(1,))
        else:
            # Apply greedy decoding
            idx_next = jnp.argmax(logits, keepdims=True)
        context = jnp.concatenate((context, idx_next))
    return context
```

First let’s note that we have a limit to the number of new tokens we’re generating which is controlled by `max_new_tokens`. Then we take our prompt (called `context` here) and truncate it so that it fits within our context window `idx_cond = context[-context_size:]`. When we run a single forward pass we get a probability distribution over **all** tokens in our context. But we only care about predicting the next token so we’ll take the logits[^11] for the last token in our context `logits = logits[-1, :]`. Then we can apply a decoding strategy to actually select a token.

We'll go over three possible decoding strategies here: greedy decoding, temperature scaling, and top-k.

Greedy decoding is the simplest strategy and is the default behavior for this function. It is simply taking the token with the largest probability. This is very simple but can lead to less "creative" output because it’s always taking the most likely token based on its training data.

Temperature scaling is the next strategy. This will scale the logits by some value typically below or above 1. A temperature value higher than 1 leads to a more diffuse probability distribution which gives tokens other than the most likely one a greater chance of being selected. When you have a value lower than 1 then the probability distributions become sharper and more similar to greedy decoding. But we are still sampling based on these probabilities using `jr.categorical` so even with a small temperature value it is not guaranteed that the most likely token will be selected.

With top-k we are actually limiting the number of tokens that can be sampled based on the logit values. The top k logits with the highest values will be kept where k is an integer chosen by the user. All other values will be set to negative infinity. For example if k is 10 then only the 10 token IDs with the highest logit values will be considered, all other tokens will be assigned a value of negative infinity to prevent them from being sampled.

We can also combine top-k and temperature scaling strategies to introduce more variety in the generated text. These aren’t the only decoding strategies available to us. To learn more about different decoding strategies check out this [article](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html).

After sampling the next token we append it to our context vector and then repeat the above process until we hit `max_new_tokens`.

For the prompt "My name is chat and I am" here is the generated text when `max_new_tokens` is 40, k is set to 10, and the temperature is 1.9:
> My name is chat and I am his that the picture for nothing--I told Mrs. Stroud so that he was his past! fullest reass hear that, in the height of his glory, he had dropped his strange-rooms with

## Enforcing jaxtyping & Unit Tests
One thing about jaxtyping is that it won’t actually check if your types are correct without taking one of two approaches. You can decorate individual functions with `jaxtyping.jaxtyped` to check that specific function but this would be cumbersome to copy and paste onto all our functions. Another approach is to use `jaxtyping.install_import_hook` to type check an entire codebase. If you want some more background on how this works check out the documentation [here](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/). We will take the hook approach and create an entry script `entry_point.py` that will serve as the main script to run, whether we are doing training or inference. Inside this script we will set up the jaxtyping hook.
```python
import importlib
import sys
from jaxtyping import install_import_hook
from rich import print as rprint

def main():
    # Parse command-line args
    if len(sys.argv) < 2:
        rprint("Usage: python entry_point.py <command> [args...]")
        rprint("Commands:")
        rprint("- train: Run the training script")
        rprint("- infer: Run the inference script")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    # Define the mapping of commands to scripts
    command_map = {
        "train": "run_train.py",
        "infer": "run_inference.py",
    }

    if command not in command_map:
        print(f"Unknown command: {command}")
        print("Available commands: train, infer")
        sys.exit(1)

    with install_import_hook(["transformer"], "typeguard.typechecked"):
        if command == "train":
            run_train = importlib.import_module("run_train")
            run_train.main(args)
        elif command == "infer":
            run_infer = importlib.import_module("run_inference")
            run_infer.main(args)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
```

Here’s an example of training our model for 30 epochs:
```bash
python entry_point.py train --nb_epochs 30 --batch_size 2 --plot_name train_loss.png --model_size small --experiment_name nb30 --data the-verdict
```

By starting training through the entry script jaxtyping will check the type annotations and raise an error if there is a mismatch. Try changing one of the annotations and see the script fail. This setup makes it easier to ensure the array types are being checked. The extra arguments are parsed by [argparse](https://docs.python.org/3/library/argparse.html) and can be customized to your liking. Check out the [repo](https://github.com/enerrio/jax-transformer/tree/main) associated with this post to see the full source code.

While writing the Equinox classes I found it helpful to write unit tests to compare my implementation with built-in layers from Equinox. This helped me ensure that my implementation was correct. Here’s a unit test to make sure the attention class matches the built-in attention class from Equinox:
```python
def test_multi_headed_attention(cfg, x, key):
    batch_size = 10

    # Initialize custom model
    custom_mha = MultiHeadedAttention(cfg, key=key)

    # Initialize Equinox model
    equiv_mha = eqx.nn.MultiheadAttention(
        num_heads=cfg["n_heads"],
        query_size=cfg["emb_dim"],
        output_size=cfg["emb_dim"],
        use_query_bias=cfg["qkv_bias"],
        use_key_bias=cfg["qkv_bias"],
        use_value_bias=cfg["qkv_bias"],
        dropout_p=cfg["drop_rate"],
        use_output_bias=True,
        key=key,
    )

    assert jnp.allclose(
        equiv_mha.query_proj.weight, custom_mha.W_q.weight
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.query_proj.bias, custom_mha.W_q.bias
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.key_proj.weight, custom_mha.W_k.weight
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.key_proj.bias, custom_mha.W_k.bias
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.value_proj.weight, custom_mha.W_v.weight
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.value_proj.bias, custom_mha.W_v.bias
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.output_proj.weight, custom_mha.out_proj.weight
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.output_proj.bias, custom_mha.out_proj.bias
    ), "weights not equal"

    # Create causal mask
    mask = jnp.tril(
        jnp.ones(
            (batch_size, cfg["context_length"], cfg["context_length"]), dtype=jnp.bool
        )
    )
    # Run custom model
    custom_output = jax.vmap(custom_mha, in_axes=(0, None, None))(x, True, None)
    # Run Equinox model
    equiv_output = jax.vmap(equiv_mha)(x, x, x, mask)

    # Compare outputs
    assert jnp.allclose(custom_output, equiv_output), "Outputs are not close"
    print("Test passed: Custom MHA and Equinox MHA produce identical outputs.")
```

I wrote my own versions of attention and the mlp block to help bolster my own understanding and to help further demonstrate the use of `vmap` in the forward passes. Check out the full [test suite](https://github.com/enerrio/jax-transformer/tree/main/tests) in the associated repo. I recommend writing unit tests often to make sure different parts of your code are working as expected.

## Final Thoughts

We’re done! Congratulations, if you’ve been following along then you can say you have implemented and trained GPT-2 in Jax. I hope this has helped bridge together some of the concepts I covered in previous posts and inspires you to build something interesting with Jax. It doesn’t have to end here though. There is a lot more to explore even with just this implementation. Here are some ideas on how you can build on this:
* Try training on a different dataset like Tiny Shakespeare[^12] or Tiny Stories[^13]
* It’s not shown here but in the repo the Rich library is used to create a progress bar that is displayed during training. Try exploring the Rich library and changing what the progress bar looks like. Check out [here](https://rich.readthedocs.io/en/stable/progress.html) for some inspiration with what is possible with Rich
* War and Peace[^14] by Leo Tolstoy is included in the repo. Try training on this dataset and see how the generated text and total train time changes
* Apply Equinox's [parallelism features](https://docs.kidger.site/equinox/examples/parallelism/) to parallelize training across multiple GPUs and see how this speeds up training

Let me know if you try out any of these ideas! Also make sure to check out the [Github repo](https://github.com/enerrio/jax-transformer/tree/main) to see the entire codebase.

## Resources

[^1]: Jax 101 Post: https://enerrio.bearblog.dev/jax-101/
[^2]: Training a NN with Jax Post: https://enerrio.bearblog.dev/training-a-neural-network-with-jax/
[^3]: Equinox & Friends Post: https://enerrio.bearblog.dev/equinox-and-friends/
[^4]: PyTorch DataLoader: https://pytorch.org/docs/stable/data.html
[^5]: Build a Large Language Model Book: https://www.manning.com/books/build-a-large-language-model-from-scratch
[^6]: The Verdict by Edith Wharton: https://en.wikisource.org/wiki/The_Verdict
[^7]: TikToken: https://github.com/openai/tiktoken
[^8]: Rich Library: https://rich.readthedocs.io/en/latest/
[^9]: Einops: http://einops.rocks
[^10]: PEP 3102: https://peps.python.org/pep-3102/
[^11]: Logits (unnormalized predictions): https://developers.google.com/machine-learning/glossary#logits
[^12]: Tiny Shakespeare: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
[^13]: TinyStories: https://huggingface.co/datasets/roneneldan/TinyStories
[^14]: War and Peace by Leo Tolstoy: https://www.kaggle.com/datasets/mehmetlaudatekman/war-and-peace-project-gutenberg
