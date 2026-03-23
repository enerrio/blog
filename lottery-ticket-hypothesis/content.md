# The Lottery Ticket Hypothesis

By 2019, it was well known that you could reduce the parameter count of a fully trained neural network without sacrificing performance[^1]. This was done by "pruning" unnecessary weights. However, that same year, a new paper[^2] came out that took the idea of pruning further and found a surprising conclusion: retraining a pruned network from scratch can lead to a model that matches the performance of the original, un-pruned dense model. 

In this post, I'll talk about what the lottery ticket hypothesis is and discuss some follow-up work in the field.

## Model Pruning

First, how do we define pruning? Pruning means removing unnecessary weights from a neural network. How do you know which weights are unnecessary? There are many ways[^3] to prune your model, but I'll discuss the simplest technique that was used in the lottery ticket hypothesis (LTH) paper. 

Usually, weights that have a low magnitude do not contribute much to the model. So removing them doesn't affect performance too much. Of course, this depends on how much you prune, but in certain settings you can prune up to 95% of weights and still match the performance of the original network.

## What is the lottery ticket hypothesis?

The lottery ticket hypothesis is defined as the following:
> A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations.

The setup for testing the lottery ticket hypothesis goes like this:
1. Randomly initialize a neural network.
2. Train the model on some task till convergence.
3. For each layer, prune p% of the lowest magnitude weights.
4. Reset the remaining weights back to their initial value from step one **before** training.
5. Train the new subnetwork until convergence.

The resulting subnetwork, trained on the same amount of data and for the same amount of time, tends to match the performance of the original dense network. This is known as a "winning ticket."

The paper explores two different ways of pruning the model: one-shot and iterative.
> **One-Shot pruning**: Prune once at the end of training and retrain sparse subnetwork.  
> **Iterative pruning**: Train the model to convergence, prune a fraction of the lowest-magnitude weights, reset the remaining weights to their original initialization, and retrain the pruned network to convergence. Repeat this cycle 3-5 times to progressively increase sparsity.

The paper uses iterative pruning rather than one-shot because iterative pruning found winning tickets at higher levels of sparsity. The downside is that iterative pruning requires multiple training runs, where each training run increases the level of sparsity. Meanwhile, one-shot pruning requires only one training run and then you prune just once at the end.

Winning tickets appeared in both shallow and deep networks, although the authors found that deeper networks tended to be more sensitive to the learning rate and winning tickets did not appear using iterative **or** one-shot pruning methods. [Later research](https://arxiv.org/abs/1903.01611) showed a way to surface winning tickets in deeper networks is to "rewind" the weights back to their values at an earlier training iteration rather than the initial values.

Rewinding showed that the sparse network architecture itself wasn't the only requirement for winning tickets to appear. Early training was necessary for "pushing" the model into a region where a subnetwork could learn effectively.

## Why does this work?

The authors think that stochastic gradient descent seeks out and trains a subset of well-initialized weights. And therefore dense randomly-initialized networks are easier to train than sparse networks because there are more possible subnetworks from which SGD might recover a winning ticket.

A closely related idea is that large neural networks are considered over-parameterized for most tasks. Meaning there are more parameters than are necessary for it to learn the task. In other words, there are more possible combinations of active weights (subnetworks) that might contain winning tickets. So LTH is, in a way, cutting through that over-parameterized network to find a winning ticket.

While LTH seeks a specific subnetwork (winning ticket) after training, there's actually a much more common way to work with subnetworks **during** training: Dropout[^4]. When you're training a network with dropout, the neuron outputs in a layer are randomly dropped, i.e. zeroed out, at each iteration. So at each training step, some weights are not utilized at all, which is like constructing a subnetwork out of the weights that **do** get utilized. Since dropout is applied randomly to different neurons, you're essentially training an ensemble of random subnetworks. 

Dropout works because it implicitly searches through a set of random subnetworks during training while LTH shows a specific winning subnetwork can be found post-training via pruning.

With that context, I'll go over my own implementation of LTH along with some related experiments.

## Paper reproduction 
I implemented a reproduction of the MLP and Convnet experiments on MNIST and CIFAR-10 datasets from the original paper in [JAX](https://docs.jax.dev/en/latest/index.html). I did not do experiments on deeper networks though. You can find this in my Github repo here: [https://github.com/enerrio/lottery-ticket-hypothesis/tree/main](https://github.com/enerrio/lottery-ticket-hypothesis/tree/main)

<p><em>Click any plot to explore it interactively.</em></p>
<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:0.8rem; text-align:center;">
  <figure>
    <a href="https://enerrio.github.io/lottery-ticket-hypothesis/outputs/mnist/mlp/plots/accuracy_vs_sparsity.html" target="_blank" rel="noopener noreferrer">
      <img src="https://i.ibb.co/2p0sLHq/mlp.png" alt="MLP on MNIST accuracy vs sparsity" style="width:100%; height:auto;" />
    </a>
    <figcaption>MLP on MNIST</figcaption>
  </figure>

  <figure>
    <a href="https://enerrio.github.io/lottery-ticket-hypothesis/outputs/cifar10/conv2/plots/accuracy_vs_sparsity.html" target="_blank" rel="noopener noreferrer">
      <img src="https://i.ibb.co/99PFhgvy/conv2.png" alt="Conv2 on CIFAR-10 accuracy vs sparsity" style="width:100%; height:auto;" />
    </a>
    <figcaption>Conv2 on CIFAR-10</figcaption>
  </figure>

  <figure>
    <a href="https://enerrio.github.io/lottery-ticket-hypothesis/outputs/cifar10/conv4/plots/accuracy_vs_sparsity.html" target="_blank" rel="noopener noreferrer">
      <img src="https://i.ibb.co/v6bcqCRs/conv4.png" alt="Conv4 on CIFAR-10 accuracy vs sparsity" style="width:100%; height:auto;" />
    </a>
    <figcaption>Conv4 on CIFAR-10</figcaption>
  </figure>

  <figure>
    <a href="https://enerrio.github.io/lottery-ticket-hypothesis/outputs/cifar10/conv6/plots/accuracy_vs_sparsity.html" target="_blank" rel="noopener noreferrer">
      <img src="https://i.ibb.co/W8x4Nmj/conv6.png" alt="Conv6 on CIFAR-10 accuracy vs sparsity" style="width:100%; height:auto;" />
    </a>
    <figcaption>Conv6 on CIFAR-10</figcaption>
  </figure>
</div>

My results were basically in line with the paper, so nothing new there. But I did some more experiments which I'll cover in the next couple sections. I did this implementation in JAX (see my earlier tutorials [here](https://enerrio.bearblog.dev/jax-101/), [here](https://enerrio.bearblog.dev/training-a-neural-network-with-jax/), and [here](https://enerrio.bearblog.dev/equinox-and-friends/)).

You can see more plots in my Github repo [here](https://github.com/enerrio/lottery-ticket-hypothesis/tree/main/outputs).

### Pruning the data

What if you prune the dataset itself? Would a sparse subnetwork trained on less data still match the dense network's test accuracy? A related idea was already explored in [Efficient Lottery Ticket Finding: Less Data is More](https://arxiv.org/abs/2106.03225). They did a "targeted" pruning to find a specific subset of the data that is critical to improving the performance of the sparse network. In some scenarios they are able to reduce their dataset size by >70%. I did not try to recreate their method, but their results look promising.

My implementation was simpler: take stratified subsets of the dataset (10%, 25%, 50%), retrain the networks (dense networks & winning tickets), and see if the resulting test accuracy matches the dense network's performance. The downsampled datasets are stratified to avoid any class imbalance. 

It's important to call out that I'm comparing sparse vs dense networks at each data budget, **not against** the dense model trained on 100% of the data. Almost none of the networks trained on subsets of the dataset beat the dense model trained on the full dataset.

I included the results of that in my repo as well.

<p><em>Click any plot to explore it interactively.</em></p>
<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:0.8rem; text-align:center;">
  <figure>
    <a href="https://enerrio.github.io/lottery-ticket-hypothesis/outputs/mnist/mlp/plots/data_efficiency.html" target="_blank" rel="noopener noreferrer">
      <img src="https://i.ibb.co/93NssvC2/mlp-DE.png" alt="MLP data efficiency on MNIST" style="width:100%; height:auto;" />
    </a>
    <figcaption>MLP Data Efficiency on MNIST</figcaption>
  </figure>

  <figure>
    <a href="https://enerrio.github.io/lottery-ticket-hypothesis/outputs/cifar10/conv2/plots/data_efficiency.html" target="_blank" rel="noopener noreferrer">
      <img src="https://i.ibb.co/T3Tq2ML/conv2DE.png" alt="Conv2 data efficiency on CIFAR-10" style="width:100%; height:auto;" />
    </a>
    <figcaption>Conv2 Data Efficiency on CIFAR-10</figcaption>
  </figure>

  <figure>
    <a href="https://enerrio.github.io/lottery-ticket-hypothesis/outputs/cifar10/conv4/plots/data_efficiency.html" target="_blank" rel="noopener noreferrer">
      <img src="https://i.ibb.co/SF2Xz31/conv4DE.png" alt="Conv4 data efficiency on CIFAR-10" style="width:100%; height:auto;" />
    </a>
    <figcaption>Conv4 Data Efficiency on CIFAR-10</figcaption>
  </figure>

  <figure>
    <a href="https://enerrio.github.io/lottery-ticket-hypothesis/outputs/cifar10/conv6/plots/data_efficiency.html" target="_blank" rel="noopener noreferrer">
      <img src="https://i.ibb.co/whX6nXVQ/conv6DE.png" alt="Conv6 data efficiency on CIFAR-10" style="width:100%; height:auto;" />
    </a>
    <figcaption>Conv6 Data Efficiency on CIFAR-10</figcaption>
  </figure>
</div>

The results on MNIST were kind of a wash because the task itself was so simple that even a dense network could learn on just 10% of the training data. The winning tickets did outperform the dense networks on pruned data, but just barely.

The CIFAR-10 experiments were more enlightening. Training sparse sub-networks (removing 50% of weights) on just a fraction of the dataset (50% and 25%) did outperform the dense baseline. At the same time though, if we pruned too much data or too much of the network, then performance collapsed.

This suggests that while the sparse architecture can help the model learn from less data because it acts as some kind of architectural prior, pruning away too much of the model leaves a network that doesn't have the capacity to learn the task from less data. Similarly, the sparse architecture of the winning ticket can only help so much, pruning too much of the data doesn't allow the subnetwork to effectively learn the task. 

### Ablations

In ML research, an ablation is an experiment where you deliberately remove or modify one component of a system to understand its contribution to the overall result. The next few experiments try to illustrate why magnitude pruning works by modifying the weights in a couple different ways.

What if we tried to do the opposite of the pruning that we discussed earlier and pruned a percentage of the **largest** magnitude weights? This was just a clean ablation to test if many low-magnitude weights can contribute meaningfully to the performance of the network. As opposed to a small percentage of large magnitude weights. As expected, pruning even a small amount of the large magnitude weights severely damages the performance of the network. You can see the results of this in one of my Jupyter notebooks named [inverse-pruning.ipynb](https://github.com/enerrio/lottery-ticket-hypothesis/blob/main/notebooks/inverse-pruning.ipynb).

What if we clamp the magnitude of weights during training so that the distribution is roughly uniform? The result is that when you prune the network, it's essentially pruning it at random, which destroys performance. You can see the results of this experiment in another notebook named [weight-clamping.ipynb](https://github.com/enerrio/lottery-ticket-hypothesis/blob/main/notebooks/weight-clamping.ipynb).

I also plotted the [Spearman rank correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) of weight magnitudes between pruning rounds in order to detect when the winning ticket structure stabilizes. This is similar to the ideas discussed in the [early bird paper](https://arxiv.org/abs/1909.11957) which tries to find winning tickets early in training. By looking at the Spearman rank we can see how pruning affects weight magnitudes, this acts as a proxy for seeing when the ranking of weights by magnitude stabilizes across pruning rounds.

My results showed that rank correlation increases steadily across all pruning rounds for the MLP without a clear stabilization point. The output layer stabilizes early (correlation ~0.84) but hidden layers continue evolving. This could be because the task itself is too easy for the network. However the deeper convolutional networks trained in CIFAR-10 showed clear early bird signals. The middle convolutional layers had a Spearman correlation of ~0.8 by round 3 of pruning, indicating a stable winning ticket structure. For the largest convnet, the highest level of sparsity (~94% of weights pruned) had a collapse of rank correlations to near-zero or even negative. At this same level of sparsity, the network's performance collapsed to chance (10% accuracy). So visualizing the Spearman correlation did help indicate when a potential winning ticket was found early and when a network's performance collapsed. Check out the "mask_similarity.html" files in the repo for the visuals (here is an [example plot](https://enerrio.github.io/lottery-ticket-hypothesis/outputs/cifar10/conv6/plots/mask_similarity.html) for conv6 experiments).

## Beyond Pruning
If there's a singular lesson to be learned from this line of research, it's that sparsity is an important part of model training. Perhaps as important as the training data and architecture.

This lesson has been applied in modern model architectures. Mixture of expert-based models (MoE) are sparse by nature: only a portion of the network is active at a time. MoE models are sparse in their activations (via routing) vs LTH which is sparse in its weights (via pruning). They share the same underlying principle: not all weights need to be active for strong performance.

The "rewinding" technique showed that *which* weights survive matters. It's not just the architecture of the sparse subnetwork that is important, but early training dynamics impart important information, especially for deep networks.

Finally, even though some weights are zeroed out with pruning, the computational cost is the same. This is because the zeroed out weights are still being multiplied in the forward pass. To save compute at inference time, you need to apply other tricks to see the benefit. These can be hardware based, like ["sparse-aware" GPUs](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) or software based like [SlimLLM](https://arxiv.org/abs/2505.22689).

## Resources
[^1]: Some pre-lottery ticket work on pruning: [Optimal brain damage](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html), [Second order derivatives for network pruning: Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html), and [Learning both weights and connections for efficient neural network](https://arxiv.org/abs/1506.02626)
[^2]: Lottery ticket hypothesis: [https://arxiv.org/abs/1803.03635](https://arxiv.org/abs/1803.03635)
[^3]: Pruning techniques: [https://datature.io/blog/a-comprehensive-guide-to-neural-network-model-pruning](https://datature.io/blog/a-comprehensive-guide-to-neural-network-model-pruning)
[^4]: Dropout: [https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
