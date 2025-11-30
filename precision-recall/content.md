# Precision & Recall

Up until recently, I **always** had to look up the equations for precision and recall because I could never remember which one was which. I understood what they meant and, more importantly, when to use these metrics to measure the performance of a machine learning model. But because they go hand in hand, I always found it challenging tell them apart - like trying to distinguish identical twins. I know at least a couple colleagues who have had this experience in the past, plus I see a lot of articles about it, so I suspect it's not just me. With that said, here's one more blog post on the subject.

Let's back up a bit and talk about what precision and recall are in the first place and why you would use them. If you're already familiar with both and just want tips on how to keep them straight in your head, feel free to skip the next section.

## What is Precision & Recall?
Precision and recall are metrics commonly used in supervised learning tasks to measure a model's performance. Unlike accuracy, which measures the overall fraction of correctly classified samples, precision and recall measure how well the model performs given the distribution of classes. This becomes critical when you're working with an imbalanced dataset. 

Imagine you have a dataset of 1,000 samples: 900 labelled positive, 100 labelled negative. Let's say after training your model, it completely fails to learn anything meaningful about the data and instead just always classifies any input as positive. Your accuracy for that training dataset will be 900 out of 1,000 = 90%. On paper, a 90% accuracy score seems great, but in this case, it's hiding the fact that your model is biased to label everything as positive. In practice, this can lead you to overestimate your model’s usefulness.

Let's define accuracy, precision, and recall in mathematical terms along with some terminology:

> TP = True positive = The model correctly labelled a positive sample.  
TN ‎ =  True negative = The model correctly labelled a negative sample.  
FP = False positive = The model incorrectly labelled a sample as positive. The actual label is negative.  
FN = False negative = The model incorrectly labelled a sample as negative. The actual label is positive.

<div align="center">
  <img src="https://i.ibb.co/fdLF51dT/precision-Recall.png" alt="Precision & Recall formula diagram">
</div>

<br>

$$
\mathrm{Accuracy} \;=\; \frac{TP + TN}{TP + TN + FP + FN}
$$
<br>

$$
\mathrm{Precision} \;=\; \frac{TP}{TP + FP}
$$
<br>

$$
\mathrm{Recall} \;=\; \frac{TP}{TP + FN}
$$

- **Precision** = 900 / (900 + 100) = 90% (it misses all 100 negatives)  
- **Recall**    = 900 / (900 + 0) = 100% (it never makes a false negative)

It's also helpful to visualize the results in a confusion matrix, which really drives home the failings of this model:

<div align="center">
  <img src="https://i.ibb.co/vvKyysbN/precision-Recall-Example.png" alt="Confusion matrix example">
</div>

## Just Tell Me The Answer
So how do I keep them straight? The mnemonic that has helped me the most is the following:
> Precision starts with the letter P

That's it.

<div style="text-align: center; width: 100%;">
  <img 
    src="https://media1.tenor.com/m/_BiwWBWhYucAAAAd/what-huh.gif" 
    style="max-width: 100%; height: auto;"
  />
</div>

OK, I'm cheating a little. I actually have another piece of information that repeated exposure has beaten into my brain:
* The equation for both metrics has the form $\frac{TP}{TP + X}$ where X is either FP or FN.

We covered the equations in the previous section. As long as I remember that precision starts with the letter P, then I can remember to replace that placeholder in the denominator with FP, which also contains the letter P. From there, it's a short step to jump from equation to intuition: Precision is saying out of all the **predicted** positives, how many did the model correctly classify?

And recall is the inverse of that: Out of all the **ground truth** positives, how many did the model correctly classify?

After this sets in, you can hopefully remove the precision and recall Wikipedia page [^1] from your bookmarks 😉

## More than Metrics
While mnemonics are helpful for exams or personal knowledge, you can always just look up the definitions online. In an industry setting, it's far more important to understand when to take these metrics out of your toolbox and use them than to have them memorized. However, memorization may come with experience anyway.

When you are training your own model or hearing a pitch about some tool that boasts a high accuracy, remember to ask yourself in what setting did it achieve that high score. Is your dataset imbalanced? What's the impact of a misclassification? Precision and recall can help you get a better idea of what your model's strong and weak points are and can influence future efforts to improve it. Precision and recall aren't as tidy as a single accuracy number, but they paint a much clearer picture [^2].

## PostScript: Intuition for Precision and Recall
Besides knowing the meaning, it's also helpful to have some intuition for interpreting the different scenarios you might come across with these metrics.

A high precision means you rarely make false positive mistakes. While a high recall means you catch almost every real positive.

Here's a table describing a few common scenarios:

| Precision | Recall | What's Happening                                                                                         |
|-----------|--------|----------------------------------------------------------------------------------------------------------|
| High      | High   | Ideal: very few errors of either kind.                                                                   |
| High      | Low    | When the model predicts positives, it's often right. But it misses many ground truth positives.          |
| Low       | High   | The model catches most real positives, but it casts too wide a net. It pulls in a lot of false positives. |
| Low       | Low    | Predicted positives are often incorrect.                                                                 |

Oftentimes you’ll make a trade-off between precision and recall because some models will spit out logits[^3] and you set a threshold: if the logit is above the threshold, it's marked positive; if it's below, it's considered negative. The trade-off comes from setting the threshold. A higher threshold means fewer predicted positives, which increases precision but decreases recall. A lower threshold does the opposite: more predicted positives, higher recall, and lower precision. You can sweep over threshold values and plot the precision and recall scores at each threshold. This is called the precision-recall curve, and is a useful visualization.

Another super useful visualization is a confusion matrix which looks like the earlier grid images except there are values in each square representing how many true positives, false positives, true negatives, and false negatives there are. This visual becomes especially useful when you are dealing with multiple classes.

## Thanks for reading! 
If you found this helpful or have your own tricks for remembering precision and recall, I'd love to hear them. Reach out to me on [BlueSky](https://bsky.app/profile/enerrio.bsky.social) or [LinkedIn](http://linkedin.com/in/aaronmar/).

## Links
[^1]: Wikipedia page: https://en.wikipedia.org/wiki/Precision_and_recall
[^2]: Side note: If you still want a single number you can always use [F1](https://en.wikipedia.org/wiki/F-score): the harmonic mean of precision and recall.
[^3]: Logits: https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean
