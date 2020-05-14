---
section: post
date: "2018-01-28"
title: "Intro to Recurrent Neural Network (RNN)"
description: "Recurrent neural networks are able to learn from sequences of data. In this tutorial, we'll learn the concepts behind recurrent networks."
slug: "intro-to-recurrent-neural-network-rnn"
tags:
 - rnn
 - deeplearning
---

![rnn-lstm-intro.jpg](/images/articles/2018/RNN/rnn-lstm-intro.jpg "rnn-lstm-intro.jpg")
image borrowed from <a href="udacity.com">UDACITY</a>

Recurrent Neural Network is one of the advanced algorithms that existed in the wold of ```supervised deep learning```.

# Prelude
In ANN and CNN we use neural network that takes few inputs, propagates it forward through hidden layers to the output layer. This is termed as a ``` FEED FORWARD NETWORK ```. In this network there is no sence of order in the inputs.

OK now lets bring some order to the network.

<br/>

Lets take an example data and go forward.

```
"STEEP"
```

![feedforward_ex1.png](/images/articles/2018/RNN/feedforward_ex1.png "feedforward_ex1.png")
<center>image borrowed from <a href="udacity.com">UDACITY</a></center>

<br/>
Here ``x`` is the input layer ``h`` the hidden layer and ``y`` the output layer.

Our goal is to make the network predict the next letter to the sequence.

- First we feed ``s`` to the network and our desired output is ``t``.
- Then we feed ``t`` to the network and our desired output is ``e``.
- Next we feed the previous ``e`` to the network and our desired output should ``e`` or ``p``.
- But this feed forward network does not have the information about the which character to predict next as it does not know about the previous letters ``s`` ``t`` ``e``.

To include this information in the network we just need to add the previous hidden layer output to the hidden layer in the next step to carry forward the information about the sequence.

This architecture is know as Recurrent Neural Network or RNN

![rnn-representation.png](/images/articles/2018/RNN/rnn-representation.png "rnn-representation.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

The above image is the standard representation of RNN

![rnn-representation.png](/images/articles/2018/RNN/rnn-representation-time.png "rnn-representation.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

The above image shows RNN expanded in TIME.

# Examples of RNNs

## One to Many Relationship

![rnn-one-many.png](/images/articles/2018/RNN/rnn-one-many.png "rnn-one-many.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

![rnn-one-many.png](/images/articles/2018/RNN/rnn-one-many-example.png "rnn-one-many.png")
<center>image borrowed from <a href="karpathy.github.io">karpathy.github.io</a></center>

Here there is only one input and multiple outputs. For example the image is the input to tht CNN network and then into a RNN then the computer will come up with the word describing the image.

![rnn-one-many.png](/images/articles/2018/RNN/rnn-many-one.png "rnn-one-many.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

![rnn-one-many.png](/images/articles/2018/RNN/rnn-many-one-example.png "rnn-one-many.png")
<center>image borrowed from <a href="dev.havenondemand.com">dev.havenondemand.com</a></center>

Mostly for sentiment analysis from lot of texts like comments etc.

![rnn-one-many.png](/images/articles/2018/RNN/rnn-many-many.png "rnn-one-many.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

![rnn-one-many.png](/images/articles/2018/RNN/rnn-many-many-example.png "rnn-one-many.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

Can be used in a translator system. Having reference to genders and tense to reframe the translated text.

![rnn-one-many.png](/images/articles/2018/RNN/rnn-many-many2.png "rnn-one-many.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

![rnn-one-many.png](/images/articles/2018/RNN/rnn-many-many2-example.png "rnn-one-many.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

For subtitling movies. Get the information from previous frame to get the context to subtitle other scenes.

# Vanishing Gradient

This problem was first discovered by Seph Hochreiter in 1991 later by Joshua Benjio in 1994.

In Simple words:

Previously we saw that in the feedback loop through time, we pass in weights to and fro to adjust the cost function as to minimize the gradients. Here randomly assigned weights may be large or small. During forward and backward propagation we are multiplying these weights together. Multipling large weights with each other results in another larger valued weight and multipling smaller weights results in a small value. This make the gradients going through the network really small and vanishing and large causing it to explode. This also causes imbalance in the network where few nodes are trained properly and others very low. Also since these trained information is carried to other nodes over time, the heavily trained nodes are indirectly trained by the information from the less trained node. Which causes erroneous results at the end.

This is termed as vanishing and exploding gradients or simply Vanishing Gradients.

# Solutions for Vanishing Gradient

For Exploding gradient

- Truncated Backpropagation: Here we stop the backpropagation after certain point. (Its not optimal as we are not updating all the weights)
- Penalties: Gradients been penalized and manually reduced
- Gradient Clipping: Max value for gradient

For Vanishing Gradient

- Weight Initialization: Be smart enought to initial weights to minimize the vanishing gradient problem
- Echo State Networks They some how manage to solve
- Long Short-Term Memory (LSTM)

# LSTMs and its Varients

One of the best place to understand LSTM is to read the blogpost at http://colah.github.io/posts/2015-08-Understanding-LSTMs/

For My Reference:
![LSTM3-chain.png](/images/articles/2018/RNN/LSTM3-chain.png "LSTM3-chain.png")
<center>image borrowed from <a href="http://colah.github.io">colah.github.io</a></center>

## GRU - Gated Recurrent Network
![LSTM3-var-GRU.png](/images/articles/2018/RNN/LSTM3-var-GRU.png "LSTM3-var-GRU.png")
<center>image borrowed from <a href="http://colah.github.io">colah.github.io</a></center>

# Attentions

The idea is to let every step of an RNN pick information to look at from some larger collection of information. For example, if you are using an RNN to create a caption describing an image, it might pick a part of the image to look at for every word it outputs.






