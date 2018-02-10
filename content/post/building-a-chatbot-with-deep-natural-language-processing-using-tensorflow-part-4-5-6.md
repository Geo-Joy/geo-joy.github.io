---
section: post
date: "2018-02-11"
title: "Building a Chatbot with Deep Natural Language Processing using Tensorflow [PART-4-5-6]"
description: "Using tensorflow to build a chatbot demonstrating the power of Deep Natural Language Processing"
slug: "building-a-chatbot-with-deep-natural-language-processing-using-tensorflow-part-4-5-6"
tags:
 - Chatbot
 - NLP
 - TensorFlow
 - Project
 - Deep Learning
 - Artificial Intelligence
 - Tutorial 
---
<!-- # How a software developer re-focused his life to learn artificial intelligence -->

![Chat Bot](/images/articles/2018/chat-bot-intro-image.jpg "Chatbot")

#### This tutorial is divided into the following parts
1. Installing and configuring Anaconda
2. Getting the Dataset
3. Data Preprocessing
4. Building the Seq2Seq model
5. Training the Seq2Seq model
6. Testing the Seq2Seq model
7. Other Implementation

# Lets Start

## PART-4. Building the Seq2Seq model

Here we will start using TendorFlow library to do all our deep learning stuffs.

In TensorFlow we first write out our logic and later run using a session. So we need to create some variable holder which will accept variables when we run it in session. For this we use TensorFlow Placeholder. It is basic an advanced datastructure that can contain tensors and also other features.

```html
Definition : placeholder(dtype, shape=None, name=None)
```

```py
# creating placeholders for the inputs and the targets

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name='inputs') # dtype of sorted_clean_questions is integer, shape is a 2 dimentional array with padding
    targets = tf.placeholder(tf.int32, [None,None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate') #learning rate - no matrix as its a hyper parameter
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #keep probe used to control the drop off rate usually 20% neurons are deactivated
    return inputs, targets, lr, keep_prob
```