---
section: post
date: "2018-01-27"
title: "Fraud Detection Using Self-Organizing Maps"
description: " "
slug: "fraud-detection-using-self-organizing-maps"
draft: true
tags:
 - Project
 - SOM
---

![TensorFlow Basics](/images/articles/2018/TensorFlow-banner.jpg "TensorFlow-banner.jpg")

# 1. What is Tensor
 - Just a fancy word for an n-dimentional matrix
 - Or an advanced array
 - Holds only single type
 - They are the standard way of representing data in tensorflow.

# 2. Tensor Data Types

| python type&nbsp;&nbsp;&nbsp; | Description |
| ---|--- |
| tf.float32 | 32 bits floating point |
| tf.float64 | 64 bits floating point |
| tf.int8 | 8 bits signed integer |
| tf.int16 | 16 bits signed integer |
| tf.int32 | 32 bits signed integer |
| tf.int64 | 64 bits signed integer |
| tf.uint8 | 8 bits signed integer |
| tf.string | Variable length byte array |
| tf.bool | Boolean |

- DataTypes need not be explicitly mentioned it get automatically assigned by the tensorflor library.
- Only mention it if you need to save few memory of tensor

# 3. What is TensorFlow
 - Open Source library released by Google in 2015
 - It works by first defining and describing a model and later run in a session

# 4. What are constant
 - It takes no inputs and cannot change the value stored in it

# 5. What are placeholders
 - A placeholder is a promise to provide a value later.
 - Use the **feed_dict** parameter in **tf.session.run()** to set the placeholder tensor.
 - They are initially empty and are used to feed in the actual raining examples.
 - Need to declare the required type as **tf.float32** and shape as optional.

# 6. What are variables
 - They need to be initialized
 - Values that keeps on changing
 - They are in memory buffers
 - use **tf.global_variables_initializer** to initialize the variables before running the session.

# 7.TensorFlow Graphs
 - Graphs are sets of connected nodes(vertices)
 - THe connections are referred to as edges.
 - In TensorFlow each node is an operation with possible inputs that can supply some output.
 - Its like we construct a graph and execute it.