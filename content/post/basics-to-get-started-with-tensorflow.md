---
section: post
date: "2018-02-10"
title: "Basics to get started with Tensorflow"
description: "Things to learn before starting up with Tensorflow"
slug: "basics-to-get-started-with-tensorflow"
draft: true
tags:
 - 101
 - TensorFlow
---

![TensorFlow Basics](/images/articles/2018/TensorFlow-banner.jpg "TensorFlow-banner.jpg")

# 1. What is Tensor
 - An n-dimentional matrix
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

# 6. What are variables
 - Values that keeps on changing
 - They are in memory buffers