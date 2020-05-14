---
section: post
date: "2018-01-21"
title: "Matrix Math and Numpy Refresher"
description: "This tutorial provides a short refresher on what is needed to know for doing deeplearning, along with some guidance for using the NumPy library to work efficiently with matrices in Python."
slug: "matrix-math-and-numpy-refresher"
tags:
 - 101
 - programming
 - numpy
 - mathematics
---

![matrix-math.jpg](/images/articles/2018/matrix-math.jpg "matrix-math.jpg")
image borrowed from <a href="udacity.com">UDACITY</a>

# Data Dimentions
We fist need to understand how we represent the data specifically the shape the data can have.

Eg: we can have a number representing a person height or a list of numbers for height, weight, age etc. May be a display picture represented as grid having rows and columns of each individual pixels and each pixels having its color value for RED, BLUE, GREEN.

We describe this data in terms of number of dimentions.

First we have the smallest simplest shape called ```SCALARS``` (single value)
eg: 1, -5, 8.9 etc

It is said that scalars have zero dimentions.

So from the previous example of the person the height, weight and age are all scalar.

<br/>

Then there are lists of values called ``VECTORS``.

They are of two types ``row vector`` and ``column vector``

```py
# row vector
[1,2,3]

# column vector
[1
2
3]
```

**Vectors** are said to have 1 dimention called length.

So from the previous example we have store the persons height, weight and age as Vector [height, weight, age]

<br>

Then we have ``MATRICS``. Its a 2-dimentional grid of value.

```py
[[1,2,3]
[4,5,6]]
```

The above code snippet shows a 2x3( two cross three ) matrix ie. 2 rows and 3 columns.

<br/>

Finally there are tensors named for a collection of n-dimentional values.

![tensor-data-shape.png](/images/articles/2018/tensor-data-shape.png "tensor-data-shape.png")
image borrowed from <a href="udacity.com">UDACITY</a>

<br/>

Mostly everyone uses the terms SCALAR and MATRICES for all tutorials.
VECTORS are hereby considered are a MATRIX having either of its row or column (dimentions) as size 1.

<br/>

To get values from a matrix we use indexes. Indexes start from a11, a12, a21 etc. <br/>
``But in programming most languages have indexes starting from 0. So we have indexes as a00, a01, a10 etc``

![matrix-indexes.jpg](/images/articles/2018/matrix-indexes.jpg "matrix-indexes.jpg")
image borrowed from <a href="udacity.com">UDACITY</a> and modified by me :)

---
<br/>

# Numpy and its Data in Python

Python is convenient, but it can also be slow. However, it does allow us to access libraries that execute faster code written in languages like C. NumPy is one such library: it provides fast alternatives to math operations in Python and is designed to work efficiently with groups of numbers - like matrices.

## Data Types and Shapes

The most common way to work with numbers in NumPy is through ndarray objects.

```
https://docs.scipy.org/doc/numpy/reference/arrays.html
```

<br/>
### SCALARS in Numpy
Python language has basic data types are int, float, boolean, string. But Numpy, since it is written in C language, has additional datatypes like uint8, int8, uint16, int16 and so on.

These datatypes are important because every object we make (vectors, matrices, tensors) eventually stores scalars.

```Do note that every item in an array must have the same data type.```

```py
import numpy as np
```

```py
# to create a numpy array that stores a scalar
scalar_array = np.array(8)
```

To get the shape of this array we use the arribute ``shape``.

```py
scalar_array.shape
```

Here we get the output as ``()``. This means that it has zero dimension.

<br/>
### VECTORS in Numpy

Vectors are created by passing a python list [1,2,3] to the numpy array.

```py
vector_array = np.array([1,2,3])
```

to check its shape
```py
vector_array.shape
```

The above snippet returns ``(3,)``. This is a tuple having the dimension of 1 and a comma suffixed.

```
https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences
```

We can access an element within a vector as

```py
x = vector_array[1]
# which gives x as 1

# to get all elements from second element we use
x = vector_array[1:]
```

<br/>
### MATRICES in Numpy

To create a matrix using numpy array we just need to pass lists of lists, where each list represents a row. So to create a 3x3 matrix we do as

```py
matrix_33 = np.array([1,2,3], [4,5,6], [7,8,9])
```

checking the shape we get tuple ``(3,3)`` indicating it has 2 dimentions, each length of 3.

<br/>
### TENSORS in Numpy

Same as matrices, here tensors can have n-dimensions.

```py
tensor = np.array([[[[1],[2]],[[3],[4]],[[5],[6]]],[[[7],[8]],[[9],[10]],[[11],[12]]],[[[13],[14]],[[15],[16]],[[17],[17]]]])
```

This returns ``(3, 3, 2, 1)``. Which says ``4`` dimensions and each number represents the ``length of each lists`` in tensor.

<br/>
### Changing shape of a matrix

Sometimes we'll need to change the shape of our data without actually changing its contents. Eg: you may have a vector, which is one-dimensional, but need a matrix, which is two-dimensional.

```py
v = np.array([1,2,3,4])

# v.shape returns (4,)

# to change shape we use the reshape function 
x = v.reshape(1,4)

# x.shape returns (1,4)
```

Experienced users do it the following way.

```py
x = v[None, :]

# x.shape returns (1,4)
```

<br/>
### Element-wise Operator

Suppose we wanted to add a value to every element in the list

```py
values = [1,2,3,4,5]
values = np.array(values) + 5

# value gives [6,7,8,9,10]
```

Similarly numpy has function for addition, subtraction, multiplication, division etc.

```py
x = np.multiply(some_array, 5)
x = some_array * 5
```

<br/>
### Element-wise Matrix Operation

The same functions and operators that work with scalars and matrices also work with other dimensions. We just need to make sure that the items we perform the operation on have compatible shapes.


```py
a = np.array([[1,3],[5,7]])
a
# displays the following result:
# array([[1, 3],
#        [5, 7]])

b = np.array([[2,4],[6,8]])
b
# displays the following result:
# array([[2, 4],
#        [6, 8]])

a + b
# displays the following result
#      array([[ 3,  7],
#             [11, 15]])

```

And if we try working with incompatible shapes, we'd get an error:

```py
a = np.array([[1,3],[5,7]])
a
# displays the following result:
# array([[1, 3],
#        [5, 7]])
c = np.array([[2,3,6],[4,5,9],[1,8,7]])
c
# displays the following result:
# array([[2, 3, 6],
#        [4, 5, 9],
#        [1, 8, 7]])

a.shape
# displays the following result:
#  (2, 2)

c.shape
# displays the following result:
#  (3, 3)

a + c
# displays the following error:
# ValueError: operands could not be broadcast together with shapes (2,2) (3,3)
```

### Important Reminders About Matrix Multiplication
- The **number** of **columns** in the **left** matrix **must equal** the **number** of **rows** in the **right** matrix.
- The **answer** matrix **always has** the **same number** of **rows** as the **left** matrix and the **same number** of **columns** as the **right** matrix.
- **Order matters**. Multiplying **A•B** is not the same as multiplying **B•A**.
- Data in the **left** matrix **should be** arranged as **rows**., while data in the **right** matrix **should be** arranged as **columns**.

If we keep these four points in mind, you should always be able to figure out how to properly arrange your matrix multiplications when building a neural network.

### NumPy Matrix Multiplication

It's important to know that NumPy supports several types of matrix multiplication.

To find the matrix product, you use NumPy's matmul function.

```py
a = np.array([[1,2,3,4],[5,6,7,8]])
a
# displays the following result:
# array([[1, 2, 3, 4],
#        [5, 6, 7, 8]])
a.shape
# displays the following result:
# (2, 4)

b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
b
# displays the following result:
# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])
b.shape
# displays the following result:
# (4, 3)

c = np.matmul(a, b)
c
# displays the following result:
# array([[ 70,  80,  90],
#        [158, 184, 210]])
c.shape
# displays the following result:
# (2, 3)
```

We may sometimes see NumPy's dot function in places where you would expect a matmul. It turns out that the results of dot and matmul are the same if the matrices are two dimensional.

```py
a = np.array([[1,2],[3,4]])
a
# displays the following result:
# array([[1, 2],
#        [3, 4]])

np.dot(a,a)
# displays the following result:
# array([[ 7, 10],
#        [15, 22]])

a.dot(a)  # you can call `dot` directly on the `ndarray`
# displays the following result:
# array([[ 7, 10],
#        [15, 22]])

np.matmul(a,a)
# array([[ 7, 10],
#        [15, 22]])
```


### Transpose

This is used to convert a row to column and viceversa. Do note that numpy transpose simply changes the way it indexes the original matrix. It does not create a new matrix.

```py
m = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
m
# displays the following result:
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

m.T
# displays the following result:
# array([[ 1,  5,  9],
#        [ 2,  6, 10],
#        [ 3,  7, 11],
#        [ 4,  8, 12]])
```




