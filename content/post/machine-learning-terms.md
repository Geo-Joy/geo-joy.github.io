---
section: post
date: "2018-03-06"
title: "Machine Learning & Deep learning Terms"
description: "Important terms to remember in the field of machine learning with deep learning."
slug: "machine-learning-and-deep-learning-terms"
tags:
 - 101
 - Machine_Learning
---

![machine-Learning-terms.jpg](/images/articles/2018/machine-Learning-terms.jpg "machine-Learning-terms.jpg")

# The Machine Learning Landscape

## 1. What is Machine Learning?
- It is the field of study that gives computers the ability to learn without being explicitly programmed.

## 2. Types of Machine Learning
- Whether or not they are trained with human supervision
    - Supervised Learning
    - Unsupervised Learning
    - Semisupervised Learning
    - Reinforcement Learning
- Whether or not they can learn incrementally on the fly
    - Online Learning
    - Batch Learning
- Whether they work by simly comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do.
    - Instance based Learning
    - Model Based Learning

## 3. Whats is a labeled training set?
- They are dataset containing the desired solution.
- A label is the thing we're predicting—the ```y``` variable in simple linear regression. The label could be the future price of wheat, the kind of animal shown in a picture, the meaning of an audio clip, or just about anything.

## 4. Supervised Learning
- In supervised learning the training data (features) fed to the algorithm includes the desired solutions called labels.

## 5 Feature?
- Features are the way we represent the data. eg: Age, height, location, words in an email etc.
- A feature is an input variable—the x variable in simple linear regression. A simple machine learning project might use a single feature, while a more sophisticated machine learning project could use millions of features.

## Feature Set
- The group of features your machine learning model trains on. For example, postal code, property size, and property condition might comprise a simple feature set for a model that predicts housing prices.

## 6. Network?
- Networks as untrained artificial neural networks (basically just raw data)

## 7. Model?
- Models as what networks become once they are trained (through exposure to data).
- A model defines the relationship between features and label. For example, a spam detection model might associate certain features strongly with "spam".

## 8. Training
- Training means creating or learning the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.
- The process of determining the ideal parameters comprising a model.

## What is the goal of training a model
- The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples.

## 9. What are the two most common supervised tasks?
- Regression
- Classification

## 12. Regression
- A regression model predicts continuous values. For example, regression models make predictions that answer questions like the following:
    - What is the value of a house in California?
    - What is the probability that a user will click on this ad?

## 11. Classification
- A classification model predicts discrete values. For example, classification models make predictions that answer questions like the following:
    - Is a given email message spam or not spam?
    - Is this an image of a dog, a cat, or a hamster?

## 12. List some of the important supervised learning algorithms.
- K-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVMs)
- Decision Tree and Random Forests
- Neural Networks

## 13. Unsupervised Learning
- Here the training data is unlabeled.

## 14. Name 4 common unsupervised tasks
- Clustering Algorithms
- Visualization Algorithms
- Dimentionality Reduction
- Anamoly Detection
- Association Rule Learning

## 15. Semi-Supervised Learning
- Some algorithms  can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. This is called semisupervised learning.
- eg: Google Photos: It recognizes that the same person A shows up in photos 1,5 and 11, while another persion B shows up inphotos 2,5 and 7.

## 16. Reinforcement Learning
- 

## Linear Regression
- It is a method for finding the straight line or hyperplane that best fits a set of points.
- A type of regression model that outputs a continuous value from a linear combination of input features.
- y = wx + b

<iframe width="560" height="315" src="https://www.youtube.com/embed/-07pO9iv23U?rel=0&amp;showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Bias
- An intercept or offset from an origin. Bias (also known as the bias term) is referred to as b or w0 in machine learning models.

## Inference
- In machine learning, often refers to the process of making predictions by applying the trained model to unlabeled examples.

## Weights
- A coefficient for a feature in a linear model, or an edge in a deep network. The goal of training a linear model is to determine the ideal weight for each feature. If a weight is 0, then its corresponding feature does not contribute to the model.

## Empirical Risk Minimization.
- In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called empirical risk minimization.

<iframe width="560" height="315" src="https://www.youtube.com/embed/jfKShxGAbok?rel=0&amp;controls=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Loss
- Loss is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater.

## L1 Loss
- Loss function based on the absolute value of the difference between the values that a model is predicting and the actual values of the labels. L1 loss is less sensitive to outliers than L2 loss.

## L2 Loss or squared loss
- It is also called as squared loss
- Its the difference between prediction and label
- (observation - prediction)^2
- This function calculates the squares of the difference between a model's predicted value for a labeled example and the actual value of the label. Due to squaring, this loss function amplifies the influence of bad predictions. That is, squared loss reacts more strongly to outliers than L1 loss.

<iframe width="560" height="315" src="https://www.youtube.com/embed/Rm2KxFaPiJg?rel=0&amp;controls=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Mean square error (MSE) | CrossEntropy 
- Is the average squared loss per example.
- MSE is calculated by dividing the squared loss by the number of examples.

![machine-Learning-terms.jpg](/images/articles/2018/MSE.jpg"machine-Learning-terms.jpg")

- x is the set of features (for example, temperature, age, and mating success) that the model uses to make predictions
- y is the example's label
- prediction(x) is a function o the weights and bias in combination with the set of features
- D is a data set containing many labeled examples which are (x,y) pairs
- N is the number of examples in D

![machine-Learning-terms.jpg](/images/articles/2018/MCEDescendingIntoMLRight.png"machine-Learning-terms.jpg")

- The eight examples on the line incur a total loss of 0. However, although only two points lay off the line, both of those points are twice as far off the line as the outlier points in the left figure. Squared loss amplifies those differences, so an offset of two incurs a loss four times greater than an offset of one.
![machine-Learning-terms.jpg](/images/articles/2018/MSP311bb3d60a26dc5e7b00002575ca91hgi8ge85.jpg"machine-Learning-terms.jpg")

## Loss function vs Cost Function
- Both are almost mean the same
- Loss function is usually a function defined on a data point, prediction and label, and measures the penalty
    - square loss used in linear regression
    - hinge loss used in SVM
    - 0/1 loss used in theoretical analysis and definition of accuracy
- Cost function is usually more general. It might be a sum of loss functions over your training set
    - Mean Squared Error
    - SVM cost function

<iframe width="560" height="315" src="https://www.youtube.com/embed/7VdGCHxGUYQ?rel=0&amp;controls=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Convergence
- Informally, often refers to a state reached during training in which training loss and validation loss change very little or not at all with each iteration after a certain number of iterations. In other words, a model reaches convergence when additional training on the current data will not improve the model. In deep learning, loss values sometimes stay constant or nearly so for many iterations before finally descending, temporarily producing a false sense of convergence.

## When do we say a model has converged?
-  We iterate until overall loss stops changing or at least changes extremely slowly. When this happens, we say that the model has converged.

## Ploting loss vs weight 
![machine-Learning-terms.jpg](/images/articles/2018/lossVSweight.png"machine-Learning-terms.jpg")

- When we plot loss vs weights of a regression problem we get the above shown graph which is shaped like a bowl/convex shape.
- Convex problems have only one minimum; that is, only one place where the slope is exactly 0. That minimum is where the loss function converges.

## Gradient Descent
- A mechanism to calculate the point of convergence(local/global minima) is called Gradient Descent.
- a gradient is a vector of partial derivatives having both direction and magnitude.
- The gradient always points in the direction of steepest increase in the loss function.
- The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.
- To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point.
- The gradient descent then repeats this process, edging ever closer to the minimum
- A technique to minimize loss by computing the gradients of loss with respect to the model's parameters, conditioned on training data. 
- Informally, gradient descent iteratively adjusts parameters, gradually finding the best combination of weights and bias to minimize loss.

## Gradient Step
- A forward and backward evaluation of one batch.

## learning rate or Step Size
- Gradient descent algorithms multiply the gradient by a scalar known as the learning rate (also sometimes called step size) to determine the next point.
- For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.
- A scalar used to train a model via gradient descent. During each iteration, the gradient descent algorithm multiplies the learning rate by the gradient. The resulting product is called the gradient step.

## Hyperparameters
- The "knobs" that you tweak during successive runs of training a model. For example, learning rate is a hyperparameter.

## Batch
- The set of examples used in one iteration (that is, one gradient update) of model training.

## Batch Size
- The number of examples in a batch. For example, the batch size of SGD is 1, while the batch size of a mini-batch is usually between 10 and 1000. Batch size is usually fixed during training and inference; however, TensorFlow does permit dynamic batch sizes.

## Stochastic gradient descent (SGD)
- A gradient descent algorithm in which the batch size is one. 
- In other words, SGD relies on a single example chosen uniformly at random from a data set to calculate an estimate of the gradient at each step.

## mini-batch
- A small, randomly selected subset of the entire batch of examples run together in a single iteration of training or inference. The batch size of a mini-batch is usually between 10 and 1,000. It is much more efficient to calculate the loss on a mini-batch than on the full training data.

## Mini-batch stochastic gradient descent (mini-batch SGD)
- It is a compromise between full-batch iteration and SGD. 
- A mini-batch is typically between 10 and 1,000 examples, chosen at random. 
- Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.
- A gradient descent algorithm that uses mini-batches. In other words, mini-batch SGD estimates the gradient based on a small subset of the training data. Vanilla SGD uses a mini-batch of size 1.

## Generalization
- Refers to your model's ability to make correct predictions on new, previously unseen data as opposed to the data used to train the model.

## Overfitting
- Creating a model that matches the training data so closely that the model fails to make correct predictions on new data.

## Validation Set
- A subset of the data set—disjunct from the training set—that you use to adjust hyperparameters.

## Feature Engineering
- The process of determining which features might be useful in training a model, and then converting raw data from log files and other sources into said features.

## Discrete Feature
- A feature with a finite set of possible values. For example, a feature whose values may only be animal, vegetable, or mineral is a discrete (or categorical) feature

<iframe width="560" height="315" src="https://www.youtube.com/embed/AePvjhyvsBo?rel=0&amp;controls=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## One-Hot Encoding
- A sparse vector in which:
    - One element is set to 1.
    - All other elements are set to 0.
-One-hot encoding is commonly used to represent strings or identifiers that have a finite set of possible values. For example, suppose a given botany data set chronicles 15,000 different species, each denoted with a unique string identifier. As part of feature engineering, you'll probably encode those string identifiers as one-hot vectors in which the vector has a size of 15,000.

## Outliers
- Values distant from most other values. In machine learning, any of the following are outliers:
    - Weights with high absolute values.
    - Predicted values relatively far away from the actual values.
    - Input data whose values are more than roughly 3 standard deviations from the mean.

## Scaling
- A commonly used practice in feature engineering to tame a feature's range of values to match the range of other features in the data set. For example, suppose that you want all floating-point features in the data set to have a range of 0 to 1. Given a particular feature's range of 0 to 500, you could scale that feature by dividing each value by 500.

## Advantages of scaling
- Helps gradient descent converge more quickly.
- Helps avoid the "NaN trap," in which one number in the model becomes a NaN.
- Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.

## Binning or Bucketing
- Converting a (usually continuous) feature into multiple binary features called buckets or bins, typically based on value range. For example, instead of representing temperature as a single continuous floating-point feature, you could chop ranges of temperatures into discrete bins. Given temperature data sensitive to a tenth of a degree, all temperatures between 0.0 and 15.0 degrees could be put into one bin, 15.1 to 30.0 degrees could be a second bin, and 30.1 to 50.0 degrees could be a third bin.

## NaN trap
- When one number in your model becomes a NaN during training, which causes many or all other numbers in your model to eventually become a NaN (e.g., when a value exceeds the floating-point precision limit during training), and—due to math operations—every other number in the model also eventually becomes a NaN.
- NaN is an abbreviation for "Not a Number."

## Synthetic Feature
- A feature that is not present among the input features, but is derived from one or more of them.

## Feature Cross 
- A feature cross is a synthetic feature formed by multiplying (crossing) two or more features. 
- Crossing combinations of features can provide predictive abilities beyond what those features can provide individually.

## Regularization
- The penalty on a model's complexity. Regularization helps prevent overfitting.
    - L1 regularization
    - L2 regularization
    - dropout regularization
    - early stopping (this is not a formal regularization method, but can effectively limit overfitting)

## Structural Risk Minimization (SRM)
- An algorithm that balances two goals:
    - The desire to build the most predictive model (for example, lowest loss).
    - The desire to keep the model as simple as possible (for example, strong regularization).
- For example, a model function that minimizes loss+regularization on the training set is a structural risk minimization algorithm.

## L1 regularization
- A type of regularization that penalizes weights in proportion to the sum of the absolute values of the weights. In models relying on sparse features, L1 regularization helps drive the weights of irrelevant or barely relevant features to exactly 0, which removes those features from the model. Contrast with L2 regularization.
- L1 regularization reduces the model size.

## L2 regularization
- A type of regularization that penalizes weights in proportion to the sum of the squares of the weights. L2 regularization helps drive outlier weights (those with high positive or low negative values) closer to 0 but not quite to 0. (Contrast with L1 regularization.) L2 regularization always improves generalization in linear models.

## Regularization Rate (Lambda)
- A scalar value, represented as lambda, specifying the relative importance of the regularization function.
- Raising the regularization rate reduces overfitting but may make the model less accurate.
- If your lambda value is too high, your model will be simple, but you run the risk of underfitting your data. Your model won't learn enough about the training data to make useful predictions.
- If your lambda value is too low, your model will be more complex, and you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.

## Early Stopping
- A method for regularization that involves ending model training before training loss finishes decreasing. In early stopping, you end model training when the loss on a validation data set starts to increase, that is, when generalization performance worsens.

##  classification
- A type of classification task that outputs one of two mutually exclusive classes. For example, a machine learning model that evaluates email messages and outputs either "spam" or "not spam" is a binary classifier.

## Logistic Regression
- A model that generates a probability for each possible discrete label value in classification problems by applying a sigmoid function to a linear prediction. Although logistic regression is often used in binary classification problems, it can also be used in multi-class classification problems (where it becomes called multi-class logistic regression or multinomial regression).

## Sigmoid Function
- A function that maps logistic or multinomial regression output (log odds) to probabilities, returning a value between 0 and 1.
- In other words, the sigmoid function converts sigma(sum of bias, weights and features) into a probability between 0 and 1.
- In some neural networks, the sigmoid function acts as the activation function.

## Log Loss
- The loss function for logistic regression is Log Loss
- The loss function for linear regression is squared loss.

## Binary Classification
- A type of classification task that outputs one of two mutually exclusive classes. For example, a machine learning model that evaluates email messages and outputs either "spam" or "not spam" is a binary classifier.

## Classification Model
- A type of machine learning model for distinguishing among two or more discrete classes. For example, a natural language processing classification model could determine whether an input sentence was in French, Spanish, or Italian.

## Classification Threshold
- A scalar-value criterion that is applied to a model's predicted score in order to separate the positive class from the negative class. Used when mapping logistic regression results to binary classification. For example, consider a logistic regression model that determines the probability of a given email message being spam. If the classification threshold is 0.9, then logistic regression values above 0.9 are classified as spam and those below 0.9 are classified as not spam.

![machine-Learning-terms.jpg](/images/articles/2018/confusion_matric_wolf.PNG"machine-Learning-terms.jpg")

## Confusion Matrix
- An NxN table that summarizes how successful a classification model's predictions were; that is, the correlation between the label and the model's classification. One axis of a confusion matrix is the label that the model predicted, and the other axis is the actual label. N represents the number of classes. In a binary classification problem, N=2.
- Confusion matrices contain sufficient information to calculate a variety of performance metrics, including precision and recall.

 | Tumor (predicted) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Non-Tumor (predicted)
 --- | --- | ---
Tumor (actual)&nbsp;&nbsp;&nbsp;&nbsp;| 18 |&nbsp;&nbsp;&nbsp;&nbsp;1
Non-Tumor (actual)&nbsp;&nbsp;&nbsp;&nbsp;| 6 |&nbsp;&nbsp;&nbsp;&nbsp;452
<br/>

- The preceding confusion matrix shows that of the 19 samples that actually had tumors, the model correctly classified 18 as having tumors (18 true positives), and incorrectly classified 1 as not having a tumor (1 false negative). Similarly, of 458 samples that actually did not have tumors, 452 were correctly classified (452 true negatives) and 6 were incorrectly classified (6 false positives).

## Class-Imbalanced Data Set
- A binary classification problem in which the labels for the two classes have significantly different frequencies. For example, a disease data set in which 0.0001 of examples have positive labels and 0.9999 have negative labels is a class-imbalanced problem, but a football game predictor in which 0.51 of examples label one team winning and 0.49 label the other team winning is not a class-imbalanced problem.

## Accuracy
- The fraction of predictions that a classification model got right. In multi-class classification, accuracy is defined as follows:
![machine-Learning-terms.jpg](/images/articles/2018/MSP01bb3i5d249207aed000042d82gi907c3h8d8.jpg"machine-Learning-terms.jpg")
- In binary classification, accuracy has the following definition:
![machine-Learning-terms.jpg](/images/articles/2018/MSP11bb3i5d249207aed00002cc94d911df050b0.jpg"machine-Learning-terms.jpg")

## Precision
- A metric for classification models.
- Precision identifies the frequency with which a model was correct when predicting the positive class.
![machine-Learning-terms.jpg](/images/articles/2018/MSP41bb3i5d249207aed00006144643b463f0ic3.jpg"machine-Learning-terms.jpg")

## Recall
- Out of all the possible positive labels, how many did the model correctly identify?
![machine-Learning-terms.jpg](/images/articles/2018/MSP31bb3i5d249207aed000034d4c5073776c1f6.jpg"machine-Learning-terms.jpg")

## ROC (receiver operating characteristic) Curve
A curve of true positive rate vs. false positive rate at different classification thresholds.

## True Positive Rate (TP rate)
- Synonym for recall.
- True positive rate is the y-axis in an ROC curve.
![machine-Learning-terms.jpg](/images/articles/2018/MSP511bb3d60a26dc5e7b00003b9eb0cd8id767if.jpg"machine-Learning-terms.jpg")

## False Positive Rate (FP rate)
- The x-axis in an ROC curve. The FP rate is defined as follows
![machine-Learning-terms.jpg](/images/articles/2018/MSP521bb3d60a26dc5e7b000036e5080df65ie42a.jpg"machine-Learning-terms.jpg")

## Prediction Bias
- A value indicating how far apart the average of predictions is from the average of labels in the data set.

## Neural Network
A model that, taking inspiration from the brain, is composed of layers (at least one of which is hidden) consisting of simple connected units or neurons followed by nonlinearities.

## Neuron
A node in a neural network, typically taking in multiple input values and generating one output value. The neuron calculates the output value by applying an activation function (nonlinear transformation) to a weighted sum of input values.

## Rectified Linear Unit (ReLU)
- An activation function with the following rules:
    - If input is negative or zero, output is 0.
    - If input is positive, output is equal to input.

## Sigmoid Function
- A function that maps logistic or multinomial regression output (log odds) to probabilities, returning a value between 0 and 1.

## Hidden Layer
A synthetic layer in a neural network between the input layer (that is, the features) and the output layer (the prediction). A neural network contains one or more hidden layers.

## Backpropagation
The primary algorithm for performing gradient descent on neural networks. First, the output values of each node are calculated (and cached) in a forward pass. Then, the partial derivative of the error with respect to each parameter is calculated in a backward pass through the graph.

## Vanishing Gradients
- The gradients for the lower layers (closer to the input) can become very small. In deep networks, computing these gradients can involve taking the product of many small terms.
- When the gradients vanish toward 0 for the lower layers, these layers train very slowly, or not at all.
- The ReLU activation function can help prevent vanishing gradients.

## Exploding Gradients
- If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms. In this case you can have exploding gradients: gradients that get too large to converge.
- Batch normalization can help prevent exploding gradients, as can lowering the learning rate.

## Dead ReLU Units
- Once the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck. It outputs 0 activation, contributing nothing to the network's output, and gradients can no longer flow through it during backpropagation.
- Lowering the learning rate can help keep ReLU units from dying.

## Dropout Regularization
- A form of regularization useful in training neural networks. Dropout regularization works by removing a random selection of a fixed number of the units in a network layer for a single gradient step. The more units dropped out, the stronger the regularization.

## Multi-Class Classification
- Classification problems that distinguish among more than two classes. For example, there are approximately 128 species of maple trees, so a model that categorized maple tree species would be multi-class. Conversely, a model that divided emails into only two categories (spam and not spam) would be a binary classification model.

## Multi-Class Classification
Classification problems that distinguish among more than two classes. For example, there are approximately 128 species of maple trees, so a model that categorized maple tree species would be multi-class. Conversely, a model that divided emails into only two categories (spam and not spam) would be a binary classification model.

## Softmax
- A function that provides probabilities for each possible class in a multi-class classification model. The probabilities add up to exactly 1.0. For example, softmax might determine that the probability of a particular image being a dog at 0.9, a cat at 0.08, and a horse at 0.02. (Also called full softmax.)

## Candidate Sampling
- A training-time optimization in which a probability is calculated for all the positive labels, using, for example, softmax, but only for a random sample of negative labels. For example, if we have an example labeled beagle and dog candidate sampling computes the predicted probabilities and corresponding loss terms for the beagle and dog class outputs in addition to a random subset of the remaining classes (cat, lollipop, fence). The idea is that the negative classes can learn from less frequent negative reinforcement as long as positive classes always get proper positive reinforcement, and this is indeed observed empirically. The motivation for candidate sampling is a computational efficiency win from not computing predictions for all negatives.

## One Label vs. Many Labels
- Softmax assumes that each example is a member of exactly one class. Some examples, however, can simultaneously be a member of multiple classes. For such examples:
    - You may not use Softmax.
    - You must rely on multiple logistic regressions.
- For example, suppose your examples are images containing exactly one item—a piece of fruit. Softmax can determine the likelihood of that one item being a pear, an orange, an apple, and so on. If your examples are images containing all sorts of things—bowls of different kinds of fruit—then you'll have to use multiple logistic regressions instead.

## Sparse Feature
- Feature vector whose values are predominately zero or empty. For example, a vector containing a single 1 value and a million 0 values is sparse. As another example, words in a search query could also be a sparse feature—there are many possible words in a given language, but only a few of them occur in a given query.

## Embeddings
- A categorical feature represented as a continuous-valued feature.
- Typically, an embedding is a translation of a high-dimensional vector into a low-dimensional space. For example, you can represent the words in an English sentence in either of the following two ways:
    - As a million-element (high-dimensional) sparse vector in which all elements are integers. Each cell in the vector represents a separate English word; the value in a cell represents the number of times that word appears in a sentence. Since a single English sentence is unlikely to contain more than 50 words, nearly every cell in the vector will contain a 0. The few cells that aren't 0 will contain a low integer (usually 1) representing the number of times that word appeared in the sentence.
    - As a several-hundred-element (low-dimensional) dense vector in which each element holds a floating-point value between 0 and 1. This is an embedding.

## Standard Dimensionality Reduction Techniques

## Word2vec

## static model
-A model that is trained offline.

## dynamic model
- A model that is trained online in a continuously updating fashion. That is, data is continuously entering the model.

## online inference
- Generating predictions on demand. Contrast with offline inference.
    - Pro: Can make a prediction on any new item as it comes in — great for long tail.
    - Con: Compute intensive, latency sensitive—may limit model complexity.
    - Con: Monitoring needs are more intensive.

## offline inference
- Generating a group of predictions, storing those predictions, and then retrieving those predictions on demand. Contrast with online inference.
- meaning that you make all possible predictions in a batch, using a MapReduce or something similar. You then write the predictions to an SSTable or Bigtable, and then feed these to a cache/lookup table.
    - Pro: Don’t need to worry much about cost of inference.
    - Pro: Can likely use batch quota or some giant MapReduce.
    - Pro: Can do post-verification of predictions before pushing.
    - Con: Can only predict things we know about — bad for long tail.
    - Con: Update latency is likely measured in hours or days.

##


