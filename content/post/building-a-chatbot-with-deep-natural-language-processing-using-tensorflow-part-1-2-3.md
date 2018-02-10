---
section: post
date: "2018-02-10"
title: "Building a Chatbot with Deep Natural Language Processing using Tensorflow [PART-1-2-3]"
description: "Using tensorflow to build a chatbot demonstrating the power of Deep Natural Language Processing"
slug: "building-a-chatbot-with-deep-natural-language-processing-using-tensorflow-part-1-2-3"
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

## PART-1. Installing and configuring Anacond

### 1a. Installing Anaconda

![Anaconda Logo](/images/articles/2018/anaconda-logo-dark.png "Anaconda IDE")

[From anaconda.com/what-is-anaconda/](https://www.anaconda.com/what-is-anaconda/)

> With over 6 million users, the open source Anaconda Distribution is the easiest way to do Python data science and machine learning. It includes hundreds of popular > data science packages and the conda package and virtual environment manager for Windows, Linux, and MacOS. Conda makes it quick and easy to install, run, and upgrade complex data science and machine learning environments like Scikit-learn, TensorFlow, and SciPy. Anaconda Distribution is the foundation of millions of data science projects as well as Amazon Web Services' Machine Learning AMIs and Anaconda for Microsoft on Azure and Windows.

[Download for latest python version](https://www.anaconda.com/download/)
 

### 1b. Configuring Anaconda & Creating Virtual Environment

> A virtual environment is a named, isolated, working copy of Python that that maintains its own files, directories, and paths so that you can work with specific versions of libraries or Python itself without affecting other Python projects. Virtual environmets make it easy to cleanly separate different projects and avoid problems with different dependencies and version requiremetns across components. The conda command is the preferred interface for managing intstallations and virtual environments with the Anaconda Python distribution. If you have a vanilla Python installation or other Python distribution see virtualenv

I will name the virtual environment as **chatbot**

Open terminal / Command Prompt and type.

```py
conda create -n chatbot python=3.5 anaconda
```

After finishing the download we need to activate our newly created environment. To do this type

For mac /linux users

```py
source activate chatbot
```

For windows users (Powershell users - try to activate using Command prompt. )

```py
activate chatbot
```
![Activating source](/images/articles/2018/activate-chatbot.png "Activating source")

### 1c. Installing TensorFlow

Now after activating our virtual environment named **chatbot**. We now have to install tensorflow library.
I will be using v1.0.0 of tensorflow for this tutorial.

```py
pip install tensorflow==1.0.0
```

![Installing Tensorflow](/images/articles/2018/activate-chatbot-tensorflow.png "Installing Tensorflow")

### 1d. Finalizing setup.

Its time to open our installed anaconda program (Anaconda Navigator).
After opening Anaconda Navigator, we need to switch the application to use the virtual environment that we have created.
To do this use the dropdown menu on top left corner - near to **Application on** and select **chatbot** from the list.

![Switch Env](/images/articles/2018/anaconda-switch-env.png "Switch Env")

<br/><br/>
Now select "SPYDER" to start our development IDE.

> Spyder is a powerful interactive development environment for the Python language with advanced editing, interactive testing, debugging and introspection features. Additionally, Spyder is a numerical computing environment thanks to the support of IPython and popular Python libraries such as NumPy, SciPy, or matplotlib.

![Spyder](/images/articles/2018/anaconda-spyder.png "Spyder")

In FileExplorer select a working directory for this project.

![spyder-file-explorer](/images/articles/2018/spyder-file-explorer.png "spyder-file-explorer")

---
<br/><br/>

## PART-2. Getting the Dataset

Download the dataset from - [cornell_movie_dialogs_corpus.zip](http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip)

This corpus contains a large metadata-rich collection of fictional conversations extracted from raw movie scripts:

- 220,579 conversational exchanges between 10,292 pairs of movie characters
- involves 9,035 characters from 617 movies
- in total 304,713 utterances
- movie metadata included:
    - genres
    - release year
    - IMDB rating
    - number of IMDB votes
    - IMDB rating
- character metadata included:
    - gender (for 3,774 characters)
    - position on movie credits (3,321 characters)
- see README.txt (included) for details

We are interested only in two files from the downloaded zip.

1. movie_conversations.txt
    -  each row has one single **conversations** between characters. <br/>u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
2. movie_lines.txt
    - contains some extract from movies. <br/> L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!<br/>L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!
<br/>
<br/><br/>

#### DATAset explanation
```py
movie_lines.txt
L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ 
```

**+++$+++**: column splitter<br/>
**L1045**: Just an ID for the column<br/>
**u0**: id of user 0<br/>
**m0**: if of movie 0<br/>
**BIANCA**: name of user 0<br/>
**They do not!**: dialog of user0 BIANCA<br/>

```py
movie_conversations.txt
u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
```
**+++$+++**: column splitter<br/>
**u0**: id of user 0<br/>
**u2**: id of user 2<br/>
**m0**: if of movie 0<br/>
**L194, L195**: row id reference to movie_lines.txt<br/>

---
<br/>


## PART-3. Data Preprocessing

Importing the necessary libraries

```py
# Importing the libraries

import numpy as np # to work with array
import tensorflow as tf # to do all the deep learning stuffs
import re # to clear the text and replace
import time # to measure the training time

```

To run the above code - select the four lines and press CMD+ENTER / CTRL+ENTER and find the below results in the iPython console.
![ipython_run](/images/articles/2018/ipython_run.png "ipython_run.png")


**Make sure you get no error to continue further.**


```py
## importing the dataset ##

## giving a reference for the conversatons and movie lines ##
# open and read a file with utf-8 encoding and ignore all errors then split all observations by lines

movie_lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
movie_conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
```

run and check variable explorer for the results
![variable_explorer_import](/images/articles/2018/variable_explorer_import.png "variable_explorer_import")

<br/>
Now we have to create **python dictionary** that maps the **line-ID** and the **line-conversation**.<br/>
The purpose of doing is this so that we get a dataset containing the **inputs** fed into the network and the **outputs** that are the target for the network to learn.<br/> So the easiest way is to create a dictionary.

```py
## creating a dictionary to map each line identifier(id) with its line conversation.
id2line = {} #initialize a dict with {}
for line in movie_lines: #line is a reference to each line in movie_line.txt
    _line = line.split(' +++$+++ ') #be careful and add the extra space with +++$+++ & _line is a temp variable inside for loop
    if len( _line ) == 5 : # just to check if there are 5 elements after split.
        id2line[ _line[0] ] = _line[4] #index in python starts from 0
```
![variable_explorer_id2line1](/images/articles/2018/variable_explorer_id2line1.png "variable_explorer_id2line1")
![variable_explorer_id2line2](/images/articles/2018/variable_explorer_id2line2.png "variable_explorer_id2line2")

```py
## cleaning up the list of all conversations
conversations_ids = []
for conversation in movie_conversations[:-1]: #last row is empty so skipping
    #split and get the last element either use 4 or -1 to get the last element directly 
    #then remove the square brackets on both end 
    #then replace single quote with nothing
    #then replace the space between words with nothing
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    # now we have the list as string - split each by comma to convert from string.
    conversations_ids.append(_conversation.split(','))
```
![comparing two lists](/images/articles/2018/conversation_compare.png "variable_explorer_id2line2")

<br/>

Now we have to convert our conversations ids to relevant questions and answers taking ID from **conversations_ids** and its text from **id2lines**. <br/>

Refer the image below. Here **L194** will be treated as question and next **L195** as answer. Then **L195** becomes question for **L196** and so on.

![conversation_compare_idea](/images/articles/2018/conversation_compare_idea.png "conversation_compare_idea.png")

```py
## Getting the questions and answers from our conversation_ids and pairing it with id2line to get the actual texts
questions = [] # creating a array holder for the questions
answers = [] # creating an array holder for the answers
for conversation in conversations_ids: # conversation local variable for lopping inside conversations_ids
    for i in range(len(conversation) - 1): # i refering to the size of each individual array in list
        questions.append(id2line[conversation[i]]) # refering the i value from id2line dictionary to get the actual text.
        answers.append(id2line[conversation[i+1]]) # first value of i will be treated as question then next i+1 for answer and later i+1 as question
```

Lets clean the text now. We will convert all the text to smallcase and change all short hand words to its actual form. Also remove some unwanted characters.
I hav'nt removed **!(exclamation symbol)** so that this dataset will help in sentiment analysis :P

```py
## doing the cleaning of text -
# converting all texts to smallcase
# converting that's to that is and so on.
def clean_text(text):
    text = text.lower() #to lowercase
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text) # carefull with space he'll to he will
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text
```
```py
## cleaning the question
clean_questions = []
for question in questions:
     clean_questions.append(clean_text(question))
     
## cleaning the answers
clean_answers = []
for answer in answers:
     clean_answers.append(clean_text(answer))  
```

![clean_answer](/images/articles/2018/clean_answer.png "clean_answer.png")


Now lets remove the non-frequently used words from the question and answer so that we can optimize the training session.<br/>
To do this we need to create a dictionary that has word as the key and number of occurances as the value.

```py
## creating a dictionary to map the number of occurances
word2count = {} # creating a new dictionary
for question in clean_questions: #question as the local variable to iterate from the clean_questions list
    for word in question.split():   #word local variable to iterate throught each clean_questions
        if word not in word2count:  # to check if the word exists in the dictionary
            word2count[word] = 1 # adds the word to dictionary and assigns a value of 1
        else:
            word2count[word] += 1 # if word exists in dict then increment its value
     
for answer in clean_answers: #answer as the local variable to iterate from the clean_answers list
    for word in answer.split(): #word local variable to iterate throught each clean_answers
        if word not in word2count:  # to check if the word exists in the dictionary
            word2count[word] = 1 # adds the word to dictionary and assigns a value of 1
        else:
            word2count[word] += 1 # if word exists in dict then increment its value     
```

<br/>

Now lets create two dictionaries that map each word of all the questions to a unique integer and vice versa for answers and while creating this dictionary we will check if the word has higher than a certain threshold to be included in the list else remove those words.

```py
## tokenization (unique number for each word)
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items(): # get words and count from dictionary separately
    if count >= threshold:
        questionswords2int[word] = word_number;
        word_number += 1;
        
answerswords2int = {}
word_number = 0
for word, count in word2count.items(): # get words and count from dictionary separately
    if count >= threshold:
        answerswords2int[word] = word_number;
        word_number += 1;
```

Adding the last tokens to the above two dictionaries - useful for the encoder and decoder in SEQ2SEQ.

```py
# <PAD> : to match the length of question and answer dict
# <EOS> : End of String
# <OUT> : to replace it with the removed words previously
# <SOS> : Start of String
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>'] # use the same format do not change the order
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
```

Creating the inverse dictionary of the answerswords2int dictionary. We will need the inverse mapping from the intergers to answers in the seq2seq model only for the answers

```py
# w_i is the loop variable (integers of the words )
# w key identifiers of the dictionary
answersints2word = { w_i:w for w, w_i in answerswords2int.items()} # inverse mapping shortcut !important!!!
```
![ansint2words.png](/images/articles/2018/ansint2words.png "ansint2words.png")

Now we need to add the < EOS > token to the end of every answers. This is to detect the end of sentense.

```py
## Adding EOS at the end of every answers.
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>' # careful to add the space to seperate the last word woth EOS
```
![eos_clean_answer](/images/articles/2018/eos_clean_answer.png "eos_clean_answer.png")

Translating all words in the clean_answers and clean_questions to their respective interger.

```py
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
            
    questions_to_int.append(ints);
            
answers_to_int = []
for answer in clean_questions:
    ints = []
    for word in question.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints);
    
```

![qa_to_int](/images/articles/2018/qa_to_int.png "qa_to_int.png")


Sorting questions and answers by the length of questions will speed up the training also we will remove the amount of padding during training.

```py
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1,25+1): # upper bound is excluded in python so to get 25 add 1 - For why 25 (dont want the questions too long )
    for i in enumerate(questions_to_int):
        if len(i[1]) == length: # because i is a couple with index and the question as we used enumerate function
            sorted_clean_questions.append(questions_to_int[i[0]]) # use index 0 to get its value
            sorted_clean_answers.append(answers_to_int[i[0]]) # to align questions and answers
     
```
![sorted_clean_questions.png](/images/articles/2018/sorted_clean_questions.png "sorted_clean_questions.png")

---

continued
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