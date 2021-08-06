# # Lab 8 - Deep Learning 3

# The goal of this week's lab is to learn to use another widely-used
# neural network module: recurrent neural networks (RNNs). We can use
# it to learn features from sequences such as time series and then
# discuss text processing and NLP.

# ![image](https://www.researchgate.net/profile/Huy-Tien-Nguyen/publication/321259272/figure/fig2/AS:572716866433034@1513557749934/Illustration-of-our-LSTM-model-for-sentiment-classification-Each-word-is-transfered-to-a_W640.jpg)

# How should we learn features from sequences, which might be of
# variable length? Recurrent Neural Networks (RNNs) provide a solution
# by summarizing the entire sequence into a feature
# representation. This week we will walk through how to use RNNs for
# sequence processing.

# * **Review**: Convolutional Neural Networks (CNNs)
# * **Unit A**: Time Series Classification and Recurrent Neural Networks (RNNs)
# * **Unit B**: Recurrent Neural Networks for Text Classification

# Last time we learned the basics of convolutional neural networks (CNNs) and used them for image classification.

# For Tables
import pandas as pd
# For Visualization
import altair as alt
# For Scikit-Learn
import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
# For Neural Networks
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape

# We will also turn off warnings.

import warnings
warnings.filterwarnings('ignore')


# ## Review

# We saw in last week how to store images and their labels in Pandas dataframes.

df_train = pd.read_csv('https://srush.github.io/BT-AI/notebooks/mnist_train.csv.gz',
                       compression='gzip')
df_test = pd.read_csv('https://srush.github.io/BT-AI/notebooks/mnist_test.csv.gz',
                      compression='gzip')

# The column `class` stores the class of each image, which is a number between 0 and 9.

df_train[:100]["class"].unique()

# The rest of columns store the features, where we have 784 features since our images are 28x28. Each feature stores the intensity at each pixel : for instance, the column "3x4" stores the pixel value at the 3rd row and the 4th column. Since the size of each image is 28x28, there are 28 rows and 28 columns.

# To make later processing easier, we store the names of pixel value columns in a list `features`.

features = []
for i in range(1, 29):
    for j in range(1, 29):
        features.append(str(i) + "x" + str(j))
len(features)


# We used the below utility functions for visualizing the images.

# Convert feature to x, y, and value.

def position(row):
    y, x = row["index"].split("x")
    return {"x":int(x),
            "y":int(y),
            "val":row["val"]}
def draw_image(i):
    t = df_train[i:i+1].T.reset_index().rename(columns={i: "val"})
    out = t.loc[t["index"] != "class"].apply(position, axis=1, result_type="expand")

    label = df_train.loc[i]["class"]
    title = "Image of a " + str(label)
        
    return (alt.Chart(out)
            .mark_rect()
            .properties(title=title)
            .encode(
                x="x:O",
                y="y:O",
                fill="val:Q",
                color="val:Q",
                tooltip=("x", "y", "val")
            ))

# We can visualize an example image.

im = draw_image(0)
im


im = draw_image(5)
im

# The task is to classify the label given an image. To do that, we
# first need to define a function that creates our model.

# Here is what a CNN model looks like. It contains two convolution
# layers and two max pooling layers.

def create_cnn_model():
    # create model
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Reshape(input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax")) # output a vector of size 10
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])
    return model

# Then we create the model and fit it on training data.

model = KerasClassifier(build_fn=create_cnn_model,
                         epochs=2,
                         batch_size=20,
                         verbose=1)
# fit model
model.fit(x=df_train[features].astype(float),
          y=df_train["class"])

# With a trained model, we can apply it to the test dataset and measure the test accuracy.

df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)

# ### Review Exercise

# Change the model above to have the kernel size of convolution layers to be (1, 1). How does this affect the performance? Why?

#📝📝📝📝 FILLME
pass

# Change the model above to have the kernel size of the first convolution layer to be (28, 28), and remove other convolution and pooling layers. How does this affect the performance? Why?

#📝📝📝📝 FILLME
pass

# ## Unit A

# ### Time Series Classification

# The focus of today's class will be on classification just like the
# last couple of weeks. However this week we will focus on `time series`
# data.

# Time series data consists of observations that we see over a period
# of time. Examples include musical notes, medical records,
# temperature changes over time, and as we will see, natural language. 


# Before working on real-world examples let us start with a shape
# classification example. We will see a sequence of observations over
# time in a Pandas Data frame. Each observation will be a feature, and they
# will be given in order. 

# Specifically we are going to classify into one of three possible classes:
# rectangle, triangle, and ellipse:

df_train = pd.read_csv('https://srush.github.io/BT-AI/notebooks/shape_classification_train.csv')
df_test = pd.read_csv('https://srush.github.io/BT-AI/notebooks/shape_classification_test.csv')
df_train


shapes = df_train["class"].unique()
shapes


# In this example, the curve is stored with 55 features. In the
# dataframe, the i-th entry is stored in a column named $i$.  Just
# like with CNNs, we store the names of feature columns in a list
# `features`.

input_length = 55
features = []
for i in range(input_length):
    features.append(str(i))

# We can visualize some examples using the following function.

def position(row):
    return {"time_step":int(row["index"]),
            "val":row["val"]}

def draw_curve(i):
    t = df_train[i:i+1].T.reset_index().rename(columns={i: "val"})
    out = t.loc[t["index"] != "class"].apply(position, axis=1, result_type="expand")
    label = df_train.loc[i]["class"]
    
    return (alt.Chart(out)
            .properties(title="Class " + label)
            .mark_line()
            .encode(
                x="time_step:O",
                y="val:Q",
                tooltip=("time_step", "val")
            ))
    
im = draw_curve(0)
im

im = draw_curve(3)
im

im = draw_curve(5)
im


# Let's try to apply an NN classifier to this problem. This is basically
# the same NN that we build in week 6.  The only difference
# is we need to set the size of the last layer to be 3 since there
# are three output classes.

def create_mlp_model():
    # create model
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax')) # output a vector of size 3
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

# Remember the two steps for building and fitting.

# create model
model = KerasClassifier(build_fn=create_mlp_model,
                        epochs=10,
                        batch_size=20,
                        verbose=0)
model.fit(x=df_train[features], y=df_train["class"])

# Now we predict on the test set.

df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)


# The accuracy of the MLP classifier is quite low considering the simplicity of this task. The major challenge of this task is that even for the same shapes, the positions where they appear in the sequence, and their sizes may vary. 

# 👩🎓**Student question: Do you think MLPs are suitable for this task? Why or why not?**

#📝📝📝📝 FILLME
pass


# The important thing here  though that two versions of the same shape might look quite different. Here are two squares.
    
im = draw_curve(0)
im2 = draw_curve(1)
im + im2.mark_line(color="orange")

# Here are two triangles.

im = draw_curve(35)
im2 = draw_curve(37)
im + im2.mark_line(color="orange")


# Notice that the input features might look quite different. Look how different feature 43 is for these two triangles.  


# ### Recurrent Neural Networks (RNNs)

# Just like CNNs are suitable for the processing of images, recurrent
# neural networks (RNNs) are suitable for sequence modeling. They are
# particularly good at learning features for things that might be
# stretched out like the shapes above.

# RNNs work by applying the same base NN repeatedly (like a for loop). To keep track of
# what it has seen so far, it also maintains `state`. We
# call the base NN the `cell`.

# Here is some code for what an RNN does internally.
# Let's assume that we have a sequence of inputs $x_1, \ldots,
# x_T$.

def RNN(cell, x):
    state = 0
    for i in range(len(x)):
        state = cell(state, x[i])
    return state


# As an example `x`, we can start with one of our shapes.

x = df_train.loc[9][features]



# Do you remember last class when we made "edge features" with convolutions?
# We can also make features with RNNs. For example, here is a cell that
# remembers the highest value that are shape reaches.

def cell_high(state, x_i):
    if x_i > state:
        return x_i
    else:
        return state

# Now we run it.

new_feature = RNN(cell_high, x)
new_feature

# RNNs can be much more powerful though. For example, a more complex feature would
# be to check the average non-zero values it passes through.

def average(state, x_i):
    if state == 0:
        return (0, 1)
    if x_i == 0:
        return state

    v, i = state
    return ((v + (x_i - v) / i, i+1))

# This feature will look very different on a square then a triangle.

new_feature = RNN(average, x)[0]
new_feature

# ### LSTMs in Keras

# In practice we will not need to write these features ourselves. Just like
# other neural networks these features will be learned from data. 

# Throughout this lab, we will use a special cell, called a Long
# short-term memory (`LSTM`). You can think of LSTM and RNN interchangeably,
# but you will often see the term LSTM so it is worth remembering.


# The internal computations
# of the LSTM cell are beyond the scope of this course, but for anyone
# interested in knowing further details, [this blog
# post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
# might be a good starting point. 


# In `Keras`, the API for an LSTM cell looks like this. 
# ```
# LSTM(
#     units,
#     **kwargs
# )
# ```

# If you look at the full docs it will appear intimidating. We only
# need to set `units` in this lab. `Units` is the same as size or
# channels from the last lab. It just determines how many
# transformations we perform each step.


from keras.layers import LSTM
units = 32
lstm_layer = LSTM(units)

# This layer can be applied to a sequence of inputs $x_1, \ldots,
# x_T$, and the output will be the final features.

values_per_time = 1
input_shape = (input_length, values_per_time)
model = Sequential()
model.add(Reshape(input_shape))
model.add(lstm_layer)

# Note that we need to use a `Reshape` because we have 1 value at each
# time.  The LSTM layer allows `sequence_length x values per time` inputs as
# we will see later. 

# To run the model it looks exactly as we saw the last couple weeks. 


# take the first example as input
# input shape: num samples x input_length
# output shape: num samples x hidden size
inputs = tf.convert_to_tensor(df_train[features].iloc[:1])
output = model(inputs)
print (inputs.shape)
print (output.shape)


# ## Group Exercise A

# ### Question 0

# Icebreakers

# Who are other members of your group today?

#📝📝📝📝 FILLME


# * What's their favorite place?

#📝📝📝📝 FILLME


# * What are their goals by the end of the decade?

#📝📝📝📝 FILLME

# ### Question 1

# Look at this figure again. Can you figure out where are `inputs` and
# which are the `cells` and where is the `state`? (Don't worry if you
# not understand the entire diagram does for now, we will elaborate on
# it later)**

# ![image](https://www.researchgate.net/profile/Huy-Tien-Nguyen/publication/321259272/figure/fig2/AS:572716866433034@1513557749934/Illustration-of-our-LSTM-model-for-sentiment-classification-Each-word-is-transfered-to-a_W640.jpg)

#📝📝📝📝 FILLME
pass

# Why can LSTMs process variable length inputs?

#📝📝📝📝 FILLME
pass

# ### Question 2

# We saw an RNN Cell that can compute the maximum value and one that could compute the average value.

# Can you write one that can compute the minimum value it sees? 

#📝📝📝📝 FILLME
pass


# Can you compute one that checks if it ever sees a value that is not 0 or 1? (Helpful to filter out squares)

#📝📝📝📝 FILLME
pass


# Hard: Can you create a cell the computes the biggest difference between steps? 

#📝📝📝📝 FILLME


# ### Question 3

# Modify the NN code to use LSTM instead. We recommend using units of 32 or 64. Train the model and report the test accuracy. You should expect to see at least 90% accuracy. Hint: don't forget the reshape layer before the LSTM!

#📝📝📝📝 FILLME
pass
#SOLUTION
def create_rnn_model():
    tf.random.set_seed(1234)
    # create model
    input_shape = (input_length, 1)
    model = Sequential()
    model.add(Reshape(input_shape))
    model.add(LSTM(32))
    model.add(Dense(3, activation='softmax')) # output a vector of size 3
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])
    return model
#
# create model
model = KerasClassifier(build_fn=create_rnn_model,
                         epochs=10,
                         batch_size=20,
                         verbose=1)
# fit model
model.fit(x=df_train[features], y=df_train["class"])
# print summary
print (model.model.summary())
# predict on test set
df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)

# ## Unit B

# ### Recurrent Neural Networks for Text Classification

# Now let's go to a real application: text classification. Text classification raises new challenges, as the inputs are strings instead of the numeric values we have been familiar with in this course. In this unit, we will first find a suitable feature representation for the text input, and then we will apply an LSTM-based model to this task.

# The text classification task we will be working with is sentiment analysis,  where the goal is to classify the sentiment of a text sequence. In particular, we will use the Stanford Sentiment Treebank v2 (SST-2) dataset, where we want to predict the sentiment (positive or negative) for a movie review.

df_train = pd.read_csv('sst_movie_reviews_processed_train.csv.gz', compression='gzip')
df_test = pd.read_csv('sst_movie_reviews_processed_test.csv.gz', compression='gzip')
df_train[:10]

# The column `class` stores the sentiment of each review, which is either "positive" or "negative".

df_train[:100]["class"].unique()

# The other columns store the words, where the i-th word is stored in a feature column i (counting from 0). For example, the first word of each movie review is stored in column 0, the second word is stored in column 1, and so on. As before, we store all feature column names in a list. The maximum length of sentences is 55 on this dataset, so we have 55 feature columns.

input_length = 55
features = []
for i in range(input_length):
    features.append(str(i))

# Notice that some tokens towards the end are `PAD`. They are actually placeholders to pad every sentence into the same length such that we can store them in a table.

data = df_train[features].values
labels = df_train['class'].values
print (labels[1], data[1])
print (labels[8], data[8])

# ### Input Representation

# Different from all examples we've seen so far, the input in text classification cannot be directly fed into a neural network, since they are strings but not numeric values. A natural idea is to associate each word type with an integer id, such that we can use those integer ids to represent words.

# First, we need to build the mapping from word types to integer ids.

# build vocabulary
word2id = {}
id2word = {}
unassigned_id = 0
for review in data:
    for token in review:
        if token not in word2id:
            word2id[token] = unassigned_id
            id2word[unassigned_id] = token
            unassigned_id += 1
vocab_size = len(word2id)
print ('Vocab size: ', vocab_size)

# With `word2id`, we can map a word to its associated id:

print (word2id['the'])

# With `id2word`, we can map an integer id to the corresponding word:

print (id2word[51])


# 👩🎓**Student question: Convert the sentence "a great cast" to a sequence of integer ids using `word2id`.**

#📝📝📝📝 FILLME
pass
#SOLUTION
print (word2id['a'], word2id['great'], word2id['cast'])

# 👩🎓**Student question: Convert a sequence of integer ids `[7, 8, 9]` to the original sentence.**

#📝📝📝📝 FILLME
pass
#SOLUTION
print (id2word[7], id2word[8], id2word[9])

# Now we can convert all the strings into integer ids using those mappings.

def word_to_id(word):
    return word2id[word]
df_train[features] = df_train[features].applymap(word_to_id)
df_test[features] = df_test[features].applymap(word_to_id)
df_train[:10]

# ### Word Embeddings

# Now that we can convert the original text (a sequence of strings) into a sequence of integer ids, can we directly feed that into the LSTM layer as we did for the shape classification problem?

# If we directly use those integer ids in the neural network, we are implicitly assuming that the word with id `1001` is closer to the word with id `1002` than it is to the word with id `10`. However, the way we constructed the mappings between word types and ids does not provide this property. 
# Instead of directly using those word ids, for each word id, we maintain a different vector (usually termed an embedding), which can be stored in a matrix $E$ of size `vocab_size x embedding_size`. To get the word embedding for word id i, we can simply take the i-th row in the matrix $E_i$.

# In `Keras`, this embedding matrix is maintained in an `Embedding` layer.
# ```
# Embedding(
#    input_dim,
#    output_dim,
#    embeddings_initializer="uniform",
#    embeddings_regularizer=None,
#    activity_regularizer=None,
#    embeddings_constraint=None,
#    mask_zero=False,
#    input_length=None,
#    **kwargs
#)
# ```
# Again, we don't need to use all arguments. The only two arguments that we need to understand are: `input_dim`, which is the size of the vocabulary `vocab_size`, and `output_dim`, which is the size of the word embeddings `embedding_size`.

from keras.layers import Embedding
model = Sequential()
embedding_size = 32
model.add(Embedding(vocab_size, embedding_size))
# The model will take as input an integer matrix of size (num_samples, input_length).
# The output_shape is (num_samples, input_length, embedding_size)
#
# take the first example as input
inputs = tf.convert_to_tensor(df_train[features].iloc[:1])
outputs = model(inputs)
print (inputs.shape)
print (outputs.shape)

# So now we have converted words to their word ids to their embeddings (token strings -> integer word ids -> vector word embeddings). (You might notice that the intermediate word id step is not necessary and we can directly map each word type to a word embedding: we used this intermediate word id step since tensors are easier to work with than strings, and we only need to do this conversion once for the dataset.)

# 👩🎓**Student question: By representing words as word embeddings, are we still making implicit assumptions that the 1001-st word is closer to the 1002-nd word than it is to the 10-th word?**

#📝📝📝📝 FILLME
pass

# ### Putting Everything Together

# Now we can put everything together and assemble a model for text classification: we have converted the token strings into word ids. The model first uses an embedding layer to convert those word ids into word embeddings, then the LSTM runs on top of those word embeddings, and we use a final projection layer to project to the output shape.

# ## Group Exercise B

# ### Question 1

# Take another look at this model diagram. Can you explain what's happening in this diagram? What are the modules used? What are the inputs and outputs of each module?

# ![image](https://www.researchgate.net/profile/Huy-Tien-Nguyen/publication/321259272/figure/fig2/AS:572716866433034@1513557749934/Illustration-of-our-LSTM-model-for-sentiment-classification-Each-word-is-transfered-to-a_W640.jpg)

#📝📝📝📝 FILLME
pass

# ### Question 2

# Finish the `TODO`s in the `create_rnn_model` function, train the network and report the test accuracy.

#📝📝📝📝 FILLME
def create_rnn_model():
    tf.random.set_seed(1234)
    # create model
    model = Sequential()
    # TODO: add embedding layer with embedding_size 32
    pass
    # TODO: add LSTM layer with hidden_size 32
    pass
    model.add(Dense(2, activation='softmax')) 
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model
#
# create model
model = KerasClassifier(build_fn=create_rnn_model,
                        epochs=6,
                        batch_size=150,
                        verbose=1)
# fit model
model.fit(x=df_train[features], y=df_train["class"])
# print summary
print (model.model.summary())
# predict on test set
df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)
pass
# SOLUTION
def create_rnn_model():
    tf.random.set_seed(1234)
    # create model
    model = Sequential()
    # TODO: add embedding layer
    model.add(Embedding(vocab_size, 32)) # output size: length, 32
    # TODO: add LSTM layer
    model.add(LSTM(32))
    model.add(Dense(2, activation='softmax')) 
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model
#
# create model
model = KerasClassifier(build_fn=create_rnn_model,
                        epochs=6,
                        batch_size=150,
                        verbose=1)
# fit model
model.fit(x=df_train[features], y=df_train["class"])
# print summary
print (model.model.summary())
# predict on test set
df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)
pass

# ### Question 3

# Word embeddings might sound like a very abstract concept: we are associating each word with a vector, but what do these vectors mean? What properties do they possess? In this question, we will use the [Tensorflow Embedding Projector](https://projector.tensorflow.org/) to explore some pretrained word embeddings. (We can also take our trained model and visualize the embeddings from the embedding layer, but we usually need to train on very large datasets to see meaningful visualizations)

# [Embedding Projector](https://projector.tensorflow.org/)

# This visualization tool visualizes word embeddings in a 3-D space, but keep in mind that those embeddings are actually of much higher dimensionality (`embedding_size` is 200 in the default setting), and their neighbors are found in the original (200-D) space, not the 3-D space, which might lead to some seemingly nearby points not being shown as nearest neighbors.

# * Use the search tool in the right-side panel and search "smith". The point cloud in the middle pannel will show this word as well as its nearest neighbors. What did you observe about the neighbors of the word "smith"?

#📝📝📝📝 FILLME
pass

# * Let's try another word "apple" and find its nearest neighbors. What's your observation? Is "apple" considered a fruit or a company here? Do you consider this an issue for using word embeddings to represent text?

#📝📝📝📝 FILLME
pass

# ### Question 4

# Word embeddings are numeric representations of word types, hence they support algebraic operations. For example, we cannot compute `water + bird - air` in the string space, but we can compute `embedding_of_water + embedding_of_bird - embedding_of_air`. Then we can convert the resulting vector back to word by finding its nearest neighbors like we did in the previous question.

# Let's use this demo to perform word algebra.

# [Word Algebra](https://turbomaze.github.io/word2vecjson/)

# * Use the tool under section "Word Algebra" to find the nearest neighbors of `water + bird -air`. What do you get?

#📝📝📝📝 FILLME
pass

# * Can you find some other interesting examples?

#📝📝📝📝 FILLME
pass
