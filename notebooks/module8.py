# # Lab 8 - Deep Learning 3

# The goal of this week's lab is to learn to use another widely-used neural network module: recurrent neural networks (RNNs). We can use it to learn features from sequences such as time series and text.

# ![image](https://www.researchgate.net/profile/Huy-Tien-Nguyen/publication/321259272/figure/fig2/AS:572716866433034@1513557749934/Illustration-of-our-LSTM-model-for-sentiment-classification-Each-word-is-transfered-to-a_W640.jpg)

# How should we extract features from sequences, which might be of variable lengths? Recurrent Neural Networks (RNNs) provide a solution by iteratively applying the same operation to each element of the input sequence, and by maintaining an internal state (memory) to keep track of what they have seen. The final state can be used as a feature representation of the entire sequence.

# This week we will walk through how to use RNNs for sequence processing.

# * **Review**: Convolutional Neural Networks (CNNs)
# * **Unit A**: Time Series Classification and Recurrent Neural Networks (RNNs)
# * **Unit B**: Recurrent Neural Networks for Text Classification

# ## Review

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

# We saw in last week how to store images and their labels in Pandas dataframes.

df_train = pd.read_csv('https://srush.github.io/BT-AI/notebooks/mnist_train.csv.gz', compression='gzip')
df_test = pd.read_csv('https://srush.github.io/BT-AI/notebooks/mnist_test.csv.gz', compression='gzip')

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

# We can visualize an example image.

def draw_image(i, shuffle=False):
    t = df_train[i:i+1].T.reset_index().rename(columns={i: "val"})
    out = t.loc[t["index"] != "class"].apply(position, axis=1, result_type="expand")

    label = df_train.loc[i]["class"]
    title = "Image of a " + str(label)
    if shuffle:
        out["val"] = sklearn.utils.shuffle(out["val"], random_state=1234).reset_index()["val"]
        title = "Shuffled Image of a " + str(label)
        
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
im = draw_image(0)
im

# The task is to classify the label given an image. To do that, we first need to define a function that creates our model.

# Here is what a CNN model looks like.

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
#SOLUTION
def create_cnn_model():
    # create model
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Reshape(input_shape))
    model.add(Conv2D(32, kernel_size=(1, 1), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax")) # output a vector of size 10
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])
    return model
model = KerasClassifier(build_fn=create_cnn_model,
                         epochs=2,
                         batch_size=20,
                         verbose=1)
# fit model
model.fit(x=df_train[features].astype(float),
          y=df_train["class"])
df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)

# Change the model above to have the kernel size of the first convolution layer to be (28, 28), and remove other convolution and pooling layers. How does this affect the performance? Why?

#📝📝📝📝 FILLME
pass
#📝📝📝📝 FILLME
pass
#SOLUTION
def create_cnn_model():
    # create model
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Reshape(input_shape))
    model.add(Conv2D(32, kernel_size=(28, 28), activation="relu"))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax")) # output a vector of size 10
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])
    return model
model = KerasClassifier(build_fn=create_cnn_model,
                         epochs=2,
                         batch_size=20,
                         verbose=1)
# fit model
model.fit(x=df_train[features].astype(float),
          y=df_train["class"])
df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)

# ## Unit A

# ### Time Series Classification and Recurrent Neural Networks (RNNs)

# Just like CNNs are suitable for the processing of 2-D images (and 1-D sequences since they can be viewed as a special case of 2-D images), recurrent neural networks (RNNs) are suitable for 1-D sequence modeling.

# Let's start from a concrete sequence classification example, where we want to classify a curve into one of three possible classes: rectangle, triangle, and ellipse:

df_train = pd.read_csv('shape_classification_train.csv')
df_test = pd.read_csv('shape_classification_test.csv')
df_train

# In this example, the curve is stored as a vector with 55 entries. In the dataframe, the i-th entry of this vector is stored in a column named $i$.
# As before, we store the names of feature columns in a list `features`.

input_length = 55
features = []
for i in range(input_length):
    features.append(str(i))

# We can visualize some examples using the following function.

import matplotlib.pyplot as plt
def draw_curve(i):
    t = df_train[features].iloc[i]
    c = df_train['class'].iloc[i]
    plt.plot(list(t))
    plt.title(f'class {c}')
    plt.show()
draw_curve(0)
draw_curve(3)
draw_curve(5)

# First, let's try to apply an MLP classifier to this problem. Note that we need to set the size of the last layer to be 3 since there are three output classes.

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

# create model
model = KerasClassifier(build_fn=create_mlp_model,
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


# The accuracy of the MLP classifier is quite low considering the simplicity of this task. The major challenge of this task is that even for the same shapes, the positions where they appear in the sequence, and their sizes may vary. For example, let's look at some rectangles.

draw_curve(0)
draw_curve(1)
draw_curve(4)

# 👩🎓**Student question: Do you think MLPs are suitable for this task? Why or why not?**

#📝📝📝📝 FILLME
pass

# ### Recurrent Neural Networks (RNNs)

# Recurrent Neural Networks (RNNs) work by iteratively applying the same base operation to each element in the input sequence. To keep track of what it has seen so far, it also maintains an internal state. We call the base operation an RNN cell. Throughout this lab, we will use a special kind of RNN cell, a Long short-term memory (LSTM) cell due to its empirical successes.

# Let's assume that we have a sequence of inputs $x_1, \ldots, x_T$. (We're notating the input elements as if they are scalars, but you should keep in mind that they might well be vectors themselves.) Let's consider a single update operation at step $t$, where the current internal state is denoted by $h_t$:

# $h_{t+1} = {\text{LSTM}}_{\phi} (h_t, x_t)$,

# where the LSTM cell has two inputs and one output: it uses both the input element $x_t$ and the old memory $h_t$ to compute the updated memory $h_{t+1}$. $\phi$ denotes the parameters of the LSTM cell, which we can adjust during training. For simplicy, we drop these parameters througout the rest of this lab. The internal computations of the LSTM cell are beyond the scope of this course, but for anyone interested in knowing further details, [this blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) might be a good starting point.

# Now that we have defined a single update step, we can chain them together to produce a summary of $x_1, \ldots, x_T$, starting from $h_0=0$:


#\begin{align}
# h_0 &= 0 \\
# h_1 &= \text{LSTM} (h_0, x_1) \\ 
# h_2 &= \text{LSTM} (h_1, x_2) \\ 
# h_3 &= \text{LSTM} (h_2, x_3) \\ 
# \vdots\\
# h_T &= \text{LSTM} (h_{T-1}, x_T) \\ 
# \end{align}"

# $h_T$ can be used as a feature representation for the entire input sequence $x_1, \ldots, x_T$.

# In `Keras`, the API for an LSTM cell is
# ```
# LSTM(
#     units,
#     activation="tanh",
#     recurrent_activation="sigmoid",
#     use_bias=True,
#     kernel_initializer="glorot_uniform",
#     recurrent_initializer="orthogonal",
#     bias_initializer="zeros",
#     unit_forget_bias=True,
#     kernel_regularizer=None,
#     recurrent_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     recurrent_constraint=None,
#     bias_constraint=None,
#     dropout=0.0,
#     recurrent_dropout=0.0,
#     return_sequences=False,
#     return_state=False,
#     go_backwards=False,
#     stateful=False,
#     time_major=False,
#     unroll=False,
#     **kwargs
# )
# ```
# It appears intimidating, but we only need to set `units` in this lab. In a nutshell, it controls the size of the hidden states in the LSTM cell: the larger the size, the more powerful the model will be, but the more likely the model will overfit to training data.

from keras.layers import LSTM
hidden_size = 32
lstm_layer = LSTM(hidden_size)

# This layer can be applied to a sequence of inputs $x_1, \ldots, x_T$, and the output will be the final hidden state $h_T$. Below shows an example of how to use this layer. Note that we need to use a `Reshape` layer to add one additional dimension to the input sequence since the expected input shape of the LSTM layer is `sequence_length x input size`, and the `input_size` in the case is 1 since $x_t$'s are scalars is 1.

from keras.layers import LSTM
hidden_size = 32
lstm_layer = LSTM(hidden_size)

input_shape = (input_length, 1)
model = Sequential()
model.add(Reshape(input_shape))
model.add(lstm_layer)
#
# take the first example as input
inputs = tf.convert_to_tensor(df_train[features].iloc[:1])
output = model(inputs)
print (inputs.shape)
print (output.shape)


# # Group Exercise A

# ## Question 0

# Icebreakers

# Who are other members of your group today?

#📝📝📝📝 FILLME


# * What's their favorite place?

#📝📝📝📝 FILLME


# * What are their goals by the end of the decade?

#📝📝📝📝 FILLME

# ## Question 1

# Look at this figure again. Can you figure out where are $h_t$'s and $x_t$'s? (Don't worry if you not understand the entire diagram does for now, we will elaborate on it later)**

# ![image](https://www.researchgate.net/profile/Huy-Tien-Nguyen/publication/321259272/figure/fig2/AS:572716866433034@1513557749934/Illustration-of-our-LSTM-model-for-sentiment-classification-Each-word-is-transfered-to-a_W640.jpg)

#📝📝📝📝 FILLME
pass

# Why can LSTMs process variable length inputs?

#📝📝📝📝 FILLME
pass

# ## Question 2

# Modify the MLP code to use LSTM instead. We recommend using a hidden size of 32 or 64. Train the model and report the test accuracy. You should expect to see at least 90% accuracy. Hint: don't forget the reshape layer before the LSTM!

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

# Now let's go to a real application: text classification. Text classification raises new challenges, as the inputs are strings instead of the numeric values we have seen so far in this course. In this unit, we will first find a suitable feature representation for the text input, and then we will apply an LSTM-based model to this task.

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
print (labels[2], data[2])

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

# Now that we can convert the original text (a sequence of strings) into a sequence of integer ids, can we directly feed that to the LSTM layer as we did for the shape classification problem?

# If we directly use those integer ids in the neural network, we are implicitly assuming that the word with id `1001` is closer to the word with id `1002` than it is to the word with id `10`. However, the way we constructed the mappings between word types and ids does not provide this property. Therefore, instead of directly using those word ids, for each word id, we maintain a different vector (usually termed an embedding), which can be stored in a matrix $E$ of size `vocab_size x embedding_size`. To get the word embedding for word id i, we can simply take the i-th row in the matrix $E_i$.

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

# So now we have converted words to their word ids to their embeddings. You might notice that the intermediate word id step is not necessary and we can directly map each word type to a word embedding: we used this intermediate word id step since tensors are easier to work with than strings, and we only need to do this conversion once for the dataset.

# 👩🎓**Student question: By representing words as word embeddings, are we still making implicit assumptions that the 1001-st word is closer to the 1002-nd word than it is to the 10-th word?**

#📝📝📝📝 FILLME
pass

# ### Putting Everything Together

# Now we can put everything together and assemble a model for text classification: we have converted the token strings into word ids. The model first uses an embedding layer to convert those word ids into word embeddings, then the LSTM runs on top of those word embeddings, and we use a final projection layer to project to the output shape.

# # Group Exercise B

# ## Question 1

# Take another look at this model diagram. Can you explain what's happening in this diagram? What are the modules used? What are the inputs and outputs of each module?

# ![image](https://www.researchgate.net/profile/Huy-Tien-Nguyen/publication/321259272/figure/fig2/AS:572716866433034@1513557749934/Illustration-of-our-LSTM-model-for-sentiment-classification-Each-word-is-transfered-to-a_W640.jpg)

#📝📝📝📝 FILLME
pass

# ## Question 2

# Finish the TODOs in the below `create_rnn_model` function, train the network and report the test accuracy.

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

# ## Question 3

# Word embeddings might sound like a very abstract concept: we are associating each word with a vector, but what do these vectors mean? What properties do they have? In this question, we will use the Tensorflow Embedding Projector to explore some pretrained word embeddings. (We can also take our trained model and visualize the embeddings from the embedding layer, but we usually need to train on very large datasets to see meaningful visualizations)

# [Embedding Projector](https://projector.tensorflow.org/)

# This visualization tool visualizes word embeddings in a 3-D space, but keep in mind that those embeddings are actually of much higher dimensionality (`embedding_size` is 200 in the default setting), and their neighbors are found in the original (200-D) space, not the 3-D space, which might lead to some seemingly nearby points not being shown as nearest neighbors.

# * Use the search tool in the right-side panel and search "smith". The point cloud in the middle pannel will show this word as well as its nearest neighbors. What did you observe about the neighbors of the word "smith"?

#📝📝📝📝 FILLME
pass

# * Let's try another word "apple" and find its nearest neighbors. What's your observation? Is "apple" considered a fruit or a company here? Do you consider this an issue for using word embeddings to represent text?

#📝📝📝📝 FILLME
pass

# ## Question 4

# Word embeddings are numeric representations of word, hence they support algebraic operations. For example, we cannot compute `water + bird - air` in the string space, but we can compute `embedding of water + embedding of bird - embedding of air`. Then we can convert the resulting vector back to word by finding its nearest neighbors like we did in the previous question.

# Let's use this demo to perform word algebra.

# [Word Algebra](https://turbomaze.github.io/word2vecjson/)

# * Use the tool under section "Word Algebra" to compute water + bird -air and find its nearest neighbors. What do you get?

#📝📝📝📝 FILLME
pass

# * Can you find some other interesting examples?

#📝📝📝📝 FILLME
pass
