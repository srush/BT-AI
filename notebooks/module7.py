# # Lab 7 - Deep Learning 2

# The goal of this week's lab is to learn to use a widely-used neural
# network modules: convolutional neural networks (CNNs). We can use
# them to learn features from images and even text.

# ![image](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

# Images and text are common data modalities we encounter in
# classification tasks. While we can directly apply the
# linear or multi-layer modules we learned in the past few weeks to
# those modalities, there are neural network modules specifically
# designed for processing them, namely CNNs and RNNs.

# This week we will walk through the basics of CNNs and RNNs.

# * **Review**: Training and Multi-Layer Models (NNs)
# * **Unit A**: Convolution Neural Networks (CNNs)
# * **Unit B**: Image and Text Processing

# ## Review

# Last time we took a look at what's happening inside training when we
# call `model.fit`. We did this by implementing `model.fit` ourselves.

EPOCHS = 1

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
from keras.layers import Dense

# We will also turn off warnings.

import warnings
warnings.filterwarnings('ignore')

# As we have seen in past weeks we load our structured data in
# Pandas format.

df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/circle.csv")
all_df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/all_points.csv")


# Next, we need to define a function that creates our model. This will
# determine the range or different feature shapes the model can learn.

# [TensorFlow Playground](https://playground.tensorflow.org/)

# Depending on the complexity of the data we may select a model that
# is linear or one with multiple layers.


# Here is what a linear model looks like.

def create_linear_model(learning_rate=1.0):
    # Makes it the same for everyone in class
    tf.random.set_seed(2)

    # Create model
    model = Sequential()
    model.add(Dense(1, activation="sigmoid"))

    # Compile model
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    )
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model

# Here is what a more complex multi-layer model looks like.

def create_model(learning_rate=0.4):
    tf.random.set_seed(2)
    # create model
    model = Sequential()
    model.add(Dense(4, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    )

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model

# Generally, we will use "ReLU" for the inner layers and "sigmoid" for
# the final layer. The reasons for this are beyond the class, and mostly
# have to do with computational simplicity and standard practice.

# Once we have described the shape of our model we can turn it into a classifier
# and train it on data.

model = KerasClassifier(build_fn=create_model,
                        epochs=EPOCHS,
                        batch_size=20,
                        verbose=0)


# The neural network is used just like the classifiers from Week 4 and 5. The only
# difference is that we got to design its internal shape.

model.fit(x=df[["feature1", "feature2"]],
          y=(df["class"] == "red"))
df["predict"] = model.predict(df[["feature1", "feature2"]])


# The output of the `model.fit` command tells us a lot of information about
# how the approach is doing.

# In particular if it is working the `loss` should go down and the `accuracy` should
# go up. This implies that the model is learning to fit to the data that we provided it.

# We can view how the model determines the separation of the data.

chart = (alt.Chart(df)
    .mark_point()
    .encode(
        x="feature1",
        y="feature2",
        color="class",
        fill="predict",
    ))
chart


# ### Review Exercise

# Change the model above to have three inner layers with the first having size 10, the second size 5, and the third having size 5. How well does this do on our problem?

#📝📝📝📝 FILLME
pass

# ## Unit A

# ### Image Classification

# Today's focus will be the problem of image classification. The goal is to take an
# image and predict the image class. So instead of predicting whether a point is
# red or blue we will now be predicting whether an image is a cat or a dog, or a
# house or a plane.

# We are going to start with a famous simple image classification tasks known as
# MNist. This dataset consists of pictures of hand-written numbers. The goal is to
# look at the handwriting and determine what the number is.

# Let's start with an image classification task. We will be using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), where the goal is to recognize handwritten digits.

df_train = pd.read_csv('mnist_train.csv.gz', compression='gzip')
df_test = pd.read_csv('mnist_test.csv.gz', compression='gzip')
df_train[:100]

# This data is in the same format that we have been using so far.

# The column `class` stores the class of each image, which is a number between 0 and 9.

df_train[:100]["class"].unique()

# The rest of columns store the features. However there are many more features than before!

# In particular the images are 28x28 pixels which means we have 784 features.
# To make later processing easier, we store the names of pixel value columns in a list `features`.

features = []
for i in range(1, 29):
    for j in range(1, 29):
        features.append(str(i) + "x" + str(j))
len(features)


# These features are the intensity at each pixel : for instance, the column "3x4" stores the pixel value at the 3rd row and the 4th column. Since the size of each image is 28x28, there are 28 rows and 28 columns.


# We can use pandas apply to graph these values for one image.

# Convert feature to x, y, and value.

def position(row):
    y, x = row["index"].split("x")
    return {"x":int(x),
            "y":int(y),
            "val":row["val"]}

# Draw a heat map showing the image.

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

# Here are some example images.

im = draw_image(0)
im

im = draw_image(4)
im

im = draw_image(15)
im


# How can we solve this task? The challenge is that the are many different
# aspects that can make a digit look unique.

# 👩🎓**Student question: What are some features that you use to tell apart digits?**

# We can use the NN classifier we learned
# last week, with a few modifications to change from binary
# classification to multi-class (10-way in this case) classification.


# First, the final layer needs to output 10 different values.
# Also we need to switch `sigmoid` tot a  `softmax`.
# Lastly, we need to change the loss function to
# `sparse_categorical_crossentropy`.

# The practical change is quite small, it should look very similar to
# what we have seen already

def create_model(learning_rate=1.0):
    # Makes it the same for everyone in class
    tf.random.set_seed(2)

    # Create model
    model = Sequential()
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # Compile model
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    )
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


# While these terms are a bit technical, the main thing to know
# is that they team up to change the `loss` function from last
# week to score 10 different values instead of 2.


# Create model
model = KerasClassifier(build_fn=create_model,
                        epochs=EPOCHS,
                        batch_size=20,
                        verbose=False)
# Fit model
model.fit(x=df_train[features].astype(float),
          y=df_train["class"])


# Now that it is fit we can print a summary

print (model.model.summary())

# And predict on test set

df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)

# This simple MLP classifier was able to get 90% accuracy!

# 👩🎓**Student question: what is the size of the input to the model??**

#📝📝📝📝 FILLME
pass

# It is hard to visualize the behavior of the classifier under such a
# high dimensionality. Instead we can look at some examples that the
# classifier gets wrong.

wrong = (df_test["predict"] != df_test["class"])
examples = df_test.loc[wrong]
num = 0
charts = alt.vconcat()
for idx, example in examples.iterrows():
    label = example["class"]
    predicted_label = example["predict"]
    charts &= draw_image(idx)
    num += 1
    if num > 10:
        break

charts

# ### What does a NN see?

# While NN classifiers reach a decent accuracy on this task, it
# doesn't take into account locations in the model. To see this, let's
# shuffle each image in MNIST using the **same** shuffling order.

im = draw_image(0, shuffle=True)
im


im = draw_image(4, shuffle=True)
im


im = draw_image(15, shuffle=True)
im


# 👩🎓**Student question: can you recognize what those images are? Train an MLP classifier on the shuffled images and report the test accuracy. (Hint: replace every `features` with `shuffled_features`)**

#📝📝📝📝 FILLME
pass
# SOLUTION
# create model
model = KerasClassifier(build_fn=create_model,
                        epochs=EPOCHS,
                        batch_size=20,
                        verbose=0)

shuffled_features = sklearn.utils.shuffle(features, random_state=1234)

# Fit model
model.fit(x=df_train[shuffled_features].astype(float),
          y=df_train["class"])

# Predict on test set
df_test["predict"] = model.predict(df_test[shuffled_features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
output = "accuracy: ", accuracy
output

# If you implemented correctly, you should see that the test accuracy
# on the shuffled images is similar to that on the original
# images. Think for a moment why shuffling doesn't change the accuracy
# much.

# NNs do not inherently take into account locations in the input as humans
# do. For instance, the model is not aware that the feature at position
# (4, 4) is closer to the feature at position (5, 4) compared to
# position (14, 4).

# So we can see that an NN classifier does not take into account the
# spatial information: it's simply maintaining a input feature, and
# there is no sense of "closeness" between pixels that are spatially
# close to each other.


# ### Convolutions

# Instead of standard NNs we are going to instead use Convolutional
# Neural Networks (CNNS). CNNs are commonly used to learn
# features from images and time series. They takes into account the
# spatial information allowing for locality.

# Let's first define a CNN layer for images. For now let's assume that
# the input of the CNN layer is a image, and the output of the
# CNN layer is also an image, usually of a smaller size compared
# to the input (without input padding). The `parameters` of a CNN layer
# consist of a little NN that is trying to draw a separator of a region
# of the image. This is called the `filter`.

# To get the output of the CNN layer, we overlay the filter on
# top of the input such that it covers part of the input without
# crossing the image boundary. We start from the upperleft corner.

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/1.png)

# At each overlay position, we apply the filter with the corresponding
# portion of the input. The filter is converting the region it looks
# at into a new feature, the same way that we saw last class.

# In this illustration, the input shown in blue is of size 4x4, and
# the filter shown in pink is of size 2x2. The output size 3x3 is
# determined by the input size and the filter size.

# Now we shift the filter to the right. 

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/2.png)

# We shift the filter to the right again. 

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/3.png)

# We can't shift the filter to the right any more since doing
# so would cross the boundary. Therefore, we start from the first
# column of the second row


# ![image](https://srush.github.io/BT-AI/notebooks/imgs/4.png)

# Moving right again.

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/5.png)

# ### Pattern Matching

# Last week we saw how neural networks were able to learn to match
# patterns in the training data. For instance they were able to spot
# certain features in order to distinguish between red and blue points.

# CNNs can be used to do the same thing. However, instead of looking at
# all the features at once they look at small groups of feature. 

# CNNs can match patterns with proper filter weights. For instance, if
# we set the filter to be $\begin{bmatrix}-0.5 & 0.5 \\ -0.5 &
# 0.5\end{bmatrix}$, it can detect vertical edges.

# In the below example, the first column of the output takes values 1,
# corresponding to a line in the original image.

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/edge1.png)

# What if we shift the edge in the original input to the right by 1
# pixel? We can see that the 1's in the output also shift to the right
# by 1.

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/edge2.png)

# Similarly, if we shift the edge to the right by 2 pixels, the 1's in
# the output shift to the right by 2.

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/edge3.png)


# # Group Exercise A

# ## Question 0

# Icebreakers

# Who are other members of your group today?

#📝📝📝📝 FILLME


# * What's their favorite flower or plant?

#📝📝📝📝 FILLME


# * Do they prefer to work from home or in the office?

#📝📝📝📝 FILLME

# ## Question 1

# Brainstorm ways that you might write a program tell apart the digits 5 and 6.

#📝📝📝📝 FILLME
pass

# Do these methods need to look at the whole image? Which parts do they need to look at?

#📝📝📝📝 FILLME
pass

#  Would these approaches still work if the digits were bigger or smaller? What if they moved around on the page? 

#📝📝📝📝 FILLME
pass

# ## Question 2

# Now it is your turn to finish the result of the CNN computations. Write
# down the full result of applying this CNN layer to the given
# input.

#📝📝📝📝 FILLME
pass

# If we increase the height of the input by 1, how would the size of the output change?
# If we increase the width of the filter by 1, how would the size of the output change?**

#📝📝📝📝 FILLME
pass


# In the above illustrations, it seems that a CNN layer processes the input image in a sequential order. Are computations at different positions dependent upon each other? Can we use a different order such as starting from the bottom right corner? 

#📝📝📝📝 FILLME
pass

# ## Question 3

# Design a filter to detect edges in the horizontal direction.**

#📝📝📝📝 FILLME
pass


# Design a filter to detect edges along a diagonal direction.**

#📝📝📝📝 FILLME
pass

# Is it possible to design a filter to detect other edges such as curves?

#📝📝📝📝 FILLME
pass


# ## Unit B

# ### Multiple Filters 

# In practice, we want to go beyond only being able to detect a single
# type of edge. Therefore, instead of only using a single
# filter, we use multiple "output channels", each with a
# different filter, such that each output channel can detect a
# different kind of pattern.

# The output shape is thereby augmented by
# an output channel dimension, forming a 3-D shape of size (output
# height, output width, output channels).

# Similarly, the input can also have multiple channels: for example,
# the input may be color image split into red, green, and blue channels.
# Or it can be the output of a
# previous CNN layer with multiple output channels.

# Instead of using a single filter, we use one filter per input
# channel, then apply the convolution operation to each input channel
# independently. This process is illustrated below, where for
# simplicity we only use a single output channel.

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/multi_input_channels.png)


# One of the tricky parts of CNNs is keeping track of all of the
# elements grouped together.

# 1. input: (num images, input height, input width, input channels).
# 2. output: (num images, output height, output width, output channels).
# 3. filter: (filter height, filter width, input channels, output channels).


# ### CNN Visualization 

# In the above discussions, we can see that a CNN layer can detect
# edges in the input image. By stacking multiple layers of CNNs
# together, it can learn to build up more and more sophisticated
# pattern matchers, detecting not only edges, but also mid-level and
# high-level patterns.

# To get a sense of how a CNN works in practice
# and the types of patterns it can match, let us look at the CNN
# Explainer. For now let's just play with the demo on the top of the
# website.

# [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

# There are several things to look at in the tool.

# 1. What's the height and width of the input image?
# 2. What's the number of the output channels of conv_1_1?
# 3. What kind of patterns does the model use for each image?

# ### CNN in Keras

# Let's take a look at how to create a CNN layer in Keras.
# ```
# Conv2D(
#    filters,
#    kernel_size,
#    strides=(1, 1),
#    padding="valid",
#)
# ```

# The argument `filters` specifies the number of output channels. The argument `kernel_size` is a tuple (height, width) specifying the size of the filter.

# Now let's discuss what `strides` does.  In the above convolution layer examples, when we shift the filter to the right, we move by 1 pixel; similarly, when we move the filter down, we move down 1 pixel.

# We can generalize the step size of movements using strides. For example, we can use a stride of 2 along the width dimension, so we move by two pixels each time we move right (note that we still move down by 1 pixel since the stride along the height dimension is 1), resulting an output of size 3 x 2.

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/stride1.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/stride2.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/stride3.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/stride4.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/stride5.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/stride6.png)

# 👩🎓**Student question: What would the output be if we use a stride of 2 both along the width dimension and the height dimension?**


#📝📝📝📝 FILLME
pass
# SOLUTION
# $\begin{bmatrix}2 & 3 \\ 1 & 0\end{bmatrix}$

# ### Convolution Layers in Keras

# Now we are ready to implement a CNN layer in Keras! First, we need to import `Conv2D` from `keras.layers`.

from keras.layers import Conv2D

# Now let's use Keras to verify the convolution results we calculated before.

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/9.png)

# Note that we need to do lots of reshapes to add the sample dimension, or the input/output channel dimension. To recap, the relevant shapes are:

# 1. input: (num samples, input height, input width, input channels).
# 2. output: (num samples, output height, output width, output channels).
# 3. filter: (filter height, filter width, input channels, output channels).

input = [[1, 0, 1, 0],
         [0, 1, 1, 1],
         [0, 1, 0, 0],
         [1, 0, 0, 0]]
filter = [[1, 0.],
          [1, 1.]] 

def cnn(input, filter):
    # Code to shape the correct
    filter = tf.convert_to_tensor(filter, dtype=tf.float32)
    filter = tf.reshape(filter, (2, 2, 1, 1))
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.reshape(input, (1, 4, 4, 1))

    # Call Keras 
    cnn_layer = Conv2D(filters=1, kernel_size=(2, 2))
    cnn_layer(input)
    cnn_layer.set_weights((kernel, tf.convert_to_tensor([0.])))
    output = cnn_layer(input)

    # Output
    return tf.reshape(output, (3, 3))

print(cnn(input, filter))

# Yeah our calculations were correct!

# ### Pooling

# In a convolution layer, we took the convolution between the filter
# and a portion of the input to calculate the output. However sometimes
# these areas are very small. What if we want a feature over a larger
# area. 


# If we simply take the max value of that portion of input instead of using the convolution, we get a max pooling layer, as illustrated below. 

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max1.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max2.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max3.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max4.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max5.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max6.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max7.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max8.png)

# ![image](https://srush.github.io/BT-AI/notebooks/imgs/max9.png)

# Let's take a look at how to create a max pooling layer in Keras.
# ```
# MaxPool2D(
#     pool_size=(2, 2),
#     strides=None,
#     **kwargs
#)
# ```
# Notice how similar it is to convolution layers? Can you infer what `strides` does here?

# Now let's use Keras to verify the max pooling results above.

from keras.layers import MaxPool2D

input = [
    [1, 0, 1, 0],
    [0, 1, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
]

input_shape = (1, 4, 4, 1)
input = tf.convert_to_tensor(input, dtype=tf.float32)
input = tf.reshape(input, input_shape)
pooling_layer = MaxPool2D(pool_size=(2, 2),
                          strides=(1, 1))
output = pooling_layer(input)
print (tf.reshape(output, (3, 3)))

# In the above example we used `strides=(1,1)`. However, the most
# common way of using a pooling layer is to set `strides` to be the
# same as `pool_size`, which is the default behavior if we don't set
# `strides`.

pooling_layer = MaxPool2D(pool_size=(2, 2))
output = pooling_layer(input)
print (tf.reshape(output, (2, 2)))

# #### Putting Everything Together

# ![image](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

# Now we can put everything together to build a full CNN classifier for MNIST classification. Below shows an example model, where we need to use a `Reshape` layer to reshape the input into a single-channel 2-D image, as well as a `Flatten` layer to flatten the feature map back to a vector.

from keras.layers import Flatten, Reshape

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
 #
 # create model
model = KerasClassifier(build_fn=create_cnn_model,
                         epochs=EPOCHS,
                         batch_size=20,
                         verbose=0)
# fit model
model.fit(x=df_train[features].astype(float),
          y=df_train["class"])
# print summary
#print (model.model.summary())
# predict on test set
df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)

# We are able to get much better accuracy than using basic NNs!

# # Group Exercise B

# ## Question 1

# Apply the CNN model to the shuffled MNIST dataset. What accuracy do
# you get? Is that what you expected?

#📝📝📝📝 FILLME
pass

# ## Question 2

# For this question we will use the CNN Explainer website.

# [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

# * Use the tool under the section "Understanding Hyperparameters" to figure out the output shape of each layer in the above CNN model.

#📝📝📝📝 FILLME
pass

# * Use `print (model.model.summary())` to print the output shape of each layer. Did you get the same results as above?

#📝📝📝📝 FILLME
pass

# ## Question 3

# Let's apply our model to a different dataset, [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), where the goal is to classify an image into one of the below 10 classes:
#```
#Label 	Description
#0 	T-shirt/top
#1 	Trouser
#2 	Pullover
#3 	Dress
#4 	Coat
#5 	Sandal
#6 	Shirt
#7 	Sneaker
#8 	Bag
#9 	Ankle boot
#```

# Some examples from the dataset are shown below, where each class takes three rows.

# ![image](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

# We have processed the dataset into the same format as MNIST:

df_train = pd.read_csv('fashion_mnist_train.csv.zip', compression='zip')
df_test = pd.read_csv('fashion_mnist_test.csv.zip', compression='zip')
df_train

# Let's visualize some examples first

draw_image(1)

draw_image(4)

draw_image(10)

# Apply the CNN model to this dataset and print out the accuracy.

#📝📝📝📝 FILLME
pass
