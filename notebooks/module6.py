# # Lab 6 - Deep Learning 1

# The goal of this week's lab is to learn what a deep neural network is and how it relates to previous linear models. We are going to build networks today that can automatically learn to adapt to very complex shapes.



# [image](graph.png)


# In the past two weeks we used a simple linear model for classification. In practice however, many datasets are not linearly separable,
# i.e., not all classification problems can be solved by a linear classifier, as we saw earlier on the circle dataset.


# Deep neural networks provide a way to learn flexible shapes without specifying features.
# This is achieved by stacking multiple layers on top of each others.

# Deep neural networks form the backbone of deep learning approaches, which have seen tremendous successes in applications such as machine translation,
# document summarization, image classification, and speech recognition.

# This week we will walk through the basics of deep neural networks.

# * **Review**: Linear Classifiers and Features
# * **Unit A**: TensorFlow, Training Linear Classifiers 
# * **Unit B**: Neural Networks

# # Review

# Last time we trained a linear classifier for binary classification.

import altair as alt
import pandas as pd
import sklearn.linear_model

# We will also turn off warnings.

import warnings
warnings.filterwarnings('ignore')

# Step 1. Create out data.
url = "https://srush.github.io/BT-AI/notebooks/circle.csv"
df = pd.read_csv("circle.csv")
df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]

# Step 2. Create a linear model and fit it to data.

model = sklearn.linear_model.LogisticRegression()
model.fit(X=df_train[["feature1", "feature2"]],
          y=df_train["class"])

# Step 3. Predict.

df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])

# We can see that the linear classifier fails to classify some data points due to its being linear.

correct = (df_test["predict"] ==  df_test["class"])
df_test["correct"] = correct
chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
        fill = "predict",
        tooltip = ["correct"]
    ))
chart

# We can improve on this approach by adding a new feature column.

def mkdistance(row):
    f1 = row["feature1"] - 0.5
    f2 = row["feature2"] - 0.5
    return f1*f1 + f2*f2

# We add it to the model with either `map` or `apply`.
df_train["feature3"] = df_train.apply(mkdistance, axis=1)
df_test["feature3"] = df_test.apply(mkdistance, axis=1)

# If instead we fit with the new features

model.fit(X=df_train[["feature2", "feature3"]],
          y=df_train["class"] == "red")
df_test["predict"] = model.predict(df_test[["feature2", "feature3"]])


chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
        fill = "predict",
        tooltip = ["correct"]
    ))
chart


# We can also check the behavior of our classifier on all possible inputs.
url = "https://srush.github.io/BT-AI/notebooks/all_points.csv"
all_df = pd.read_csv("all_points.csv")
all_df["feature3"] = all_df.apply(mkdistance, axis=1)
all_df["predict"] = model.predict(all_df[["feature2", "feature3"]])
chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart


# Remember though this is because we changed the features. The model is
# still a linear seperator in the new features.

chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature2",
        y = "feature3",
        color = "class",
        fill = "predict",
        tooltip = ["correct"]
    ))
chart


# ## Review Exercise

# What's our model's accuracy on the test set `df_test`?

#📝📝📝📝 FILLME
pass
df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])
correct = (df_test["predict"] ==  df_test["class"])
accuracy = correct.sum() / correct.size
accuracy

# # Unit A

# Linear classifiers can only produce linear "decision boundaries", i.e., there is a line such that points on one side of the line
# are classified as one class, whereas points on the other side are classified as the other class.

# Last class we got around this issue by telling the ML system what shapes we wanted. However this required us to know enough about the problem to specify what these were.

# For today's class we are going to extend our model such that it can produce more general shapes without requiring us to tell it what they are.

# We will do this with a library known as TensorFlow. TensorFlow is the main library
# that companies doing machine learning use. When people talk about AI these days they are often talking about machine learning with TensorFlow.

# To get started we will need these imports. We will use TensorFlow through a library known as Keras.

import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation

# The first thing we will do is rebuild our LinearClassifier in Keras.

# We also need a function to create model, which is required for building a general `KerasClassifier`.

def create_model():
    # Create model
    tf.random.set_seed(2)
    model = Sequential()
    model.add(Dense(1)) # Linear
    model.add(Activation("sigmoid")) # Classifier
    # Compile model
    sgd = tf.keras.optimizers.SGD(
        learning_rate=1.00
    )
    model.compile(loss="binary_crossentropy",
                  optimizer=sgd,
                  metrics=["accuracy"])
    return model

# Next, we use `KerasClassifier` to turn a TensorFlow model into one
# that can be used by Scikit Learn. We have to give it a couple of
# arguments to get it started.


model = KerasClassifier(build_fn=create_model,
                        epochs=10,
                        batch_size=20,
                        verbose=False)

# Lastly, we fit the classifier on training data and apply it to all points to check the shape of its decision boundary.

url = "https://srush.github.io/BT-AI/notebooks/simple.csv"
df = pd.read_csv("simple.csv")
model.fit(x=df[["feature1", "feature2"]],
          y=(df["class"] == "red"))

df["predict"] = model.predict(df[["feature1", "feature2"]])

chart = (alt.Chart(df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
        fill = "predict",
    ))
chart

# ## Visualizing Linear Training


# We construct this model to give us a better sense of what it is that
# we are doing when we do linear training.

# A benefit of TensorFlow is that it provides a way to influence
# the internal decisions made during the training process and build
# different models.

# To get a sense how this works, let us look at the TensorFlow playground.

# [Tensorflow Playground](https://playground.tensorflow.org/)

# This is a tool that allows us to build different machine learning models
# and play with them in the browser.

# Here is an example that looks like our linear model. Press the `Play` button to run it. 

# [Example 1](https://playground.tensorflow.org/#activation=linear&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.0001&regularizationRate=0&noise=0&networkShape=&seed=0.28207&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


# There are several things to look at in the tool.

# 1. Try clicking on a different Dataset to see how it is difficult for the linear model to fit it.
# 2. Try altering the features to see some of the tricks that we used last week (squares and sines) help fix this issue.
# 3. Look at the `Output` graph above the points on the right. What is that graph showing. 




# ## What happens in training?

# Now let us look at what happens when we build a tool like this
# ourselves. This is the stuff that gets done automatically for us
# inside everything we have see so far.

model = create_model()


# We first make our features and target class. We then convert them to
# TensorFlow format.

X = df[["feature1", "feature2"]]
y = df["class"] == "red"
X = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(y)


# The `loss` tells us how well the model is currently doing. This is
# the graph that we saw in the browser version of the tool. 

loss = model.compiled_loss(y, model(X))
print(loss)

# If we graph it, we can see that it is not doing well.  Many of the points are on the wrong side of the line.

all_df["predict"] = model.predict(all_df[["feature1", "feature2"]]) > 0.5


chart = (alt.Chart(df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
    ))

chart2 = (alt.Chart(all_df)
    .mark_point(color="white")
    .encode(
        x = "feature1",
        y = "feature2",
        fill = "predict",
    ))
chart2 + chart


# To improve our model, we need to adjust the line to minimize the loss function. This can be done using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm to adjust the parameters a tiny bit to reduce the loss.

# The below code takes one step of gradient descent and adjusts the parameters of the model.

pick = []
for i in range(50):
    with tf.GradientTape() as tape: # this turns on gradient computation
        loss = model.compiled_loss(y, model(X, training=True))

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Results
    if i % 2 == 0:
        t = all_df.copy()
        t["predict"] = model.predict(all_df[["feature1", "feature2"]]) > 0.5
        pick.append(t)


vc = alt.vconcat()
for p in pick: 
    chart2 = (alt.Chart(p)
              .mark_point(color="white")
              .encode(
                  x = "feature1",
                  y = "feature2",
                  fill = "predict" 
              ))
    vc &= (chart2 + chart)
vc


# To be more specific, we adjust the parameters opposite the direction of the gradient of the loss with respect to them.
# Doing so decreases the loss if we are cautious enough to take a small adjustment at a time.
# We won't get into details how gradient descent works in this class, but if you are interested, check out its Wikipedia page.




# # Group Exercise A

# ## Question 0

# Icebreakers

# Who are other members of your group today?

#📝📝📝📝 FILLME
pass

# * What sport would they compete in if they were in the Olympics?

#📝📝📝📝 FILLME
pass

# * Do they prefer Popcorn or M&Ms?

#📝📝📝📝 FILLME
pass

# ## Question 1

# For this question we will use the TensorFlow Playground

# [Tensorflow Playground](https://playground.tensorflow.org/)

# * Try changing the learning rate. What happens when it is really low? What happens when it is really high?

#📝📝📝📝 FILLME
pass


# * On the left hand side there is a noise bar. What happens when you move that? Does this make things easier or harder?  

#📝📝📝📝 FILLME
pass

# * Can you get some of the harder datasets using different features? Which ones can you get right?

#📝📝📝📝 FILLME
pass

# ## Question 2

# Now it's a good time for us to look back at how we created and trained a `KerasClassifier` before.
# Do you understand what's happening behind the scene now? Why did we recommend using `epochs=1` for larget datasets
# and `epochs=500` for small datasets?
# Why is the size of the first dimension 5 in the printed summary of the model?

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(1)) # input size: 2 (inferred on-the-fly); output size: 1
    model.add(Activation("sigmoid")) # input size: 1; output size: 1
    # Compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

# model = KerasClassifier(build_fn=create_model,
#                          epochs=500,
#                          batch_size=5,
#                          verbose=0)
# model.fit(x=df_train[["feature1", "feature2"]],
#           y=df_train["class"])
# model.model.summary()

#📝📝📝📝 FILLME
pass



# # Unit B


# ## Back to the Playground

# In this unit we are going to explore how neural networks can learn to produce some of the features from last class automatically.

# Do you remember last class how we had this graph?

df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/complex.csv")

chart = (alt.Chart(df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# We saw that this shape was hard for our linear model to
# learn. Instead we introduced a feature that cut off the
# top and the bottom. This made it possible for us to learn
# a linear model. 


# Let us see how a neural network can handle this problem. 

# [Example 2](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=3&seed=0.42314&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


# This model is able to separate out the points without us telling it the features. Because it can do it by itself, we call it a "feature learning" model.

# Note that this model has two stages, one that decides on the features and one that decides on the final separation. Because of this we call it a Multi-Layer model.

# Let us look back at the playground. If we mouse over the hidden layers, we can see that they each correspond to a linear split. This linear split is how the model decides on the intermediate features.

# For instance, do you remember the feature from last week that looked like this? That is exactly the kind of intermediate feature that a model like this would learn. 

def mkupperfeature(row):
    if row["feature1"] <= 0.5:
        return -1.0
    else:
        return row["feature1"]

all_df["feature3"] = all_df.apply(mkupperfeature, axis=1)

chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "feature3:Q"
    ))
chart


# The same approach can be used for more complex figures like our circle. 

# [Example 3](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=3&seed=0.43804&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


# ## Coding Multiple Leayers

# At its heart, TensorFlow is a library for allowing us to build
# stacks of multiple-layers.


# Let's directly go to the code and add one more layer to our model.


def create_model():
    # create model
    tf.random.set_seed(2)
    model = Sequential()
    model.add(Dense(10)) # the added layer
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # Compile model
    sgd = tf.keras.optimizers.SGD(
        learning_rate=0.5
    )

    model.compile(loss="binary_crossentropy",
                  optimizer=sgd,
                  metrics=["accuracy"])
    return model

# In the above code, we added a linear layer `Dense(4)` which projects the input to a vector of size 16, and then `Dense(1)`
# projects this vector to a scalar. Finally, the sigmoid activation normalizes the output to be a valid probability.

# As before, we create the model, fit it on the training data, and finally apply it to all points to check the shape of its decision
# boundary.

model = KerasClassifier(build_fn=create_model,
                         epochs=20,
                         batch_size=20,
                         verbose=True)

model.fit(x=df[["feature1", "feature2"]],
          y=df["class"]=="red")

model.model.summary()

df["predict"] = model.predict(df[["feature1", "feature2"]])

chart = (alt.Chart(df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
        fill = "predict",
    ))
chart



# # Group Exercise B

# ## Question 1

# For this question, you will use the TensorFlow Playground.

# [Tensorflow Playground](https://playground.tensorflow.org/)

#

# ## Question 2

# Can you further improve the above model to achieve even better accuracy?
# Hint: try stacking more layers (don't forget the nonlinear transformations between linear layers)

# Redraw the decision boundary graph above to visualize the new classifier.

#📝📝📝📝 FILLME
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(32))
    model.add(Activation("relu"))
    model.add(Dense(32))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model
# create model
model = KerasClassifier(build_fn=create_model,
                         epochs=500,
                         batch_size=5,
                         verbose=0)
# fit model on training set
# model.fit(x=df_train[["feature1", "feature2"]],
#           y=df_train["class"])
# # print model info
# print (model.model.summary())
# # predict on all data points
# all_df["predict"] = model.predict(all_df[["feature1", "feature2"]])
# # visualize decision boundary
# chart = (alt.Chart(all_df)
#     .mark_point()
#     .encode(
#         x = "feature1",
#         y = "feature2",
#         color="predict",
#         fill = "predict",
#     ))
# # print accuracy
# df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])
# correct = (df_test["predict"] ==  df_test["class"])
# accuracy = correct.sum() / correct.size
# print ("accuracy: ", accuracy)
# chart


# ## Question 3

# Apply a multi-laye perceptron (MLP) to predict whether a city's temperature is over 20 degrees celsius,
# based on its latitude, the month and year of the query date. We provide basic data processing and a
# naive baseline below

df = pd.read_csv("data/Temperatures.csv", index_col=0, parse_dates=[1]) # load data
df.dropna() # drop rows containing empty values
filter_train = (df["dt"].dt.year >= 1950) & (df["dt"].dt.year <= 2000) # use data between 1950 and 2000 for training
filter_test = (df["dt"].dt.year > 2000) #  use data after 2000 for test
df["Month"] = df["dt"].dt.month # add feature Month
df["Year"] = df["dt"].dt.year # add feature Year
df["class"] = df["AverageTemperature"] > 20 # the label we want to predict, whether temperature is over 20 degrees
df_train = df.loc[filter_train]
df_test = df.loc[filter_test]
out = df_train.describe()
out

# We can visualize our data in different ways.

chart = alt.Chart(df_train.sample(n=500)).mark_point().encode(
    y = "Latitude",
    x = "Month:O",
    color = "class",
    fill = "class",
    tooltip=["City"],
)
chart

chart = alt.Chart(df_train.sample(n=500)).mark_point().encode(
    y = "Latitude",
    x = "Year:O",
    color = "class",
    fill = "class",
    tooltip=["City"],
)
chart

# From the above visualizations, can you tell what features are more informative?

# A simple baseline is to always predict `False`, which is the majority class in the training dataset.

df_test["predict"] = False
correct = (df_test["predict"] ==  df_test["class"])
accuracy = correct.sum() / correct.size
accuracy

# Now it's your turn to develop a model! What accuracy can you get?
# We recommend using a smaller number of epochs by setting `epochs=1` in `KerasClassifier` to make training faster.

#📝📝📝📝 FILLME
pass
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(32))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model
# create model
model = KerasClassifier(build_fn=create_model,
                         epochs=1,
                         batch_size=5,
                         verbose=0)
# fit model
# model.fit(x=df_train[["Latitude", "Month", "Year"]],
#           y=df_train["class"])
# # print summary
# print (model.model.summary())
# # predict on test set
# df_test["predict"] = model.predict(df_test[["Latitude", "Month", "Year"]])
# correct = (df_test["predict"] ==  df_test["class"])
# accuracy = correct.sum() / correct.size
# print ("accuracy: ", accuracy)
