# # Lab 6 - Deep Learning!

# The goal of this week's lab is to learn what a deep neural network is and how it relates to previous linear models.

# ![python](https://upload.wikimedia.org/wikipedia/commons/0/00/Multi-Layer_Neural_Network-Vector-Blank.svg)

# In the past two weeks we used a simple linear model for classification. In practice however, many datasets are not linearly separable,
# i.e., not all classification problems can be solved by a linear classifier, as we saw earlier on the circle dataset. 
# Deep neural networks provide a way
# of parameterizing more flexible functions beyond linear classifiers. This is achieved by stacking multiple layers of linear and nonlinear transformations. Deep neural networks form
# the backbone of deep learning approaches, which have seen tremendous successes in applications such as machine translation,
# document summarization, image classification, and speech recognition.

# This week we will walk through the basics of deep neural networks.

# * **Review**: Linear Classifiers
# * **Unit A**: Multi-layer Perceptrons (MLPs)
# * **Unit B**: Models, Losses, and Optimizers

# # Review

# Last time we've trained a linear classifier for binary classification.

import warnings
import altair as alt
import pandas as pd
import sklearn.linear_model
warnings.filterwarnings('ignore')

# Step 1. Create out data.

df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/circle.csv")
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

# We can also check the behavior of our classifier on all possible inputs.

all_df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/all_points.csv")
all_df["predict"] = model.predict(all_df[["feature1", "feature2"]])
chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
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

# Linear classifiers can only produce linear decision boundaries, i.e., there is a line such that points on one side of the line
# are classified as one class, whereas points on the other side are classified as the other class. This can be seen from the classification
# results on all data points above (you might notice that the decision boundary does not appear to be a perfect line, which is due to
# using a finite number of points only.)

# For today's class we are going to extend our linear model such that it can produce more general decision planes. Since I don't know how to do
# that in keras, I Googled `keras how to build a general classifier`, and found this 
# [blog post](https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/) by Jason Brownlee. 
# Below provides a summary of the steps involved in creating a more general classifier:

# First, we need to import some new methods from keras:

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation

# We also need a function to create model, which is required for building a general `KerasClassifier`.

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

# Dense creates a linear projection, 
# which produces an output of the size specified in the first argument passed in (we don't need to specify the size of the input since
# it can be inferred on-the-fly or from the output size of the previous layer). Sigmoid is used to normalize the model's output into a valid probability: $\text{sigmoid}(x) = \frac{\exp(x)}{1+\exp(x)}$.

# 👩‍🎓**Student question: verify that the output of a sigmoid function is a valid probability. For a number to be a valid probability, it needs to be between 0 and 1.**

#📝📝📝📝 FILLME
pass

# Next, we use `KerasClassifier` to create a classifier which includes optimization parameters such as the number of epochs, such that we
# can use it in the same way as how we used `LogisticRegression`. We will look into the optimization process in more detail and explain what `epochs` and `batch_size` means later.

model = KerasClassifier(build_fn=create_model,
                         epochs=500,
                         batch_size=5,
                         verbose=0)

# Lastly, we fit the classifier on training data and apply it to all points to check the shape of its decision boundary. This might take a while.

model.fit(x=df_train[["feature1", "feature2"]],
          y=df_train["class"])

all_df["predict"] = model.predict(all_df[["feature1", "feature2"]])

chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart


# 👩‍🎓**Student question: why is the decision boundary linear?**

#📝📝📝📝 FILLME
pass

# One function you might find useful is `model.summary()`, which prints out a summary of our model. Note that this only works either after calling `model.fit`.
# You might notice that the output shape is `(5, 1)`. The size of the first dimension `5` comes from `batch_size=5` when we created the `KerasClassifier`, which
# you'll get a better understanding after unit B.

model.model.summary()

# 👩‍🎓**How many parameters are there in the model?**

#📝📝📝📝 FILLM
pass

# ## Stacking Multiple Layers
# Given that we still end up with a linear classifier, what's the purpose of using `KerasClassifier` above? Well, one
# advantage is that it allows us to stack multiple layers together, resulting in a deep neural network usually called a
# multi-layer perceptron (MLP).

# Let's directly go to the code and add one more layer to our model.

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16)) # the added layer
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

# In the above code, we added a linear layer `Dense(16)` which projects the input to a vector of size 16, and then `Dense(1)`
# projects this vector to a scalar. Finally, the sigmoid activation normalizes the output to be a valid probability.

# As before, we create the model, fit it on the training data, and finally apply it to all points to check the shape of its decision
# boundary.

model = KerasClassifier(build_fn=create_model,
                         epochs=500,
                         batch_size=5,
                         verbose=0)

model.fit(x=df_train[["feature1", "feature2"]],
          y=df_train["class"])

model.model.summary()

all_df["predict"] = model.predict(all_df[["feature1", "feature2"]])

chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart

# 👩‍🎓**Student question: why is the decision boundary still linear? Can we have a nonlinear decision boundary by adding more linear layers?**

#📝📝📝📝 FILLME
pass

# ## Introducing Nonlinearities
# To introduce nonlinearites into our classifier, we insert nonlinear transformations between layers. A commonly used nonlinear transformation is 
# Rectified Linear Unit (ReLU), which is just a fancy name of the following nonlinear function
# $f(x) = \max(0, x)$.
# For example, we can insert a ReLU between our two linear layers:

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

model = KerasClassifier(build_fn=create_model,
                         epochs=500,
                         batch_size=5,
                         verbose=0)

model.fit(x=df_train[["feature1", "feature2"]],
          y=df_train["class"])

model.model.summary()

all_df["predict"] = model.predict(all_df[["feature1", "feature2"]])

chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart

# What's our model's accuracy on the test set `df_test`?

#📝📝📝📝 FILLME
pass
df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])
correct = (df_test["predict"] ==  df_test["class"])
accuracy = correct.sum() / correct.size
accuracy

# # Group Exercise A

# ## Question 1

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
model.fit(x=df_train[["feature1", "feature2"]],
          y=df_train["class"])
# print model info
print (model.model.summary())
# predict on all data points
all_df["predict"] = model.predict(all_df[["feature1", "feature2"]])
# visualize decision boundary
chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
# print accuracy
df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])
correct = (df_test["predict"] ==  df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)
chart


# ## Question 2

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
model.fit(x=df_train[["Latitude", "Month", "Year"]],
          y=df_train["class"])
# print summary
print (model.model.summary())
# predict on test set
df_test["predict"] = model.predict(df_test[["Latitude", "Month", "Year"]])
correct = (df_test["predict"] ==  df_test["class"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)

# ## Question 3

# Apply a multi-laye perceptron (MLP) to predict whether a movie review is positive or negative. For example,
# given a review `This a fantastic movie`, we want to predict `positive`.

df = pd.read_csv("data/movie_reviews.csv") # load data
df_train = df.loc[:40000] # training split
df_test = df.loc[40000:] # test split
out = df_train.describe()
out

# What features should we use? We write down the below 10 words that are likely to be informative.

features = ["excellent", "perfect", "great", "amazing", "superb",
            "worst", "waste", "awful", "poorly", "boring"]
for feature in features:
    for df_split in [df_train, df_test]:
        df_split[feature] = df_split["review"].str.count(feature)
df_test

# Now it's your job to create a multi-layer perceptron (MLP) and predict the sentiment of each movie review
# using those features. What test accuracy can you get?

#📝📝📝📝 FILLME
pass
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
                         epochs=1,
                         batch_size=5,
                         verbose=0)
# fit model
model.fit(x=df_train[features], y=df_train["sentiment"])
# print summary
print (model.model.summary())
# predict on test set
df_test["predict"] = model.predict(df_test[features])
correct = (df_test["predict"] == df_test["sentiment"])
accuracy = correct.sum() / correct.size
print ("accuracy: ", accuracy)

# # Unit B
# Is deep learning magic? What's happening behind `model.fit`? To understand how we fit our model on the training data,
# we need to understand three basic concepts: model, loss, and optimizer. In a nutshell, we first start with a random initial `model`,
# then we use a `loss` function to measure how close the model's predictions match the actual labels. We adjust the
# parameters of the model to reduce the loss function using an `optimizer`, and repeat this `predict->measure loss->adjust parameters to minimize loss`
# process a given number of times.

# Let's revisit the toy classification problem. We will open the black box of `model.fit` to obtain a better
# understanding of the magic of deep learning. Note that we map label red to `1`, and label blue to `0` since it's easier to work with numbers compared to strings.

df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/circle.csv")
df["class"] = (df["class"] == "red") * 1
df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]

# ## Model
# We use a linear model such that we can print out all parameters and see how they change over time.

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(1, input_dim=2)) # input size: 2; output size: 1
    model.add(Activation("sigmoid")) # input size: 1; output size: 1
    # Compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model
model = create_model()
model.summary()

# In our model, we first project the input vector of size 2 to a scalar of size 1 using a `Dense` layer, and then we use a `sigmoid` function
# (recall that $\text{sigmoid}(x) = \frac{\exp(x)}{1+\exp(x)}$) to normalize the output to be a valid probability. The math formula of our model can be written as

# $P(y_{\text{pred}}=1 | x) = \text{sigmoid}(k1 * feature1 + k2 * feature2 + bias)$,

# where `feature1` and `feature2` are input features, while `k1`, `k2`, and `bias` are the parameters of the model. Note that `sigmoid` is a fixed function
# that does not have any parameters.

# We can print out the parameters of our model using `model.trainable_variables`.

model.trainable_variables

# As an example, assume that our model has `k1=1.5`, `k2=2.5`, `bias=0`, then the predicted $P(y_{\text{pred}}=1 | x)$ for the first training example can be computed as below.

# get the first training example
x = df_train[["feature1", "feature2"]].loc[0:0]
feature1 = x["feature1"].item()
feature2 = x["feature2"].item()

# compute P(y=1 | x)
import math
def sigmoid(x):
    Ex = math.exp(x)
    return Ex / (1 + Ex)
def linear_model(feature1, feature2, k1, k2, bias):
    return sigmoid(k1*feature1 + k2*feature2 + bias)
k1 = 1.5
k2 = 2.5
bias = 0
P_y1 = linear_model(feature1, feature2, k1, k2, bias)
P_y1


# 👩‍🎓**Student question: use `print(model.trainable_variables)` to print out the parameters of the model, verify that `model.predict(x)` gets the same result as `linear_model(feature1, feature2, k1, k2, bias). (You need to guess which is k1, k2, and bias in the printed out message)`**

#📝📝📝📝 FILLME
pass
P_y1 = model.predict(x)
print (P_y1)
P_y1 = linear_model(feature1, feature2, model.trainable_variables[0][0] , model.trainable_variables[0][1], model.trainable_variables[1])
print (P_y1)

# ## Loss
# In keras, when a model is created, its parameters are initialized randomly. Without any training, our model in its initial state is unlikely to perform well on the classification task.
# This can be clearly shown by plotting the decision boundary of the original model

all_df["predict"] = model.predict(all_df[["feature1", "feature2"]]) > 0.5
chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart

# How do we measure how good our model is? Intuitively, a model is bad when it predicts a low $P(y_{\text{pred}}=1 | x)$ when the true label is 1,
# or a low $P(y_{\text{pred}}=0 | x)$ when the true label is 0. These two cases can be merged as a low $P(y_{\text{pred}}=y_{\text{truth}} | x)$ where $y_{\text{truth}}$ is the true label.
# Therefore, we can define the loss function as

# $loss = - \log P(y_{\text{pred}}=y_{\text{truth}} | x) $,

# where we use a minus sign, because we want the loss (or error) to be high when the model is bad.

# 👩‍🎓**Student question: what is the lowest possible loss? How can that be achieved??**

#📝📝📝📝 FILLME
pass

# Now our goal is to find the parameters of the model such that $loss$ is minimized, or equivalently, $\log P(y_{\text{pred}}=y_{\text{truth}} | x)$ is maximized.
# This loss function is called the cross entropy loss, and it's called binary cross entropy loss when there are only two possible classes $y_{\text{truth}}$.
# It is also called maximum-likelihood estimation (MLE) since we want to maximize the likelihood of data under our model.
# In keras, the binary cross entropy loss is defined using `model.compile(loss="binary_crossentropy",...)`

# Now let's compute the average loss over the training set.

import tensorflow as tf
x = df_train[["feature1", "feature2"]]
y = df_train["class"]
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)
# predict P(y=1|x)
y_pred = model(x)
# compute loss
loss = model.compiled_loss(y, y_pred)
print (loss)

# ## Gradient Descent Optimizer
# To improve our model, we need to adjust its parameters to minimize the loss function. This can be done using an optimizer, which uses a variant of the
# [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm to adjust the parameters a tiny bit to reduce the loss.

# ![gradient descent](https://upload.wikimedia.org/wikipedia/commons/7/79/Gradient_descent.png)

# To be more specific, we adjust the parameters opposite the direction of the gradient of the loss with respect to them. 
# Doing so decreases the loss if we are cautious enough to take a small adjustment at a time. 
# We won't get into details how gradient descent works in this class, but if you are interested, check out its Wikipedia page.

# The below code takes one step of gradient descent and adjusts the parameters of the model.
with tf.GradientTape() as tape: # this turns on gradient computation
    # predict
    y_pred = model(x, training=True)
    # compute loss
    loss = model.compiled_loss(y, y_pred)
# compute gradients
trainable_vars = model.trainable_variables
gradients = tape.gradient(loss, trainable_vars)
# adjust parameters to decrease loss using our optimizer
model.optimizer.apply_gradients(zip(gradients, trainable_vars))

# We can verify that the loss becomes a tiny bit smaller after one step of update.

# predict P(y=1|x)
y_pred = model(x)
# compute loss
loss = model.compiled_loss(y, y_pred)
print (loss)

# To get an even better model, we need to repeat the above steps for many iterations.

epochs = 5000
losses = []
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # predict
        y_pred = model(x, training=True)
        # compute loss
        loss = model.compiled_loss(y, y_pred)
        losses.append(loss.numpy()) # bookkeeping
    # compute gradients
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # adjust parameters
    model.optimizer.apply_gradients(zip(gradients, trainable_vars))

# We can see that loss is gradually decreasing as we update more.

df_loss = pd.DataFrame({'loss': losses, 'epoch': list(range(epochs))})
chart = (alt.Chart(df_loss)
    .mark_line()
    .encode(
        x = "epoch",
        y = "loss",
    ))
chart

# 👩‍🎓**Student question: how should we decide how many epochs we need?**

#📝📝📝📝 FILLME
pass

# And we can get a good final model.

all_df["predict"] = model.predict(all_df[["feature1", "feature2"]]) > 0.5
chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart


# ## Stochastic Gradient Descent
# In the above process, we computed loss and updated parameters on the entire dataset using the gradient descent algorithm. Therefore,
# we used the number of gradient updates and the number of epochs (an epoch is a full pass over the entire training dataset) interchangably.
# In practice, some datasets are huge (such as the sentiment analysis dataset), and adjusting parameters only once per the entire
# dataset is costly (since we can only make small adjustments at a time). To make training more efficient, [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
# uses a sampled *subset* of data to compute loss and adjust parameters.

# In keras, the size of the subset used for computing loss is specified using `batch_size`. For example,  `model = KerasClassifier(..., epochs=500, batch_size=5)`
# computes loss using 5 examples each time, and performing `training set size / batch_size` gradient updates is defined as an epoch,
# as after these many updates, `training set size / batch_size * batch_size = training set size` examples are used.

# 👩‍🎓**Student question: using dataset `df_train`, with `batch_size=5` and `epochs=500`, how many gradient updates are applied in total? What if we change `batch_size` to 50?**

#📝📝📝📝 FILLME
pass

# 👩‍🎓**Student question: how should we set `batch_size` if we want to use gradient descent? Does it make sense to use a batch size larger than this value?**

#📝📝📝📝 FILLME
pass

# # Group Exercise B

# ## Question 1

# We only used a linear model above. Can you change the model to a multi-layer perceptron (MLP)? For simplicity we still use gradient descent algorithm.

#📝📝📝📝 FILLME
pass
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=2)) # input size: 2; output size: 32
    model.add(Activation("relu")) # input size: 32; output size: 32
    model.add(Dense(1)) # input size: 32; output size: 1
    model.add(Activation("sigmoid")) # input size: 1; output size: 1
    # Compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model
model = create_model()
model.summary()
x = df_train[["feature1", "feature2"]]
y = df_train["class"]
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)
epochs = 5000
losses = []
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # predict
        y_pred = model(x, training=True)
        # compute loss
        loss = model.compiled_loss(y, y_pred)
        losses.append(loss.numpy())
    # compute gradients
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # adjust parameters
    model.optimizer.apply_gradients(zip(gradients, trainable_vars))
all_df["predict"] = model.predict(all_df[["feature1", "feature2"]]) > 0.5
chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart

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
model = KerasClassifier(build_fn=create_model,
                         epochs=500,
                         batch_size=5,
                         verbose=0)
model.fit(x=df_train[["feature1", "feature2"]],
          y=df_train["class"])
model.model.summary()

#📝📝📝📝 FILLME
pass
