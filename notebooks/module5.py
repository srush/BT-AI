# # Lab 5 - Machine Learning 2 - Features

# This lab will continue our journey through applied machine learning.
# The main goal of the lab is to gain intuition for *features*. We will
# learn how they can be constructed for structured data and how they allow
# us to use simple models to make complex predictions.

# We will also return to our climate change data and learn how to use
# past data to make future predictions.

# # Review


# Our dataset is a Red versus Blue classification challenge.

import pandas as pd
import altair as alt
df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/simple.csv")
df

# Machine learning data is always split into at least two groups.
# We get to look at the *train* data to *fit* our model. We then
# use the the *test* data to evaluate our model.

df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]


# We will focus on *classification*. Each data point has a *class* that we are trying to predict.

df_train["class"]

# A point graph allows us to see our problem.

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# We are using the library *scikit-learn* for prediction. We are focusing on
# a method known as Linear Classification. (It has a confusing name in the library).

import sklearn.linear_model
# Rename
LinearClassification = sklearn.linear_model.LogisticRegression


# 1. - Make a models
# 2. - Fit to our data.


model = LinearClassification()
model.fit(X=df_train[["feature1", "feature2"]],
          y=df_train["class"] == "red")


# Linear Classification draw the best linear to split the two classes.
# For now you can ignore how it does that. Intuitively the best line
# is defined by how faw each point is from the line.

# We can see this visually by trying out out every possible point and seeing where the system puts them.

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


# The goal of machine learning is to accurately predict the results of
# unseen data. We use the test data to make this possible.



# ## Review Exercise

# Use the model to predict on the test data (using `model.predict`).
# What is the accuracy (number of points where the prediction is the same as the class)?

#📝📝📝📝 FILLME
pass

# Bonus question. Can you graph the test data showing both the prediction and the true class?


#📝📝📝📝 FILLME
pass

# ## Unit A

# ## Complex Data

# Our model is effective at drawing a line between data. However as we saw last time
# this only works if our data can be split.


df = pd.read_csv("complex.csv")

df_train = df[df["split"] == "train"]
chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# If we try to fit this data in our standard way.

model = LinearClassification()
model.fit(X=df_train[["feature1", "feature2"]],
          y=df_train["class"] == "red")

df_train["predict"] = model.predict(df_train[["feature1", "feature2"]])

# It will fail pretty badly.

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
        fill = "predict",
    ))
chart


# Here is what it is trying to do.

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


# Last class we saw that we could replace the linear classification with a different classifier. Today we will examine a different approach. We will create new features.

# Let us think about how we would split this data manually.

filter = (df_train["feature1"] > 0.5) & (df_train["feature2"] > 0.5)
df_red = df_train.loc[filter]

# Here is what it looks like.

chart = (alt.Chart(df_red)
    .mark_point(color="orange")
    .encode(
        x = "feature1",
        y = "feature2",
    ))
chart


# Alternatively we can look at the blue pts

filter = (df_train["feature1"] < 0.5) | (df_train["feature2"] < 0.5)
df_blue = df_train.loc[filter]


chart = (alt.Chart(df_blue)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
    ))
chart



# ## Features

df_train["feature3"] = df_train["feature1"] > 0.5
df_train["feature4"] = df_train["feature2"] > 0.5


# Now we fit on the new data

model = LinearClassification()
model.fit(X=df_train[["feature3", "feature4"]],
          y=df_train["class"] == "red")

df_train["predict"] = model.predict(df_train[["feature3", "feature4"]])

# Now it works!

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
        fill = "predict",
    ))
chart


# Wow that was neat.

all_df["feature3"] = all_df["feature1"] > 0.5
all_df["feature4"] = all_df["feature2"] > 0.5

all_df["predict"] = model.predict(all_df[["feature3", "feature4"]])

chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "predict",
        fill = "predict",
    ))
chart


# We were able to draw a line that had a square shape. Why was that?

# The trick is that the line is straight in the features that we gave the model .

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature3:Q",
        y = "feature4:Q",
        color="class",
        fill = "predict",
    ))
chart

# And in those we can draw the graph.




# # Group Exercise A

# For this exercise we will try to learn classifiers for the following data sets.
# This will let us experiment with different features and their usefulness.



# ## Question 1

df = pd.read_csv("periodic.csv")
df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]



chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart


# Dataset 1 is a periodic data set.

import math


# The Sine function will turn a point into a curve.

x = math.sin(0.0)

# Use this function to create a new feature (feature 3). Show that training a
# model on just feature 3 will yield an accurate value


#📝📝📝📝 FILLME
pass

# ## Question 2

df = pd.read_csv("center.csv")
df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]

# Points from the equator

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart


#📝📝📝📝 FILLME
pass

# ## Question 3

df = pd.read_csv("circle.csv")

df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]

# Points from the equator

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# Distance from the center.

#📝📝📝📝 FILLME
pass

# ## Unit B

# ## Real World Data


# Temperature classification.

df = pd.read_csv("Temperatures.csv", index_col=0, parse_dates=[1])
df["Temp"] = (df["Temperature"] * 9/5) + 32

check = ((df["Country"] == "United States") &
         (df["Date"].dt.year >= 2000) & (df["Date"].dt.year <= 2001))

df = df.loc[check]


df.columns

# ## When is the temperature over 70 degrees?

df["class"] = df["Temp"] > 70

def mksplit(city):
    if city[0] > "M":
        return "test"
    else:
        return "train"
df["split"] = df["City"].map(mksplit)


# Let us try something simple to start.
df["feature1"] = df["Longitude"]
df["feature2"] = df["Latitude"]


df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]


df_train["class"]

model.fit(X=df_train[["feature1", "feature2"]],
          y=df_train["class"]
)

df_train["predict"] = model.predict(df_train[["feature1", "feature2"]])


# The model does not do that well.

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
        fill = "predict",
        tooltip = "City"
    ))
chart

# Let us see what is happening.

chart = (alt.Chart(df_train)
    .mark_tick(thickness=4)
    .encode(
        x = "City",
        y = "Temp",
        color="class",
        fill = "class",
        tooltip = "City",
        shape="City"
    ))
chart



chart = alt.Chart(df).mark_point().encode(
    y = "Temp",
    x = "Latitude",
    tooltip=["City", "Country"],
)
chart

# It even though in theory Latitude is a good way to tell temperature,
# it is not enough alone. We need to be able to separate out by season as
# well.


df


chart = alt.Chart(df).mark_point().encode(
    y = "Temp",
    x = "dt:T",
    color = "class", 
    tooltip=["City", "Country"],
)
chart

def mkfeature(time):
    return math.cos(((time.month - 1) / 11.0) * 2 * 3.14)

df["feature3"] = df["Date"].map(mkfeature)
df["feature4"] = df["Latitude"]

df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]


model.fit(X=df_train[["feature3", "feature4"]],
          y=df_train["class"]
)

df_train["predict"] = model.predict(df_train[["feature3", "feature4"]])


chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature3",
        y = "feature4",
        color="class",
        fill = "predict",
        tooltip = "City"
    ))
chart

# Check on test data.

df_test["predict"] = model.predict(df_test[["feature3", "feature4"]])

chart = alt.Chart(df_test).mark_point().encode(
    y = "Temp",
    x = "Latitude",
    color="class",
    fill = "predict",

    tooltip=["City", "Country"],
)
chart


# ## Pretty Chart

from vega_datasets import data

us_cities_df = df.loc[df["Country"] == "United States"]

df["predict"] = model.predict(df[["feature3", "feature4"]])

df_summer = df[(df["Date"].dt.month==4) & (df["Date"].dt.year==2001)]

states = alt.topo_feature(data.us_10m.url, feature='states')
background = alt.Chart(states).mark_geoshape(
    fill='lightgray',
    stroke='white'
).properties(
    width=500,
    height=300
).project('albersUsa')
points = alt.Chart(df_summer).mark_point(size=100).encode(
    longitude='Longitude',
    latitude='Latitude',
    fill="Temp",
    color="predict",
    shape="class",
    tooltip=['City', "Temp"]
)
chart = background + points
chart

# ## Other Countries

temp_df = pd.read_csv("Temperatures.csv", index_col=0, parse_dates=[1])

temp_df["Temp"] = (temp_df["Temperature"] * 9/5) + 32

filter = (((temp_df["Country"] == "Canada") |
           (temp_df["Country"] == "France") |
           (temp_df["Country"] == "Brazil")) &
          ((temp_df["Date"].dt.year == 2000) & (temp_df["Date"].dt.month==7)))
df = temp_df.loc[filter]

df


df["class"] = df["Temp"] > 70
df["feature3"] = df["Date"].map(mkfeature)
df["feature4"] = df["Latitude"]


model.fit(X=df[["feature3", "feature4"]],
          y=df["class"]
)
df["predict"] = model.predict(df[["feature3", "feature4"]])

chart = alt.Chart(df).mark_point().encode(
    y = "Temp",
    x = "Latitude",
    color="class",
    fill = "predict",

    tooltip=["City", "Country"],
)
chart


# # Group Exercise B

# For this group exercise you will puzzle through what is giong wrong when we apply the approach
# to other countries on the map.

# ## Question 1

# The model predicts the cities in Brazil mostly incorrectly. What is going wrong with this approach?

#📝📝📝📝 FILLME

# ## Question 2

# Modify one of the features in the model to help it correctly predict the Brazil cities. 

#📝📝📝📝 FILLME
pass

# ## Question 3

# Now consider a comparison of just France and Canada

filter = (((temp_df["Country"] == "France") |
           (temp_df["Country"] == "Canada")) &
          ((temp_df["Date"].dt.year == 2000) & (temp_df["Date"].dt.month==1)))
df = temp_df.loc[filter]
df["class"] = df["Temp"] > 25


# Surprisingly France is at a very northern Latitude, almost as high as Canada.

# Draw a chart like above that shows what happens when you try to run our
# standard model on this data. 


#📝📝📝📝 FILLME
pass


# Propose a different feature and add it to the model to mostly correctly classify this
# dataset. In particular, how does the classifier split France and Canada?


#📝📝📝📝 FILLME
pass

