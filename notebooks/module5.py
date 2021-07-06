# # Lab 5 - Machine Learning 2 - Features

# This lab will continue our journey through applied machine learning.
# The main goal of the lab is to gain intuition for *features*. We will
# learn how they can be constructed for structured data and how they allow
# us to use simple models to make complex predictions.

# We will also return to our climate change data and learn how to use
# past data to make future predictions.

# # Review


# Our dataset is a Red versus Blue classification challenge.

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
df_train = df[df["Split"] = "train"]
chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# If we try to fit this data in our standard way.

model = LinearModel()
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

# Last class we saw that we could replace the linear classification with a different classifier. Today we will examine a different approach. We will create new features.

# Let us think about how we would split this data manually.

filter = (df_train["feature1"] < 0.5) & (df_train["feature2"] < 0.5)
df_red = df_train.loc[filter]

# Here is what it looks like.

chart = (alt.Chart(df_red)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
    ))
chart


# Alternatively we can look at the blue pts

filter = (df_train["feature1"] > 0.5) | (df_train["feature2"] < 0.5)
df_blue = df_train.loc[filter]


chart = (alt.Chart(df_blue)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
    ))
chart


# ## Features

df_train["feature3"] = df_train["feature1"] > 0.5
df_train["feature4"] = df_train["feature2"] > 0.5


# Now we fit on the new data

model = LinearModel()
model.fit(X=df_train[["feature3", "feature4"]],
          y=df_train["class"] == "red")

df_train["predict"] = model.predict(df_train[["feature3", "feature4"]])

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


# Wow that was neat. We were able to draw a line that had a square shape. Why was that?

# The trick is that the line is straight in the features that we gave the model .

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature3",
        y = "feature4",
        color="class",
        fill = "predict",
    ))
chart




# # Group Exercise A

# For this exercise we will try to learn classifiers for the following data sets.
# This will let us experiment with different features and their usefulness.



# ## Question 1

df = pd.read_csv("periodic.csv")

# Dataset 1 is a periodic data set.


# The Sine function will turn a point into a curve.

x = sin()

# Use this function to create a new feature (feature 3). Show that training a
# model on just feature 3 will yield an accurate value

#📝📝g📝📝 FILLME
pass

# ## Question 2


# Points from the equator

#📝📝g📝📝 FILLME
pass

# ## Question 3


# Distance from the center.

#📝📝g📝📝 FILLME
pas


# ## Unit B

# ## Real World Data


# Equator
# Seasons

df = pd.read_csv("simple.csv")
df

df_train = df.loc[df["split"] == "Train"]

model = LogisticRegression()
model.fit(df_train[["feature1", "feature2"]],
          df_train["class"])

df_test = df.loc[df["split"] == "Test"]
df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])


# Alternative Approach
from sklearn.neighbors import KNeighborsClassifier

# model = KNNClassifier()
# model.fit(df_train[["feature1", "feature2"]],
#           df_train["class"])


df_test = df.loc[df["split"] == "Test"]
df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])


# ## Evaluation.

df_test["class"]
df_test["predict"]

# Real World example



# Temperature classification.

df = pd.read_csv("data/Temperatures.csv", index_col=0, parse_dates=[1])


check = ((df["Country"] == "United States") &
         (df["dt"].dt.year == 1950) &
         (df["dt"].dt.month == 7) )
df2 = df.loc[check]


out = df2.describe()
out

chart = alt.Chart(df2).mark_point().encode(
    y = "AverageTemperature",
    x = "Latitude",
    tooltip=["City", "Country"],
)
chart


model = sklearn.linear_model.LinearRegression()
model.fit(df2[["Latitude"]], df2["AverageTemperature"])

df_pred = pd.DataFrame({"Latitude": np.linspace(25, 50, 10)})
df_pred["AverageTemperature"] = model.predict(df_pred[["Latitude"]])
df_pred


chart2 = alt.Chart(df_pred).mark_line(color="red").encode(
    y = "AverageTemperature",
    x = "Latitude",
)
out = chart + chart2
out


model_bad = sklearn.linear_model.LinearRegression()
model_bad.fit(df2[["Longitude"]], df2["AverageTemperature"])

df_pred = pd.DataFrame({"Longitude": np.linspace(-150, -75, 10)})
df_pred["AverageTemperature"] = model_bad.predict(df_pred[["Longitude"]])
df_pred


chart = alt.Chart(df2).mark_point().encode(
    y = "AverageTemperature",
    x = "Longitude",
    tooltip=["City", "Country"],
)
chart2 = alt.Chart(df_pred).mark_line(color="red").encode(
    y = "AverageTemperature",
    x = "Longitude",
)
out = chart + chart2
out



from vega_datasets import data

us_cities_df = df.loc[df["Country"] == "United States"]


states = alt.topo_feature(data.us_10m.url, feature='states')
background = alt.Chart(states).mark_geoshape(
    fill='lightgray',
    stroke='white'
).properties(
    width=500,
    height=300
).project('albersUsa')
points = alt.Chart(df2).mark_point(size=100).encode(
    longitude='Longitude',
    latitude='Latitude',
    color="AverageTemperature",
    tooltip=['City','AverageTemperature']
)
chart = background + points
chart


matr = np.linspace((25, -150), (50, -75), 20)
Lat, Log = np.meshgrid(matr[:, 0], matr[:, 1])
df_pred = pd.DataFrame({"Latitude": Lat.flatten(), "Longitude": Log.flatten()})
df_pred["AverageTemperature"] = model.predict(df_pred[["Latitude"]])

points = alt.Chart(df_pred).mark_circle(size=10).encode(
    longitude='Longitude',
    latitude='Latitude',
    color=alt.Color("AverageTemperature", scale=alt.Scale(scheme="reds"))
)
chart = chart + points
chart

# ## Input Formats


model = sklearn.linear_model.LinearRegression()
model.fit(df2[["Latitude"]], df2["AverageTemperature"])




# When are features more complex?


# import sklearn
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# import altair as alt
# import numpy as np

# df = pd.read_csv("data/DataSet.csv",
#                  index_col=0,
#                  parse_dates=[1])

# df_train = df["Split"] == "Train"
# df_test = df["Split"] == "Test"

# model = LogisticRegression()
# features = ["Latitude", "Longitude"]
# model.fit(df_train[features],
#           df_train["AverageTemperature"])

# df_test["predict"] = model.predict(df_test[features])


# results = df_test["predict"] == df_test["correct"]
# results


# # ## When are features more complex?

# # Creating New Features

# df["IsSummer"] = (df["Month"] >= 5) & (df["Month"] <= 9)
# df["IsWinter"] = (df["Month"] >= 12) & (df["Month"] <= 3)

# features = ["Latitude", "Longitude", "IsSummer", "IsWinter"]
# model = LogisticRegression()
# model.fit(df_train[features],
#           df_train["class"])

# results = df_test["predict"] == df_test["correct"]
# results


# # Summer features

# features = ["Latitude", "Longitude", "IsSummer", "IsWinter"]
# model = LogisticRegression()
# model.fit(df_train[features],
#           df_train["class"])

# results = df_test["predict"] == df_test["correct"]
# results


# # Distance from the Equator

# df["DistanceFromEq"] = df["Longitude"].abs()

# features = ["Latitude", "Longitude", "IsSummer", "IsWinter", "DistanceFromEq"]
# model = LogisticRegression()
# model.fit(df_train[features],
#           df_train["class"])

# results = df_test["predict"] == df_test["correct"]
# results


# Example 2: NLP

# Competition.

# Movie reviews, Features

#
