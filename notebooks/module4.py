# # Module 4 - Machine Learning


# Last class we saw some complex tables based operations. 

import sklearn
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import KNNClassifier
import pandas as pd
import altair as alt
import numpy as np


# In this dataset we will start with a classic ML dataset.

df = pd.read_csv("data/DataSet.csv")
df


# ## Training versus Test.

df_train = df.loc[df["Split"] == "Train"]
chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color=["class"]
    ))


df_test = df.loc[df["Split"] == "Test"]
chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = ["class"]
    ))


# ## Machine Learning


# Declarative Machine Learning
# *
# * 

model = LogisticRegression()
model.fit(df_train[["feature1", "feature2"]],
          df_train["class"])


# Predict

df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])


chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = ["class", "predict"]
    ))
chart

chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = ["class", "predict"]
    ))
chart

# Alternative data 


df = pd.read_csv("data/DataSet.csv")
df

df_train = df.loc[df["Split"] == "Train"]

model = LogisticRegression()
model.fit(df_train[["feature1", "feature2"]],
          df_train["class"])

df_test = df.loc[df["Split"] == "Test"]
df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])


# Alternative Approach

# model = KNNClassifier()
# model.fit(df_train[["feature1", "feature2"]],
#           df_train["class"])


df_test = df.loc[df["Split"] == "Test"]
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



