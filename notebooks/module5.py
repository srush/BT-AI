# # Module 5 - Features

all_df["score"] = model.predict_proba(all_df[["feature1", "feature2"]])[:, 0]
all_df

chart2 = (alt.Chart(all_df)
    .mark_rect()
    .encode(
        x = alt.X("feature1:Q", bin=alt.Bin(maxbins=50)),
        y = alt.Y("feature2:Q", bin=alt.Bin(maxbins=50)),
        color = "score"
    ))
chart2


out = chart2 + chart
out

# Alternative data 


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



