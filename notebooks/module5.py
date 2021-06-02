# # Module 5 - Features


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



