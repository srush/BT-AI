# # Lab 3 - Working with Data 2

import pandas as pd
import altair as alt


df = pd.read_csv("data/Temperatures.csv", index_col=0, parse_dates=[1])




df.columns


df.dtypes

df["Month"] = df["dt"].dt.month
df["Year"] = df["dt"].dt.year


nyc_df = df.loc[df["City"] == "New York"]
nyc_df

# Special Filtering


check = (nyc_df["dt"].dt.month == 7) & (nyc_df["dt"].dt.year == 1950)
summer = nyc_df.loc[check]
summer


# How does the temperature vary over a 5 year period?

check = (nyc_df["dt"].dt.year >= 1950) & (nyc_df["dt"].dt.year <= 1960)
period = nyc_df.loc[check]
chart = alt.Chart(period).mark_line().encode(x = "dt", y= "AverageTemperature")
chart

# How does the temperature vary over a 5 year period?

check = (nyc_df["dt"].dt.year >= 1800) & (nyc_df["dt"].dt.year <= 2000)
period = nyc_df.loc[check]
chart = alt.Chart(period).mark_line().encode(x = "dt", y= "AverageTemperature")
chart


check = (nyc_df["dt"].dt.year >= 1800) & (nyc_df["dt"].dt.year <= 2000)
period = nyc_df.loc[check]
chart = alt.Chart(period).mark_line().encode(x = "Year", y= "AverageTemperature", color="Month:N")
chart


# How does the temperature vary over a 5 year period?

cities_df = df.loc[df["City"].isin(["New York", "Los Angeles", "Detroit"])]
check = (cities_df["dt"].dt.year >= 1950) & (cities_df["dt"].dt.year <= 1960)
period = cities_df.loc[check]
chart = alt.Chart(period).mark_line().encode(x = "dt", y= "AverageTemperature", color="City", strokeDash="City")
chart



# How does the temperature vary with latitude?

check = ((df["Country"] == "United States") &
         (df["dt"].dt.year == 1950) &
         (df["dt"].dt.month == 7) )
df2 = df.loc[check]
chart = alt.Chart(df2).mark_point().encode(
    y = "AverageTemperature",
    x = "Latitude",
    tooltip=["City", "Country"],
)
chart

check = ((df["Country"] == "United States") &
         (df["dt"].dt.year == 1950))
df2 = df.loc[check]
chart = alt.Chart(df2).mark_point().encode(
    y = "AverageTemperature",
    x = "Latitude",
    tooltip=["City", "Country"],
    facet="Month"
)
chart


# ## GroupBys


# Which cities temperature changes the most during the year?


check = ((df["Country"] == "United States") &
         (df["dt"].dt.year == 1950))
df2 = df.loc[check]

grouped = df2.groupby(["City", "Latitude"])

var = grouped["AverageTemperature"].agg(['mean', 'std'])
var

var = var.reset_index().sort_values("Latitude")
chart = alt.Chart(var).mark_bar().encode(
    y = "mean",
    x = alt.X("City", sort=None),
)
chart2 = chart.mark_point(color='red').encode(
    y = "std",
    x = alt.X("City", sort=None),
    )
chart = chart + chart2
chart

# Goal: View and Present Climate Data
# 


# ## Simple Graphing

# ## Time-Series Graphs

# ## Grouping Data

# ## Labels, Axes, Colors

