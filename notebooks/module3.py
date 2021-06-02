# # Lab 3 - Complex Data and Visualization

# Last time we saw a method for selecting and working with data
# in a table. We saw how Pandas as a library allowed us to work
# with data like a spreadsheet and also go beyond simple
# selection and manipulation.


# In this class we will work with data about the temperatures and
# temperature changes of different cities over time.

# [Temperatures](https://docs.google.com/spreadsheets/d/1Jwcr6IBJbOT1G4Vq7VqaZ7S1V9gRmUb5ALkJPaG5fxI/edit?usp=sharing)

# Before we dive in let's do a little review of some of the methods we saw in last class.

import pandas as pd
import altair as alt


# Recall that we first need to load in our data. We saw the `read_csv`
# function from last time. We need to add a bit of extra options in
# order to load this data in. In particular we want to have a date
# column.  One way you can look this up is through the function
# [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).
# Although remember! Often the best thing to do is to find the
# [answer](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
# on stack overflow.

df = pd.read_csv("data/Temperatures.csv",
                 index_col=0,
                 parse_dates=["dt"])
df

# Let's now review the different tools that we have available for us. 
# We can see the different columns in the table. 

df.columns


# We can also filter the table to find only the rows with certain filtered values.

filter = df["City"] == "New York"
nyc_df = df.loc[filter]
nyc_df


# We have seen how we can use multiple filters and combine them with
# elements like or `|` and `&`.

filter = (df["City"] == "New York") | (df["City"] == "Philadelphia") 
nyc_phila_df = df.loc[filter]
nyc_phila_df

# Once there is a dataframe that is filtered in a specific manner
# we can use it to compute properties on the remaining data.

average_temp = nyc_df["AverageTemperature"].mean()
average_temp

# Finally we can add new columns by setting them in the original dataframe. 

def in_nyc(city):
    "Returns Yes if country is in the US or Canada "
    if city == "New York":
        return True
    return False
df["InNYC"] = df["City"].map(in_nyc)
df



# ## Advanced table functions

# Our filters have mainly tried to filter rows by string values,
# but we can filter by many different properties. These properties
# depend on the type of the column.

# If you remember back to lesson 1, we saw how we could use a dates
# in python. 

import datetime
date1 = datetime.datetime.now()
date1

print(date1.day, date1.month, date1.year)

# Let's look now at the types of our columns. 

df.dtypes

# We can see that `dt` is a date. Therefore
# we can access similar properties as we have seen in the table. 

df["dt"].dt.month

# Let's convert these into columns

df["Month"] = df["dt"].dt.month
df["Year"] = df["dt"].dt.year


# Now these are columns in the table. 

df.columns


# ## Advanced  Filtering

filter = (df["Month"] == 7) & (df["Year"] == 1950)
summer = nyc_df.loc[filter]
summer


filter = (df["Year"] >= 1950) & (df["Year"] <= 1960) 
fifties = df.loc[filter]
fifties


# How does the temperature vary over a 10 year period?

filter = df["InNYC"] & (df["Year"] >= 1950) & (df["Year"] <= 1960) 
period = df.loc[filter]



# ## Visualization

# Now lets look at how to graph this data. 


simple_df = pd.DataFrame({
    "names" : ["a", "b", "c"],
    "val1" : [10., 20., 30.],
    "val2" : [10., 20., 30.],
})
simple_df

# Review


chart = (alt.Chart(simple_df)
           .mark_bar()
           .encode(x = "names",
                   y = "val1"))
chart

chart = (alt.Chart(simple_df)
           .mark_line()
           .encode(x = "names",
                   y = "val1"))
chart



# * Chart - Over the last 10 years
# * Mark - Line graph
# * Encode - date and temperature


chart = (alt.Chart(period)
           .mark_line()
           .encode(x = "dt",
                   y = "AverageTemperature"))
chart


# We can instead graph different properties.

# * Chart - Over the last 10 years
# * Mark - Line graph
# * Encode - date and temperature month


chart = (alt.Chart(period)
            .mark_line()
            .encode(x = "Year",
                    y = "AverageTemperature",
                    color = "Month:N"))
chart

# How does the temperature vary over a 200 year period?

filter = df["InNYC"] & (df["Year"] >= 1800) & (df["Year"] <= 2000)
period = df.loc[filter]
period

# How does the temperature vary over a 5 year period?

filter = df["City"].isin(["New York", "Los Angeles", "Detroit"]) & (df["Year"] >= 1950) & (df["Year"] <= 1960)
period = df.loc[filter]
chart = (alt.Chart(period)
         .mark_line()
         .encode(x = "dt",
                 y = "AverageTemperature",
                 color = "City",
                 strokeDash = "City"))
chart


# How does the temperature vary with latitude?

filter = ((df["Country"] == "United States") &
         (df["Year"] == 1950) &
         (df["Month"] == 7))
period = df.loc[filter]
chart = (alt.Chart(period)
         .mark_point()
         .encode(
             y = "AverageTemperature",
             x = "Latitude",
             tooltip=["City", "Country"],
         ))
chart

filter = ((df["Country"] == "United States") &
         (df["Year"] == 1950))
period = df.loc[filter]
chart = (alt.Chart(period)
         .mark_point()
         .encode(
             y = "AverageTemperature",
             x = "Latitude",
             tooltip=["City", "Country"],
             facet="Month"
         ))
chart


# ## Advanced: GroupBys


# GroupBys
#
# 1) Filter - Figure out the data to start with
# 2) GroupBy - Determine the subset of data to use
# 3) Aggregation - Compute a property over the group 


# 1) Filter
filter = ((df["Country"] == "United States") &
          (df["Year"] == 1950))

# 2) Group By
grouped = df.loc[filter].groupby(["Country"])

# 3) Aggregated
temperature = grouped["AverageTemperature"].agg(['mean'])
temperature


# 2) Group By
grouped = df[filter].groupby(["City"])

# 3) Aggregated
temperature = grouped["AverageTemperature"].agg(['mean'])
temperature


# 2) Group By
grouped = df[filter].groupby(["Year", "Country"])

# 3) Aggregated
temperature = grouped["AverageTemperature"].agg(['mean'])
temperature




# Which cities temperature changes the most during the year?

grouped = df.groupby(["City", "Latitude"])

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

