# # Lab 2 - Working with Data


# The target of this lab session is to analyze and understand a large
# dataset efficiently. The dataset we will work with is a dataset of
# cities in the US and their climates. The module will
# discuss the challenges of loading data, finding the parts we are
# interested in, and visualizing data output.

# The main technical tool we will be working with is a library known
# as `Pandas`. Despite the silly name, Pandas is a super popular
# library for data analysis. It is used in many technology companies
# for loading and manipulating data. 

import pandas as pd

# We will also use a new library for this lesson that helps us graph and visualize
# our data. 

import altair as alt

# ## Introduction

# This data comes form 

# https://en.wikipedia.org/wiki/List_of_North_American_cities_by_population



# One way to think of Pandas is as a super-powered spreadsheet like
# Excel. For instance lets start in a spreadsheet here:

# https://docs.google.com/spreadsheets/d/1Jwcr6IBJbOT1G4Vq7VqaZ7S1V9gRmUb5ALkJPaG5fxI/edit?usp=sharing


# In this spreadsheet we can do lots of things. Do you know how to do the following:


# * Change the column names
# * Delete a row
# * Make a graph
# * Add a new column


# What about more advanced ideas. Can you?

# * Sort by a column?
# * Add a new column that changes the previous column?
# * Take the sum of a row?
# * Find the highest value in a row?


# In this lab we will work with real-world data to learn how to
# calculate important properties.

# ## Data

# The data that we are working with is located in the file "Cities.csv".
# You can get this file from the internet by running this command.

# wget https://raw.githubusercontent.com/srush/BTT-2021/main/notebooks/data/Cities.csv

# This file is raw data as a text file. We can see the output in raw form.

# head Cities.csv

# We can see that "csv" stands for "comma separated values" as each element
# of the file is split using a comma. 


# To load data in the library we use the following command. Here `df`
# refers to the "DataFrame" which is what Pandas calls a spreadsheet.

df = pd.read_csv("data/Cities.csv")
df


# Just like in a spreadsheet Pandas has multiple columns representing
# the underlying elements in the data. These each have a name here.

df.columns


# To see just the elements in a single column we can use square
# brackets to see just only column.

df["City"]


# Alternatively if we want to see just a single row we can use the `loc`
# command. 

df.loc[1]


# If we want to select several rows we can also pass in a list.

list_of_rows = [1, 5, 6]
df.loc[list_of_rows]


# ## Filters

# These commands are relatively basic though and easy to do in a
# standard spreadsheet. The main power of Pandas comes from the
# ability to select rows based on complex filters.

# For instance, if you were in a spreadsheet, how would you select only the
# rows that correspond to cities in Mexico? It's possible but a bit challenging. 


# In Pandas, we first create a filter. This is kind of like an if statement that gets
# applied to every row. It creates a variable that remembers which rows passed the filter test.

# SKILL Filtering
#
# 1) Define a filter for you dataframe
# 2) Apply filter and rename.

filter = df["Country"] == "Mexico"
filter

# We then apply the filter to select the rows that we would like to
# keep around. Here `cities_in_mexico_df` is a view of the spreadsheet
# with only those rows remaining.

cities_in_mexico_df = df.loc[filter]
cities_in_mexico_df


# We can then count the number of cities in Mexico.

total_mexican = len(cities_in_mexico_df)
total_mexican


# Filters can also be more complex. You can check for basically any property you might
# think of. For instance, here we want to keep both cities in the US and in Canada. The
# symbol `|` means `either-or`. 


filter = (df["Country"] == "United States") | (df["Country"] == "Canada")
us_or_canada_df = df.loc[filter]
us_or_canada_df



# Filters can be of many different types. For instance, when working
# with numerical fields we can have filters based on greater-than and
# less-than comparisons. This filter keeps only cities with greater than a
# million people.

filter = df["Population"] > 1000000
million_or_more_df = df.loc[filter]
million_or_more_df


# Finally Pandas includes a special way for checking for string
# properties. For instance, last class we saw the `contains` function
# which checks that a string contains as given value. Here is how to
# use that function in Pandas. 

filter = df["City"].str.contains("City")
city_df = df.loc[filter]
city_df

# (I didn't know how to do this! I just googled "how to filter by string contains in pandas"!)


# ## Manipulating Tables

# Another useful aspect of tables is to manipulate by adding in new
# columns. The easiest way to add a new column in pandas is to write a
# function that tells us how to create the new column from the other
# columns in the table.

# Here's and example of how to do this.

def in_us_or_canada(country):
    "Returns Yes if country is in the US or Canada "
    if country == "United States":
        return "Yes"
    if country == "Canada":
        return "Yes"
    return "No"


# Now we can add a new column by setting that column equal to
# the country 

df["US_or_Canada"] = df["Country"].map(in_us_or_canada)
df


# A similar technique can be used to manipulate the data in a
# column to change certain values. For instance, we might want to
# remove the final " City" from cities like "New York" 


def change_name(str1):
    return str1.replace(" City", "")


df["City"] = df["City"].map(change_name)
df


# Exercise: New Columns

abbrev = {
    "United States": "US",
    "Mexico" : "MX",
    "Canada" : "CA",
    "Haiti" : "HAT",
    "Jamaica" : "JM",
    "Cuba" : "CU",
    "Honduras" : "HO",
    "Nicaragua" : "NR",
    "Dominican Republic" : "DR",
    "Guatemala" : "G",
    }

def abbreviate(country):
    return abbrev[country]


df["Abbrev"] = df["Country"].map(abbreviate)
df



# ## Joining Together Tables


# Pandas becomes much more powerful when we start to have many
# different tables that relate to each other. For this example we will
# consider another table that provides new information about these
# cities.

# https://docs.google.com/spreadsheets/d/1Jwcr6IBJbOT1G4Vq7VqaZ7S1V9gRmUb5ALkJPaG5fxI/edit?usp=sharing

# wget https://raw.githubusercontent.com/srush/BTT-2021/main/notebooks/data/AllCities.csv

# Lets load this table into a new variable.

all_cities_df = pd.read_csv("data/AllCities.csv")
all_cities_df


# We can see where we in NYC are located in this table.

filter = all_cities_df["City"] == "New York" 
new_york_df = all_cities_df.loc[filter]
new_york_df


# But there are a lot of other cities in this table outside of North America. 

filter = all_cities_df["Country"] == "Germany" 
europe_df = all_cities_df.loc[filter]
europe_df



# We would like to make a combined table that consists of:
#
# * Only Cities in North America
# * Populations for each city
# * Locations for each city.

# This operation is known as a `join` or a `merge` since it joins together
# these two tables. We just need to tell pandas which are the shared columns
# between the two tables. 

df = df.merge(all_cities_df, on=["City", "Country"])
df


# ## Formatting Exercise

# Convert the latitude and longitude strings...

def latitude_to_number(latitude_string):
    str1 = latitude_string
    if str1[-1] == "N":
        return float(str1[:-1])        
    else:
        return -float(str1[:-1])    
df["Latitude"] = df["Latitude"].map(latitude_to_number)


def longitude_to_number(longitude_string):
    str1 = longitude_string.replace("W", "")
    return -float(str1)
df["Longitude"] = df["Longitude"].map(longitude_to_number)



# ## Plotting

# Next class we will dive deeper into plotting and visualization. But
# let's finish with a little demo to show off all the tables we created.


# Example in the spreadsheet


# We can make a graph by converting our table into a `Chart`. We do this
# in three steps

# * Chart - Convert a dataframe to a chart
# * Mark - Determine which type of chart we want
# * Encode - Say which Pandas columns correspond to which dimensions


# For instance if we want to convert our cities to a bar chart we select:

# * Chart - df
# * Mark - Bar chart
# * Encode - City by Population


chart = (alt.Chart(df)
            .mark_bar()
            .encode(x="City",
                    y="Population"))
chart

# There are many different possible graphs from the same data. For
# instance we might want to explore whether how north or south a city
# is impacts its population. This graph plots population against
# latitude. The special `tooltip` argument uses the city column to
# show the name of each city.

# * Chart - df
# * Mark - Point chart
# * Encode - Population, Latitude

chart = (alt.Chart(df)
            .mark_point()
            .encode(y="Latitude",
                    x="Population",
                    tooltip="City"
            ))
chart




# Finally lets do an advanced example. In this example we will plot
# the population of each city in the US over a map. To do this
# we first load in a map of the US states. (This part I found
# out how to do by googling).

from vega_datasets import data
states = alt.topo_feature(data.us_10m.url, feature='states')
background = alt.Chart(states).mark_geoshape().project('albersUsa')


# Now we filter our data to just cities in the United States.
us_cities_df = df.loc[df["Country"] == "United States"]

# And finally we make a plot.

# * Chart - US cities
# * Mark - Circle chart
# * Encode - Population, Longitude, Latitude

chart = (alt.Chart(us_cities_df)
             .mark_circle(color="red")
             .encode(
                 longitude='Longitude',
                 latitude='Latitude',
                 size="Population",
                 tooltip='City'
             ))
chart = background + chart
chart


# With a little more work we can even make this look cooler!

states = alt.topo_feature(data.world_110m.url, feature='countries')
background = alt.Chart(states).mark_geoshape(
    fill='lightgray',
    stroke='white'
).properties(
    width=500,
    height=300
).project('orthographic', rotate= [95, -42, 0])
points = alt.Chart(df).mark_circle().encode(
    longitude='Longitude',
    latitude='Latitude',
    size="Population",
    tooltip=['City','Population']
)
chart = background + points
chart
