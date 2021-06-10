# # Lab 2 - Working with Structured Data

# The target of this lab session is to analyze and understand a large
# dataset efficiently. The dataset we will work with is a dataset of
# cities in the US and their climates. The module will
# discuss the challenges of loading data, finding the parts we are
# interested in, and visualizing data output.

# ![pandas](data:image/jpeg;base64,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)

# The main technical tool we will be working with is a library known
# as `Pandas`. Despite the silly name, Pandas is a super popular
# library for data analysis. It is used in many technology companies
# for loading and manipulating data. 


# # Review

# Before we get started let us review some of the Python code that we saw last class. 

# ![python](https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg)

# We first saw a bunch of different types such as numbers, strings, and lists

number1 = 50.5
string1 = "New York"
list1 = ["Queens", "Brooklyn", "Manhattan", "Staten Island", "The Bronx"]

# We then saw some more complex types likes dates and counters.

import datetime
date1 = datetime.datetime.now()
date1

# As there are so many different types in Python, we discussed how
# important it was to use Google and StackOverflow to find examples.

from collections import Counter
counter = Counter(["A", "B", "A", "A", "C", "C", "B", "A", "A", "A"])
counter.most_common()

# Next, we focused on `if` and `for` the two most important control
# elements in python.

# * `if` lets us decide which block of code to run
# * `for` lets us run the same code for each element in a list

for val in range(10):
    print(val)

# Finally we discussed the special case of strings. There are
# many useful ways to find values in strings and create new strings.

"first " + "second"

str1 = "first|second"

str1.split("|")
    
# ## Review Exercise 

# Print only the values in this list that are greater than 20.

list1 = [5, 50, 15, 60, 4, 80]

#📝📝📝📝 FILLME
pass

# # Unit A

# This week is all about `data tables`. Data tables are a common way
# of representing facts anywhere from newspaper articles to scientific
# studies.

# For instance, as a running example let us consider this table from Wikipedia.

# ![new york](https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/LA_Skyline_Mountains2.jpg/120px-LA_Skyline_Mountains2.jpg)

# https://en.wikipedia.org/wiki/List_of_North_American_cities_by_population

# You may have used datatables before in spreadsheets. For example we can put that
# wikipedia table in a spreadsheet.

# https://docs.google.com/spreadsheets/d/1Jwcr6IBJbOT1G4Vq7VqaZ7S1V9gRmUb5ALkJPaG5fxI/edit?usp=sharing

# In this spreadsheet we can do lots of things.

# 👩‍🎓**Student question: Do you know how to do the following?**

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

# ## Pandas

# The data that we are working with is located in the file "Cities.csv".
# You can get this file from the internet by running this command.


# This file is raw data as a text file. We can see the output in raw form.

# https://srush.github.io/BT-AI/notebooks/Cities.csv

# We can see that "csv" stands for "comma separated values" as each element
# of the file is split using a comma. 

# Pandas is as a super-powered spreadsheet.

import pandas as pd

# To load data in the library we use the following command. Here `df`
# refers to the "DataFrame" which is what Pandas calls a spreadsheet.

df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/Cities.csv")
df

# Just like in a spreadsheet Pandas has multiple columns representing
# the underlying elements in the data. These each have a name here.

df.columns

# To see just the elements in a single column we can use square
# brackets to see just only column.

df["City"]

# 👩‍🎓**Student Question: Can you print out another column in the table?**

#📝📝📝📝 FILLME
pass

# Alternatively if we want to see just a single row we can use the `loc`
# command. 

df.loc[1]

# If we want to select several rows we can also pass in a list.

list_of_rows = [1, 5, 6]
df.loc[list_of_rows]

# 👩‍🎓**Student Question: Can you print out the rows of Philadelphia and Los Angeles?**

#📝📝📝📝 FILLME
pass

# ## Filters

# These commands are relatively basic though and easy to do in a
# standard spreadsheet. The main power of Pandas comes from the
# ability to select rows based on complex filters.

# For instance, if you were in a spreadsheet, how would you select only the
# rows that correspond to cities in *Mexico*? It's possible but a bit challenging. 

# In Pandas, we first create a filter. This is kind of like an if statement that gets
# applied to every row. It creates a variable that remembers which rows passed the filter test.

# **Filtering**
#
# 1. Decide on the conditional statements in your filter.
# 2. Define a `filter` varaible for your dataframe .
# 3. Apply filter and rename the dataframe.


# Step 1. Our filter is that we want the Country column to be Mexico

# Step 2. We create a filter variable with this conditional. Notice
# that the filter has a 1 for every city in Mexico and a 0 otherwise.

filter = df["Country"] == "Mexico"
filter

# Step 3. We then apply the filter to select the rows that we would like to
# keep around.

cities_in_mexico_df = df.loc[filter]
cities_in_mexico_df

# We need to be careful to give this a new name. It does not change the original
# dataframe it just shows us the rows we asked for. 

# Filtering is a really important step because it lets us calculate other properties.

# For example, we can then count the number of cities in Mexico.

total_cities_in_mexico = cities_in_mexico_df["City"].count()
total_cities_in_mexico

# Or we can count the population of the biggest cities in Mexico.

total_population_in_mexico = cities_in_mexico_df["Population"].sum()
total_population_in_mexico


# Filters can also be more complex. You can check for any of the  different properties
# you might check for in a standard if statement.

# For instance, here we want to keep both cities in the US and in Canada. The
# symbol `|` means `either-or`. 

filter = (df["Country"] == "United States") | (df["Country"] == "Canada")
us_or_canada_df = df.loc[filter]
us_or_canada_df

# 👩‍🎓**Student Question: How many of the cities are in the US or Canada?**

#📝📝📝📝 FILLME
pass

# Here is a list of the different statements that we commonly use.

# | Filter | Symbol |
# |--------|--------|
# | Or     | \|     |
# | And    | &      |
# | Not    | ~      |
# | Equal  | ==     |
# | Less   | <      |
# | Greater| >      |
# | Greater| >      |
# | In     | .str.contains      |
# | Is one of     | .isin      |

# Note: I didn't know many of these by heart.
# Don't be afraid to google "how to filter by ... in pandas" if you get stuck.

# # Group Exercise A

# ## Question 1 

# Filters can be of many different types. For instance, when working
# with numerical fields we can have filters based on greater-than and
# less-than comparisons.

# Write a filter that keeps only cities with greater than a
# million people.

#📝📝📝📝 FILLME
pass

# How many are there?

#📝📝📝📝 FILLME
pass

# (Be sure to print it out to check that it worked!)

# ## Question 2

# Several cities in North America include the word "City" in their name. 
# Write a filter to find the cities that have "City" in their name. 

#📝📝📝📝 FILLME
pass

# What is the smallest city on this list?

#📝📝📝📝 FILLME
pass

# ## Question 3

# Most of the cities on the list are in Canada, Mexico or the US.

# Can you write a filter to find the cities that are not in any of these countries?

#📝📝📝📝 FILLME
pass

# What is the largest city in this list?

#📝📝📝📝 FILLME
pass

# ## Question 4

# We can also apply filters that look for two properties at the same time. 

# Can you write a filter to find the cities in the US of over a million people?

#📝📝📝📝 FILLME
pass

# How many are there?

#📝📝📝📝 FILLME
pass

# # Unit B

# In this unit we will look at three more advanced Pandas functions.
# Unlike filters, which just remove rows, these will allow use to manipute
# our data to compute new properties and even new columns. 

# ## Group By's

# We saw above how to compute the total number of cities in Mexico on
# our list. We did this by first filtering and then "aggregating" by
# calling `count()`. Here is a reminder. 

filter = df["Country"] == "Mexico"
cities_in_mexico_df = df.loc[filter]
total_cities_in_mexico = cities_in_mexico_df["City"].count()
total_cities_in_mexico

# However, what if we also want to know the number of cities
# in Canada and US and all the other countries on our list.
# We can do this with a group-by operation

# **GroupBy**
#
# 1. GroupBy - Determine the subset of data to use
# 2. Aggregation - Compute a property over the group

# Step 1. Group By

grouped = df.groupby(["Country"])

# Step 2. Aggregate

count_of_cities = grouped["City"].count()
count_of_cities

# Here is another example. This one computes the population of the
# largest city in each country. 

max_pop = grouped["Population"].max()
max_pop

# 👩‍🎓 ** Student Question: Can you compute the city with the minimum population on the list for each country? **

#📝📝📝📝 FILLME
pass

# ## Manipulating Tables

# Another useful aspect of tables is is to add in new columns.
# Adding new columns allows us to group by additional properties,
# create advanced filters, or make pretty graphs.

# The easiest way to add a new column in pandas is to write a function
# that tells us how to create the new column from the other columns in
# the table.

# In order to add a new column, we need to write a function.
# If you remember last class, a function looked something like this.

# Returns if the country is in US or Canada
def in_us_or_canada(country):
    if country == "United States":
        return "US/Canada"
    if country == "Canada":
        return "US/Canada"
    return "Not US/Canada"

# Now we can add a new column by setting that column equal to
# the country. We do this by calling Pandas `map` with the function
# and the column of interest. This line of code will call our function
# for each row of the Country column. Notice how it creates a new column.

df["US_or_Canada"] = df["Country"].map(in_us_or_canada)
df

df.columns

# We can then use this column in a group by.

grouped = df.groupby(["US_or_Canada"])
count_of_cities = grouped["City"].count()
count_of_cities

# A similar technique can be used to manipulate the data in a
# column to change certain values. For instance, we might want to
# remove the final " City" from cities like "New York" 

def change_name(str1):
    return str1.replace(" City", "")

df["City"] = df["City"].map(change_name)
df

# ## Joining Together Tables

# Pandas becomes much more powerful when we start to have many
# different tables that relate to each other. For this example we will
# consider another table that provides the locations about these
# cities. You can see that here: 

# [City Location Spreadsheet](https://docs.google.com/spreadsheets/d/1Jwcr6IBJbOT1G4Vq7VqaZ7S1V9gRmUb5ALkJPaG5fxI/edit?usp=sharing)

# Lets load this table into a new variable.

all_cities_df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/AllCities.csv")
all_cities_df

# This table has most of the cities in our dataset.
# But there are also a lot of other cities in this table outside of North America. 

filter = all_cities_df["Country"] == "Germany" 
europe_df = all_cities_df.loc[filter]
europe_df

# In order to use this new information let's merge since it in to our table. 
# We just need to tell pandas which are the shared columns
# between the two tables. 

df = df.merge(all_cities_df, on=["City", "Country"])
df

# # Group Exercise B

# ## Question 1

# The following are the official abbreviation codes for the cities in our data table.

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

# Can you add a new column to the table called "Abbrev" that lists the abbreviation code for that city?

#📝📝📝📝 FILLME
pass

# ## Question 2

# Our table has the Latitude and Longitude of all the major North American Cities. 

# Can you find out where New York is located? How about Detroit, Las Vegas, and Portland?

# ## Question 3

# Currently in the table the latitude and longitude are represented as string types, because they
# have N / S and E / W in their values. These two functions will fix that issue. 

def latitude_to_number(latitude_string):
    str1 = latitude_string
    if str1[-1] == "N":
        return float(str1[:-1])        
    else:
        return -float(str1[:-1])    

def longitude_to_number(longitude_string):
    str1 = longitude_string.replace("W", "")
    return -float(str1)

lat = latitude_to_number("190N")
lat

# Can you use these functions to fix the Latitude and Longitude columns to instead use numeric values?

#📝📝📝📝 FILLME
pass

# ## Question 4

# After completing question 3 use group by and compute the Latitude of 
# most southern city in each country of the table.

#📝📝📝📝 FILLME
pass

# # Visualization

# Next class we will dive deeper into plotting and visualization. But
# let's finish with a little demo to show off all the tables we created.

# First we import some libraries

import altair as alt
from vega_datasets import data

states = alt.topo_feature(data.us_10m.url, feature='states')
background = alt.Chart(states).mark_geoshape().project('albersUsa')

# Now we can plot

states = alt.topo_feature(data.world_110m.url, feature='countries')
chart = alt.Chart(states).mark_geoshape(
        fill='lightgray',
        stroke='white'
    ).properties(
        width=500,
        height=300
    ).project('orthographic', rotate= [95, -42, 0])
if False:
    points = alt.Chart(df).mark_circle().encode(
        longitude='Longitude',
        latitude='Latitude',
        size="Population",
        tooltip=['City','Population']
    )
    chart += points
chart
