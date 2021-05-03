# # Lab Module 2 - Working with Data


# ## Story

# The target of our lab session is to analyze and understand a large dataset
# of climate change data.


# ## Goals

# The goal of this lab module is to explore how to work with
# data from a real-world application. In this module we will
# discuss the challenges of loading data, finding the parts we are
# interested in, and visualizing data output.


# * Understand the raw data
# * Load the data into a "Data Frame"
# * Distinguish different types of data
# * Filter the data based on a certain properties
# * View data tables

# ## Data

# The data that we are working with is located in the file "climate.csv".
# This file is raw data as a text file. We can see the output in raw form.

# Goal: Understand and manipulate climate data

# Steps
# * Load and word with files
# * First steps with dataframes
# * Different types in dataframes

import pandas as pd

df = pd.read_csv("data/Cities.csv")



df


# All the columns

df.columns


# Access a column

df["City"]

# Filter by columns


check = df["Country"] == "Mexico"
mexican_cities_df = df.loc[check]
mexican_cities_df


# Calling functions

total_mexican = len(mexican_cities_df)
total_mexican


# More complex filterings

check = (df["Country"] == "United States") | (df["Country"] == "Canada")
us_or_canada_df = df.loc[check]
us_or_canada_df



# More complex filterings

check = df["Population"] > 1000000
million_or_more_df = df.loc[check]
million_or_more_df


# More complex filterings

check = df["City"].str.contains("City")
city_df = df.loc[check]
city_df




# Adding new columns

check = (df["Country"] == "United States") | (df["Country"] == "Canada")
df["US_or_Canada"] = check
df



# Changing Columns

def change_name(str1):
    return str1.replace(" City", "")


df["City"] = df["City"].map(change_name)
df


# New Columns

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

def abbreviate(str1):
    return abbrev[str1]


df["Abbrev"] = df["Country"].map(abbreviate)
df


# Find facts

df.loc[df["Country"] == "Canada"]


# ## More complex joins

all_cities_df = pd.read_csv("data/AllCities.csv")
all_cities_df

check = all_cities_df["City"] == "New York" 
new_york_df = all_cities_df.loc[check]
new_york_df


# Join

df = df.merge(all_cities_df, on=["City", "Country"])
df




# Remove Strings

def latitude_to_number(latitude_string):
    str1 = latitude_string
    if str1[-1] == "N":
        return float(str1[:-1])        
    else:
        return -float(str1[:-1])        
    


def longitude_to_number(longitude_string):
    str1 = longitude_string.replace("W", "")
    return -float(str1)

# Modify Columns

df["Longitude"] = df["Longitude"].map(longitude_to_number)
df["Latitude"] = df["Latitude"].map(latitude_to_number)
df 



# Plotting

import altair as alt


chart = alt.Chart(df).mark_bar().encode(x="City", y="Population")
chart


chart = alt.Chart(df).mark_bar().encode(x=alt.X("City:N", sort=None),
                                        y="Population:Q")
chart


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
points = alt.Chart(us_cities_df).mark_circle().encode(
    longitude='Longitude',
    latitude='Latitude',
    size="Population",
    tooltip=['City','Population']
)
chart = background + points
chart

df

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
