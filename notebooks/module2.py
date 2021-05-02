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
    return str1.replace("United States", "USA")


df["Country"] = df["Country"].map(change_name)
df


# New Columns

abbrev = {
    "USA": "US",
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


df.loc[df["Country"] == "Canada"].sort_by
