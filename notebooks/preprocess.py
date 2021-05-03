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

import pandas as pd

def pop(spop):
    return int(spop.split("[")[0].replace(",", ""))


@__st.cache()
def full_cities():
    cities = "https://en.wikipedia.org/wiki/List_of_North_American_cities_by_population"
    df = pd.read_html(cities)[0]
    df["Population"] = df["Population"].map(pop)
    return df


df = full_cities()
cities = set(df["City"])
cities = cities | set([city.replace(" City", "") for city in cities])
out_df = df[["City", "Country", "Population"]]


out_df.to_csv("NorthAmerica.csv")
out_df




# ## Steps
# * Load files
# *


# ## Building Dataframes



@__st.cache()
def full_data():
    return pd.read_csv("GlobalLandTemperaturesByCity.csv")


df = full_data()


# df = df[(df["Country"] == "Mexico") | (df["Country"] == "United States") | (df["Country"] == "Canada")]


# df = df[pd.to_datetime(df["dt"]).dt.year > 1950 ]
df = df[pd.to_datetime(df["dt"]).dt.month.isin(list(range(1, 13, 3)))]


df = df[df["City"].isin(cities)]


def latitude_to_number(latitude_string):
    str1 = latitude_string
    if str1[-1] == "N":
        return float(str1[:-1])        
    else:
        return -float(str1[:-1])        
    


def longitude_to_number(longitude_string):

    str1 = longitude_string
    if str1[-1] == "W":
        return -float(str1[:-1])        
    else:
        return float(str1[:-1])        


# Modify Columns

df["Longitude"] = df["Longitude"].map(longitude_to_number)
df["Latitude"] = df["Latitude"].map(latitude_to_number)


#cities = set(df["City"])
#cities


df.to_csv("USLandTemp.csv")

df

# ## Loading Data from Files

# ## Filtering Dataframes

# ## Datatypes

# ## Writing Data to Files
