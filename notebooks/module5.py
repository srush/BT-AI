# # Lab 5 - Machine Learning 2 - Features

# This lab will continue our journey through applied machine learning.
# The main goal of the lab is to gain intuition for *features* and how
# better features allow us to capture more complex properties. 

# We will learn how features can be constructed for structured data
# and how they allow us to use simple models to make complex
# predictions.

# We will also return to our climate change data and learn how to use
# past data to make future predictions.

# ![](https://www.climate.gov/sites/default/files/styles/featured-image/public/clmdiv_avg_temp_February_2014_620.png?itok=74iCFfGv)


# # Review

# We have learned 3 libraries so far. Pandas, Altair, and
# Scikit-Learn. We will use all three heavily today.

import pandas as pd
import altair as alt
import sklearn.linear_model

# Reminder that our main sample dataset is a Red versus Blue classification
# challenge.

df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/simple.csv")
df

# Machine learning data is always split into at least two groups.
# We get to look at the *train* data to *fit* our model. We then
# use the the *test* data to evaluate our model.

df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]


# We will focus on *classification*. Each data point has a *class* that we are trying to predict.

df_train["class"]

# A point graph allows us to see our problem clearly. We need to separate the Reds from the Blues.

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart


# We can try this ourself by splitting the points doing it with one feature is not good enough.

def my_predict_feature1(point):
    if point["feature1"] > 0.5:
        return "red"
    else:
        return "blue"


df_test["predict"] = df_test.apply(my_predict_feature1, axis=1)


chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
        fill = "predict:N"
    ))
chart

# But if we base it on two features we can get it right.

def my_predict_linear(point):
    if point["feature1"] + point["feature2"] < 1.0:
        return "blue"
    else:
        return "red"

df_test["predict"] = df_test.apply(my_predict_linear, axis=1)

# Notice how our cutoff used an addition between our two features. This
# allows us to form any line as the cutoff between the colors.

chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
        fill = "predict:N"
    ))
chart

# We are using the library *scikit-learn* for prediction. We are
# focusing on a method known as Linear Classification. (Note it has a
# silly name in the library, we rename it for simplicity.).

LinearClassification = sklearn.linear_model.LogisticRegression
model = LinearClassification()

#  The main machine learning step is to fit our model to the features of the data.

model.fit(X=df_train[["feature1", "feature2"]],
          y=df_train["class"] == "red")


# Linear Classification draws the best line to split the two classes.
# That is it tries out different combinations of adding together features
# and cutoffs to find the best one.
# (For now you can ignore how it does that.) 

# We can see this visually by trying out out every possible point and seeing where the system puts them.

all_df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/all_points.csv")
all_df["predict"] = model.predict(all_df[["feature1", "feature2"]])

# Here is a graph of out the points.

chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart


# The goal of machine learning is to accurately predict the results of
# unseen data. We use the test data to make this possible.

# ## Review Exercise

# Use the model to predict on the test data (using `model.predict`).
# What is the accuracy (number of points where the *prediction* is the same as the true  *class*)?

#📝📝📝📝 FILLME
pass

# Can you graph the test data showing both the prediction and the true class?


#📝📝📝📝 FILLME
pass

# # Unit A

# ## More Complex Data 

# Our model is effective at drawing a line between data. However as we saw last time
# this only works if our data can be split easily with a line.

# Let us load some more complex data. 


df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/complex.csv")
df_train = df[df["split"] == "train"]

# This data has all the red points in a the top-right corner. This makes it hard
# to separate.

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart


# 👩‍🎓**Student question: Can you split the data yourself?**

#📝📝📝📝 FILLME
def my_predict(point):
    if True:
        return "blue"
    else:
        return "red"
    
df_test["predict"] = df_test.apply(my_predict, axis=1)



# Let us now try to fit out data in the standard way.

model.fit(X=df_train[["feature1", "feature2"]],
          y=df_train["class"] == "red")

# We can then predict on the train set to see how well it fit the data.

df_train["predict"] = model.predict(df_train[["feature1", "feature2"]])
df_train["correct"] = df_train["predict"] == (df_train["class"] == "red")

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
        fill = "predict",
        shape = "correct",
    ))
chart


# Unfortunately we can see that it did not fit well. 

# We can see that it did its best to try to find a line. However, it did
# not succeed in practice. 

all_df["predict"] = model.predict(all_df[["feature1", "feature2"]])
chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="predict",
        fill = "predict",
    ))
chart

# ## Reasoning about Seperation

# Let us think about how we would seperate this data manually.

# Remember that our main tool was to create filters that let use divide up
# the points.

# The upper right corner can be defined as the *and* of the right side
# and the upper side of our data. 

filter = (df_train["feature1"] > 0.5) & (df_train["feature2"] > 0.5)
df_red = df_train.loc[filter]

# Here is what it looks like.

chart = (alt.Chart(df_red)
    .mark_point(color="orange")
    .encode(
        x = "feature1",
        y = "feature2",
    ))
chart


# Conversely we separate the blue pts by the *or* of the the left side
# and the bottom side.

filter = (df_train["feature1"] < 0.5) | (df_train["feature2"] < 0.5)
df_blue = df_train.loc[filter]


chart = (alt.Chart(df_blue)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
    ))
chart

# ## Features

# We can create features as new functions that transform our data. Here we note that the
# feature should be aware of where the middle of the data is.

def mkupperfeature(f):
    if f <= 0.5:
        return -1.0
    else:
        return f 

# We can use standard pandas methods to create these new features.
    
df_train["feature3"] = df_train["feature1"].map(mkupperfeature)
df_train["feature4"] = df_train["feature2"].map(mkupperfeature)


# Now we can fit our model. *Note* we are passing feature 3 and 4, not 1 and 2.

model.fit(X=df_train[["feature3", "feature4"]],
          y=df_train["class"] == "red")
df_train["predict"] = model.predict(df_train[["feature3", "feature4"]])

# Now it works!

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
        fill = "predict",
    ))
chart


# Let's repeat that and look what happened. We first added new features
# that tell the model about the up/down and left/right regions.

all_df["feature3"] = all_df["feature1"].map(mkupperfeature)
all_df["feature4"] = all_df["feature2"].map(mkupperfeature)

# We then predict on these new features. 

all_df["predict"] = model.predict(all_df[["feature3", "feature4"]])

# And when we graph on the original features, it can draw fancy shapes. 

chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "predict",
        fill = "predict",
    ))
chart


# Why were able to draw a line that had a square shape?

# The key trick is that the line is straight in the features that we gave the model. Features 3 and 4.
# If we graph them it looks like this. 

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature3:Q",
        y = "feature4:Q",
        color="class",
        fill = "predict",
    ))
chart

# The key idea is that when we train on the new features. We can be
# *non-linear* in the original features.




# # Group Exercise A

# For this exercise we will try to learn classifiers for the following data sets.
# This will let us experiment with different features and their usefulness.

# ## Question 0

# Who are other members of your group today?

#📝📝📝📝 FILLME
pass

# Do they prefer cats or dogs?

#📝📝📝📝 FILLME
pass

# Do they prefer summer or winter?

#📝📝📝📝 FILLME
pass

# Do they prefer sweet or savoury breakfast?

#📝📝📝📝 FILLME
pass




# ## Question 1

# For this first question, the data is separated into two parts based on
# how far it is from the center

df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/center.csv")
df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]
chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart


# Construct new features that allow you to separate the data into parts. 

#📝📝📝📝 FILLME
pass

# ## Question 2

# Now the data is split into a circle around the point (0.4, 0.4).


df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/circle.csv")
df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]


chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# Use the formula for distance from a point $\sqrt{(x - 0.4)^2 + (y - 0.4)^2}$ to define new features for this problem.

#📝📝📝📝 FILLME
pass

# ## Question 3

# Finally consider a problem with a periodic (repeating) curve.

df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/periodic.csv")
df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]
chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# Define a new feature based on the formula $\sin(\mathrm{feature1}*10)$. (Sin is in the math library).

import math
math.sin

# Use this function to create a new feature (feature 3). Show that training with
# feature 1,2, and 3 will yield an accurate value


#📝📝📝📝 FILLME
pass


# # Unit B

# ## Real World Data


# Now we will apply the methods we learned in the previous section to a real-world
# classification problem. We will return to the temperature classification problem
# from Week 3

temp_df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/Temperatures.csv", index_col=0, parse_dates=[1])

# We convert the tempertature to Farenheit for easier visualization.

temp_df["Temp"] = (temp_df["Temperature"] * 9/5) + 32


# To make things simpler, we filter to the US in the years 2000 and 2001

filter = ((temp_df["Country"] == "United States") &
          (temp_df["Date"].dt.year >= 2000) & (temp_df["Date"].dt.year <= 2001))
df = temp_df.loc[filter]

# Just as a a reminder here are our columns.

df.columns

# ## Example: Predicting Temperature

# We will start with the question of predicting when the temperature goes over 70 degrees.

df["class"] = df["Temp"] > 70

# To make our splits we divide the data arbitrarily into a train and test set.

def mksplit(city):
    if city[0] > "M":
        return "test"
    else:
        return "train"
df["split"] = df["City"].map(mksplit)


# The main question of interest will be "Which features should we use?". We want features
# that best separate the data into warm and cold groups.

# ### Try 1

# A natural first attempt is to use location features. We can use the Longitude (east-west) and Latitude (north-south) to start.

df["feature1"] = df["Longitude"]
df["feature2"] = df["Latitude"]

# We then split the data into train and test sets.

df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]

# The different classes are True and False, i.e. is it over 70 for
# that city in that month.

df_train["class"]


# Now we follow our standard method to fit the data and see how well it does
# at separating on the train set. 

model.fit(X=df_train[["feature1", "feature2"]],
          y=df_train["class"]
)
df_train["predict"] = model.predict(df_train[["feature1", "feature2"]])


# Unfortunately the model does not do that well at all!

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color="class",
        fill = "predict",
        tooltip = "City"
    ))
chart

# ## Try 2

# Let us look at the data to see what is happening. Why is it not enough to just look at Longitude and Latitude?

chart = (alt.Chart(df_train)
    .mark_tick(thickness=4)
    .encode(
        x = "City",
        y = "Temp",
        color="class",
        fill = "class",
        tooltip = "City",
        shape="City"
    ))
chart

# The problem seems to be seasons. The same city will have very different average temperatures in different months.

# We can see this by looking at points grouped by latitude. 

chart = alt.Chart(df).mark_point().encode(
    y = "Temp",
    x = "Latitude",
    tooltip=["City", "Country"],
)
chart

# Latitude is clearly a useful feature, lower latitude means higher temperature.
# But it is not enough alone. We need to be able to separate out by season as
# well.

chart = alt.Chart(df).mark_point().encode(
    y = "Temp",
    x = "Date:T",
    color = "class", 
    tooltip=["City", "Country"],
)
chart

# This feature turns the month of the year into a value between -1 and 1.
# Summer is 1 and winter is 0.

def mkfeature(time):
    return -math.cos(((time.month - 1) / 11.0) * 2 * math.pi)

# We can include this feature with latitude.

df["feature3"] = df["Date"].map(mkfeature)
df["feature4"] = df["Latitude"]

# Apply the standard ML steps.

df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]
model.fit(X=df_train[["feature3", "feature4"]],
          y=df_train["class"]
)
df_train["predict"] = model.predict(df_train[["feature3", "feature4"]])


# And now we can plot our graph. Here the x axis is the seasonal value and the y axis
# is the latitude. Mostly the model is able to now predict correctly.

chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature3",
        y = "feature4",
        color="class",
        fill = "predict",
        tooltip = "City"
    ))
chart

# Checking on the test data we can see that the model use both features to make its predictions.

df_test["predict"] = model.predict(df_test[["feature3", "feature4"]])

chart = alt.Chart(df_test).mark_point().encode(
    y = "Temp",
    x = "Latitude",
    color="class",
    fill = "predict",

    tooltip=["City", "Country"],
)
chart


# ## Pretty Chart

# Just for fun we can chart this data on a map.

from vega_datasets import data
us_cities_df = df.loc[df["Country"] == "United States"]

# We just look at one month in the data.

df["predict"] = model.predict(df[["feature3", "feature4"]])
df_summer = df[(df["Date"].dt.month==4) & (df["Date"].dt.year==2001)]

# Draw the background map.

states = alt.topo_feature(data.us_10m.url, feature='states')
background = alt.Chart(states).mark_geoshape(
    fill='lightgray',
    stroke='white'
).properties(
    width=500,
    height=300
).project('albersUsa')


# Plot our points, predictions, and the true answer on the map.

points = alt.Chart(df_summer).mark_point(size=100).encode(
    longitude='Longitude',
    latitude='Latitude',
    fill="Temp",
    color="predict",
    shape="class",
    tooltip=['City', "Temp"]
)
chart = background + points
chart

# ## Example 2: Adding Other Countries


# For the group activity today, you will extend these ideas to other
# countries. In particular we will consider 3 others. 

filter = (((temp_df["Country"] == "Canada") |
           (temp_df["Country"] == "France") |
           (temp_df["Country"] == "Brazil")) &
          ((temp_df["Date"].dt.year == 2000) & (temp_df["Date"].dt.month==7)))
df = temp_df.loc[filter]

df

# Classification and features are the same.

df["class"] = df["Temp"] > 70
df["feature3"] = df["Date"].map(mkfeature)
df["feature4"] = df["Latitude"]

# The model and predictions are the same. 

model.fit(X=df[["feature3", "feature4"]],
          y=df["class"]
)
df["predict"] = model.predict(df[["feature3", "feature4"]])


# However, suddenly this approach does not work!

chart = alt.Chart(df).mark_point().encode(
    y = "Temp",
    x = "Latitude",
    color="class",
    fill = "predict",

    tooltip=["City", "Country"],
)
chart


# # Group Exercise B

# For this group exercise you will puzzle through what is going wrong when we apply the approach
# to other countries on the map.

# ## Question 1

# The model predicts the cities in Brazil mostly incorrectly. What is going wrong with this approach?

#📝📝📝📝 FILLME

# ## Question 2

# Modify one of the features in the model to help it correctly predict the Brazil cities. 

#📝📝📝📝 FILLME
pass

# ## Question 3

# Now consider a comparison of just France and Canada

filter = (((temp_df["Country"] == "France") |
           (temp_df["Country"] == "Canada")) &
          ((temp_df["Date"].dt.year == 2000) & (temp_df["Date"].dt.month==1)))
df = temp_df.loc[filter]
df["class"] = df["Temp"] > 25


# Surprisingly France is at a very northern Latitude, almost as high as Canada.

# Draw a chart like above that shows what happens when you try to run our
# standard model on this data. 


#📝📝📝📝 FILLME
pass


# Propose a different feature and add it to the model to mostly correctly classify this
# dataset. In particular, how does the classifier split France and Canada?


#📝📝📝📝 FILLME
pass

