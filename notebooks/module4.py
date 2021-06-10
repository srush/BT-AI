# # Lab 4 - Machine Learning!

# This lab will introduce the basic concepts behind machine learning
# and the tools that allow us to learn from data. Machine learning is
# one of the main topics in modern AI and is used for many exciting
# applications that we will see in the coming weeks.


# ![scikit](https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)

# # Review

# So far the two libraries that we have covered are Pandas which handles data frames.

import pandas as pd

# And Altair which handles data visualization. 

import altair as alt

# We will continue building on these two libraries throughout the semester.

# Let's start from a simple dataframe. We have been mainly loading
# data frames from files, but we can also create them directly.

df = pd.DataFrame({
    "City" : ["New York", "Philadelphia", "Boston"],
    "Temperature" : [25.3, 30.1, 22.1],
    "Location" : [20.0, 15.0, 25.0],
    "Population" : [10000000, 10000000, 500000],
})
df

# Here are our columns.

df.columns

# We can make a graph by converting our dataframe into a `Chart`.
# Remember we do this in three steps.

# **Charting**
#
# 1. Chart - Convert a dataframe to a chart
# 2. Mark - Determine which type of chart we want
# 3. Encode - Say which columns correspond to which dimensions

# One example is a bar chart. 

chart = (alt.Chart(df)
           .mark_bar()
           .encode(x = "City",
                   y = "Population"))
chart


# Notice that we didn't have to use all the columns and it only showed the ones we specified.

# Another example is a chart that shows the location and the temperature. 

chart = (alt.Chart(df)
           .mark_point()
           .encode(x = "Location",
                   y = "Temperature"))
chart


# The library allows us to add special features. For instance, we can add a "Tooltip" where are mouse tells us which city it is. 


chart = (alt.Chart(df)
           .mark_point()
           .encode(x = "Location",
                   y = "Temperature",
                   tooltip = "City"
           ))
chart


# ## Review Exercise 

# Make a bar chart that shows each city with its temperature and a tooltip of the city name. 

#📝📝📝📝 FILLME
pass

# # Unit A

# For today's class we are going to take a break from temperature and
# look at some simplified starter dataset. 

# Our dataset is a Red versus Blue classification challenge. Instead of
# describing this dataset let us take a look.


df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/simple.csv")
df

# The first thing to do is to look at the columns.

df.columns


# First is "split". The two options here are `train` and `test`. This is an important 
# distinction in machine learning. 
#
# * Train -> Points that we use to "fit" our machine learning model
# * Test ->  Points that we use to "predict" with our machine learning model.

# For example, if we were building a model for classifying types of birds from images, 
# our Train split might be pictures of birds from a guide, whereas our Test split
# would be new pictures of birds in the wild that we want to classify. 

# Let us separate these out. using a filter.

df_train = df.loc[df["split"] == "train"]
df_test = df.loc[df["split"] == "test"]


# Next is "class". We can see there are two options, `red` and `blue`.
# This tells us the color associated with the point. For this exercise,
# our goal is going to be splitting up these two colors.  

df_train["class"].sum()


# Finally we have "features". Features are the columns that we use
# in order to solve the challenge. The machine learning model gets to
# use the features in any way it wants it order to predict the class.


# Let us now put everything together to draw a graph. 

# **Charting**
#
# 1. Chart - Just our training split.
# 2. Mark - Point mark to show each row
# 3. Encode - The features and the class.


chart = (alt.Chart(df_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# We can see that for this example the colors are split into
# two sides of the chart. Blue is at the bottom-left and red is
# at the top-right. 

# We can also look at the test split. The test split consists of
# the additional challenge points that our model needs to get correct.
# These points will follow a similar pattern, but have different features. 

chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class"
    ))
chart

# We are interested in using the features to predict the class (red/blue).
# We can do this by writing a function.

def predict(point):
    if point["feature1"] > 0.5:
        return "red"
    else:
        return "blue"

# We can apply this function using a variant of `map` from Module 1.
# The `apply` command will call our prediction for each point in test.
    
df_test["predict"] = df_test.apply(predict, axis=1)


# Once we have made predictions, we can compute a score for how well
# our prediction did. We do this by comparing the `predict` with `class`.

correct = (df_test["predict"] ==  df_test["class"])
df_test["correct"] = correct


# Let us see how well we did. This graph puts everything together. 

chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
        fill = "predict",
        tooltip = ["correct"]
    ))
chart

# The outline of the point is blue / red based on the true class. Whereas the fill tells us
# our prediction. Mousing over the points will tell us whether they are correct or not. 

# 👩‍🎓**Student question: How well did our predictions do?**

#📝📝📝📝 FILLME
pass

# # Group Exercise A

# ## Question 1

# The `predict` function above is not able to fully separate the points into red/blue groups.
# Can you write a new function that gets all of the points correct? 

#📝📝📝📝 FILLME
def my_predict(point):
    pass
df_test["predict"] = df_test.apply(predict, axis=1)

# Redraw the graph above to show that you split up the points correctly.

#📝📝📝📝 FILLME
chart = ()
chart

# ## Question 2

# The dataset above is a bit easy. It seems like you can just
# separate the points with a line. 

# Next let us consider a harder example where the red and blue points form a circle.


df2 = pd.read_csv("https://srush.github.io/BT-AI/notebooks/circle.csv")

df2_train = df2.loc[df2["split"] == "train"]
df2_test = df2.loc[df2["split"] == "test"]

# Draw a chart with these points. 

#📝📝📝📝 FILLME
chart = ()
chart


# ## Question 3

# Try to write a function that separates the blue and the red
# points. How well can you do?


#📝📝📝📝 FILLME
def my_circle_predict(point):
    pass
df2_test["predict"] = df2_test.apply(predict, axis=1)


# Redraw the graph above to show that you split up the points correctly.

#📝📝📝📝 FILLME
chart = ()
chart

# # Unit B

# In Unit A we wrote a function to try to split the red and the blue
# data points.

# Machine learning (ML) allows us to produce that function without having
# to write it manually.

# The library Scikit-Learn is a standard toolkit for machine learning in Python. 

# ![sklearn](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)

# One warning. The documentation for Scikit-Learn is a bit intimidating. If you look something
# up it might appear like this. 

# https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

# Do not be scared though. Most of these options do not matter so much in practice. You can
# learn the important parts in 30 minutes. 

# Let us first important the library.

import sklearn.linear_model


# We are going to use this formula for all our machine learning. 

# **Model Fitting**
#
# 1. Dataframe. Create your training data (This part you are an expert in!)
# 2. Fit. Create a model and give it training features
# 3. Predict. Use the model on test data.

# Step 1. Create out data. (We did this already.

df_train

# Step 2. Create our model and fit it to data. 

# First we pick a model type. We will mostly use this one. (Don't worry about the name for now!)

model = sklearn.linear_model.LogisticRegression()

# Then we tell it which features to use as input (X) and what it goal
# is (y). Here we tell it to use `feature1` and `feature2` and to
# predict whether the point is red.

model.fit(X=df_train[["feature1", "feature2"]],
          y=df_train["class"] == "red")

# This is similar to Altair chart. Just tell it which columns to use.  

# Step 3. Predict. Once we have a model we can use it to predict the
# output classes of our model. This replaces the part where we did it
# manually.

df_test["predict"] = model.predict(df_test[["feature1", "feature2"]])

# We can see the graph that came out.

chart = (alt.Chart(df_test)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
        fill = "predict",
        tooltip = ["correct"]
    ))
chart


# That's it! You have done machine learning.

# ## Details

# What happened? How did the system know whether the output points
# would be red or blue? 

# The key idea is that behind the scenes the model uses the training data
# to learn a class for every possible point.

# For instance, if we make up a feature value.

feature1 = 0.2
feature2 = 0.5

# Our model will produce an output prediction.

predict = model.predict([[feature1, feature2]])
predict


# In fact, we can even see what the model would do for any point.


# This dataframe has most of the possible points.


all_df = pd.read_csv("https://srush.github.io/BT-AI/notebooks/all_points.csv")
chart = (alt.Chart(all_df)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
    ))
chart


# Let us see what our model would do on each of them.

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

# This makes sense. 

chart2 = (alt.Chart(df_test)
    .mark_point(color="black")
    .encode(
        x = "feature1",
        y = "feature2",
    ))
chart = chart + chart2
chart


# ## Other Data.

# So is machine learning magic? Can we just give any data
# and have it learn a separator for us?

# Well let's try the circle dataset.

chart = (alt.Chart(df2_train)
    .mark_point()
    .encode(
        x = "feature1",
        y = "feature2",
        color = "class",
    ))
chart


# First we fit. 

model.fit(X=df2_train[["feature1", "feature2"]],
          y=df2_train["class"] == "red")

# Then we predict.

df2_test["predict"] = model.predict(df2_test[["feature1", "feature2"]])


df2_test

# Finally we graph.

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


# We can put in any value to get a score



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



