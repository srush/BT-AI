# # Lab 1 - Python Basics

# The goal of this weeks lab is to work through the basics of python
# with a focus on the aspects that are important for datascience and
# machine learning.


# ![python](https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg)

# Python is pretty much the standard programming language for AI and
# machine learning these days. It is widely used in companies like Google
# or Netflix for their AI systems. It is also what is used in research
# and most non-profit organizations. This is lucky because it also
# one of the most fun programming languages!

# This week we will walkthrough the basics of Python and notebooks.


# * **Unit A**: Types and Documentation
# * **Unit B**: String and Functions

# # Unit A

# ## Working with types

# This summer we will be working with lots of different types of
# data. Sometimes that data wilil be numerical such as a temperature:

98.7

# Other times it will be a text string such as a name:

"New York City"

# More advanced cases will have lists of elements such as many names:

["Queens", "Brooklyn", "Manhattan", "Staten Island", "The Bronx"]

# ![NYC](data:image/jpeg;base64,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)

# Python has many different types like this. Knowing the type of your
# data is the first step.

# Let's look at some examples

# ### Numbers

# Numbers are the simplest type we will work with. Simply make a
# variable name and assign a number value.

my_number_variable = 80

my_number_variable

# Here are two more.

number1 = 10.5
number2 = 20

# Note: if you have learned a different programming language, such as
# Java, you might remember having to declare the type of a variables like

# ```java
# int number1 = 10
# ```

# You don't have to do that in Python. The type is still there, but it is
# added automatically.

# You can add two numbers together to make a

number3 = number1 + number2


# 👩‍🎓**Student Question: What value does ```number3``` have?**

#📝📝📝📝
pass

# ### Strings

# Strings are very easy to use in python. You can
# just use quotes to create them.

string1 = "New York "
string2 = "City"

# To combine two strings you simply add them together.

string3 = string1 + string2


# 👩‍🎓**Student Question: What value does ```string3``` have?**

#📝📝📝📝
pass

# ### Lists

# Python has a simple type for multiple values called a list.

list1 = [1, 2, 3]
list2 = [4, 5]

# Note: if you have learned a different programming language, such as
# Java, you might remember arrays. Python lists are like arrays but
# way easier. You don't need to declare their size or type. Just make
# them.

# Adding two lists together creates a new list combining the two.

list3 = list1 + list2

# 👩‍🎓**Student Question: What value does ```list3``` have?**

#📝📝📝📝
pass

# ## Dictionaries

# A dictionary type is used to link a "key" to a "value".
# You can have as many keys and values as you want, and they can
# be of most of the types that we have seen so far.

dict1 = {"apple": "red",
         "banana": "yellow"}
dict1

# To access a value of the dictionary, you use the square bracket notation
# with the key that you want to access.

dict1["apple"]


dict1["banana"]

# You can also add a new key to the dictionary by setting its value.

dict1["pear"] = "green"

# 👩‍🎓**Student Question: What value does ```dict1``` have?**

#📝📝📝📝 FILLME
pass

# ## Importing and Reading Docs

# Numbers, strings, and lists. These are the most common types of data
# that we will use throughout the summer. Every programmer should know
# these basic types.

# However to be a really good, it is important to be able to use types
# that you don't yet know. Most of the time the problem that you are
# interested will have a type that is already made by someone else.

# For instance, let's say we want a type for a date. We could try to write our own.

day = 8
month = "June"
year = 2021

# But there are so many things we would need to add! How do we represent weeks?
# Leap years? How do we count number of days?

# So instead let us use a package. To use a package first we `import` it.

import datetime

# Then we use `.` notation to use the package. This gives us a date variable for the current day.

date1 = datetime.datetime.now()
date1

# How did I know how to do this?

# Honestly, I had no idea. I completly forgot how this worked so I did this.

#
# 1. Google'd "how do i get the current time in python"
# 2. Clicked the link we get back here https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
#
# This is one of the most important skills to learn Python :)

# The format of the output of the line above is telling use the we can
# access the day and month of the current date in the following
# manner.

date1.day

date1.month

# 👩‍🎓**Student Question: Can you print the current value of the year**

#📝📝📝📝 FILLME
pass

# # Group Exercise A

# ## Question 1

# We saw that when we had a date type that it gave use the month as a number.

date1.month

# If we want to turn the months into more standard names we can
# do so by making a dictionary.


#📝📝📝📝 FILLME
#months = {
#    1 : "Jan",
#    ...
#}


# User your dictionary to convert the current month to a month name.

#📝📝📝📝 FILLME
pass

# ## Question 2

# One common data operations is to `count` unique items. For instance
# if we are voting we may have a list of votes from each person voting for
# candidate A, B, or C

votes = ["A", "B", "A", "A", "C", "C", "B", "A", "A", "A"]

# There is a special type in Python that makes this operation really easy known
# as a Counter. It lives in a package known as `collections`. We can use it
# by importing it like this

from collections import Counter

# For this exercise, you should google for how to use the Counter.
# Use what you find to print out the count of each number of votes that
# each candidate "A" received.

#📝📝📝📝 FILLME
pass

# ## Question 3

# Another useful aspect of the `Counter` is that it can tell you the most common
# elements in a list. This is particularly useful when there are a ton of different
# elements to work with. Google for how to find the most common element in a list.

# For this question, you will tell us the 10th most common letter in the beginning of the "Wizard of Oz".

wizard_of_oz = list("Dorothy lived in the midst of the great Kansas prairies, with Uncle Henry, who was a farmer, and Aunt Em, who was the farmer’s wife. Their house was small, for the lumber to build it had to be carried by wagon many miles. There were four walls, a floor and a roof, which made one room; and this room contained a rusty looking cookstove, a cupboard for the dishes, a table, three or four chairs, and the beds. Uncle Henry and Aunt Em had a big bed in one corner, and Dorothy a little bed in another corner. There was no garret at all, and no cellar—except a small hole dug in the ground, called a cyclone cellar, where the family could go in case one of those great whirlwinds arose, mighty enough to crush any building in its path. It was reached by a trap door in the middle of the floor, from which a ladder led down into the small, dark hole.")
wizard_of_oz

# Print out the 10'th most common element in this list.

#📝📝📝📝 FILLME
pass

# More than anything remember this. The best programmers use help the
# most! No one wins a prize for memerizing the most functions. If you
# want to be a good programmer, learn how to look things up quickly and
# ask the most questions.

# # Unit B

# In addition to the data types and libraries, we will sometimes
# use Python to write our own code. In general when doing data science
# you should not have to write very long amounts of code, but there are
# some cases when it is useful.

# ## Basic Structures

# ### `if` statements

# If statements check for a condition and run the
# code if it is true. In Python you need to indent
# the code under the if statement otherwise it will
# not run.

number3 = 10 + 75.0

if number3 > 50:
    print("number is greater than 50")


if number3 > 100:
    print("number is greater than 100")


# You can also have a backup `else` code block that will run if
# the condition is not true.

if number3 > 100:
    print("number is greater than 100")
else:
    print("number is not greater than 100")



# ### `for` loops

# For loops in python are used to step through the items in a list one by

list3

# You indicate a for loop in the following manner. The code will be run 5 times
# with the variable `value` taking on a new value each time through the loop.

for value in list3:
    print("Next value is: ", value)

# Note: unlike other languages Python for loops always need a list to
# walkthough. This differs from language where you have a counter variable.

# However, Python also includes a nice shortcut for making it easy to write for
# loops like this. The command `range` will make a list starting from
# a value and stop right before the end value.

for val in range(10):
    print(val)

# 👩‍🎓**Student Question: Print out each month name from your month dictionary.**

#📝📝📝📝 FILLME
for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    pass


# ## Working with Strings

# Strings are an importnat special case. Throughout this class we will
# work a lot with text. We will start with simple examples and work our
# way up to artificial intelligence over text.

# Text will also be represented with a string type. This is created with
# quotes.

str1 = "A sample string to get started"

# Just like with lists, we can make a for loop over strings to get individual letters.

for letter in str1:
    print(letter)

vowels = ["a", "e", "i", "o", "u"]
for letter in str1:
    if letter in vowels:
        print(letter)

# However, most of the time it will be better to use one of the built-in
# functions in Python. Most of the time it is best to google for these, but
# here are some important ones to remember

# * Split
# Splits a string up into a list of strings based on a separator

str1 = "a:b:c"
list_of_splits = str1.split(":")
list_of_splits[1]

# * Join
# Joins a string back together from a list.

str1 = ",".join(list_of_splits)

# 👩‍🎓**Student Question: What value does ```str1``` have?**

#📝📝📝📝 FILLME
pass

# * Replace
# Replaces some part of a string.

original_str = "Item 1 | Item 2 | Item 3"
new_str = original_str.replace("|", ",")
new_str

new_str = original_str.replace("|", "")
new_str

# * In
# Checks if one string contains another

original_str = "Item 1 | Item 2 | Item 3"
contains1 = "Item 2" in original_str

contains2 = "Item 4" in original_str

# 👩‍🎓**Student Question: What values do `contains1` and `contains2` have?**

#📝📝📝📝 FILLME
pass

# *  Conversions
# Converts between a string and a number

int1 = int("15")
int1

decimal1 = float("15.50")
decimal1

# ## Functions

# Functions are small snippets of code that you may want to use
# multiple times.

def add_man(str1):
    return str1 + "man"

out = add_man("bat")
out

# Most of the time, functions should not change the variables that
# are sent to them. For instance here we do not change the variable `y`.

y = "bat"
out = add_man(y)
out

y

# One interesting aspect of Python is that it lets your pass functions
# to functions. For instance, the built-in function `map` is a function
# applies another function to each element of a list.


# Assume we have a list like this.

word_list = ["spider", "bat", "super"]

# If we want a list with `man` added to each we cannot run the following:

# Doesn't work:  add_man(word_list)

# However, the map function makes this work, by creating a new list.

out = map(add_man, word_list)
out

# # Group Exercise B

# ## Question 1

# When processing real-world data it is very common to be given a
# complex string. that contains many different items all smashed together. 

real_word_string1 = "Sasha Rush,arush@cornell.edu,Roosevelt Island,NYC"

# Use one of the string functions above to pull out the email from
# this string and print it.

#📝📝📝📝 FILLME
pass

# ## Question 2

# Now assume the we have a list of strings.

real_word_strings2 = ["Sasha Rush,arush@cornell.edu,Roosevelt Island,NY",
                     "Bill Jones,bjones@cornell.edu,Manhattan,NY",
                     "Sarah Jones,sjones@cornell.edu,Queens,NY"]

# Write a for loop that does the following.

# * Steps through each string
# * Finds the email address
# * Prints out the email address

#📝📝📝📝 FILLME
pass

# ## Question 3

# Next we will assume that we have a list of strings where people come from
# different locations. Your goal is to step through the list and print out the
# emails of *only* the people who come from New York.

real_word_strings3 = ["Sasha Rush,arush@cornell.edu,Roosevelt Island,NY",
                      "Erica Zhou,ezhou@cornell.edu,Manhattan,NY",
                      "Jessica Peters,jpeters@cornell.edu,Miami,FL",
                      "Bill Jones,bjones@cornell.edu,Philadelpha,PA",
                      "Sarah Jones,sjones@cornell.edu,Queens,NY"]

#📝📝📝📝 FILLME
pass

# ## Question 4

# Finally lets assume that we want to create a new list of strings. We are going to do this
# by adding one more element to each string.

# Instead of "Sasha Rush,arush@cornell.edu,Roosevelt Island,NY"
# we want it to say =>
# "Sasha Rush,arush@cornell.edu,Roosevelt Island,NY,Computer Science" 

# Your task is to add this last element to each one of the strings

real_word_strings4 = ["Sasha Rush,arush@cornell.edu,Roosevelt Island,NY",
                      "Erica Zhou,ezhou@cornell.edu,Manhattan,NY",
                      "Jessica Peters,jpeters@cornell.edu,Miami,FL",
                      "Bill Jones,bjones@cornell.edu,Philadelpha,PA",
                      "Sarah Jones,sjones@cornell.edu,Queens,NY"]

#📝📝📝📝 FILLME
pass
