# # Lab 1 - Python Basics

# The goal of this lab is to work through the basics of python with
# a focus on the aspects that are important for datascience and machine learning.

# ## Working with types

# Unlike other languages that you may have worked with in the past
# Python does not make the user declare the "types" of variables
# (numbers, strings, classes). However, as a programmer it is important
# for you to know the differences and how they work.

# ### Numbers

# Numbers are the simplest type we will work with. Most of the time
# you can ignore the difference between integers and decimal types.
# Python will handle the conversions for you. 

number1 = 10.5
number2 = 20


number3 = number1 + number2
number3

# ### Strings

# Strings are very easy to use in python. You can
# just use quotes to create them. To combine two strings
# you simply add them together. 

string1 = "Hello "
string2 = "World"


string3 = string1 + string2
string3


# ### Lists

# Python has a simple type for multiple values called a list.
# This differs slightly from array types in other languages as you
# don't need to declare the size of the list. 


list1 = [1, 2, 3]
list2 = [4, 5]

# Adding two lists together creates a new list combining the two.

list3 = list1 + list2



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
dict1


# ## Control Structures

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
# with the variable `value` taking on a new value each time through the l100p.

for value in list3:
    print("Next value is: ", value)


# ## Importing and Reading Docs

# Python allows the user to specify their own types to represent
# additional properties.  We will use many other types throughout the
# class. To use these types we need to `import` them from libraries
# that store the code. 


# Here are a couple of examples.

# ## Counters

# First we add a line of code to important a new type into our program.
# it will often looks something like this.

import collect
# We can then use the type like this.

count = collections.Counter([1, 2, 1, 2, 1, 1, 1])
count

# How did we know that this function would count the items in a list?

# We didn't! We had to read the documentation. Mostly this means you
# go to Google and you type "how do I count the number of items in a
# list in python" You then click on the link from stackoverflow and
# some nice person tells you the answer. It won't always be the first
# answer but just keep trying until you find it.
# https://stackoverflow.com/questions/2161752/how-to-count-the-frequency-of-the-elements-in-an-unordered-list


# Another method is to print out the `help` method in your notebook.  

help(collections.Counter)


# This is a bit more complex, but it does tell us some more about how the Counter works. For instance it tells us how to get the most common element in the list.

help(count.most_common)


count.most_common(1)


# More than anything remember this. The best programmers use help the
# most! No one wins a prize for memerizing the most functions. If you
# want to be a good programmer, learn how to look things up quickly and
# ask the most questions. 

# ## Dates

# Another very common type that we will want to handle is the type for
# dates. I forgot how this works so let's Google it "how do i get the
# current time in python"


# The link we get back is here
# https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python

# It tells us we can do it like this,

import datetime
date1 = datetime.datetime.now()
date1


# The format of the output of the line above is telling use the we can
# access the day, month, and year of the current date in the following
# manner. Here `date1` is a special type but the day, month, and year
# are just standard numbers.

date1.day

date1.year

date1.month


# If we want to turn the months into more standard strings we can
# do so by making a dictionary.

months = {
    1 : "Jan",
    2 : "Feb",
    3 : "Mar",
    4 : "Apr",
    5 : "May",
    6 : "Jun",
    7 : "Jul",
    8 : "Aug",
    9 : "Sep",
    10: "Oct",
    11 : "Nov",
    12 : "Dec"
}
months


months[1]


# We can convert it to a month name through a lookup.

number_to_name_dict[date1.month]


# If we want to see all the months we walk through them with a `for` loop.

for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    print(months[month])


# Python also includes a nice shortcut for making it easy to write for
# loops like this. The command `range` will make a list starting from
# a value and stop right before the end value.

for month in range(1, 12 + 1):
    print(months[month])
    

# ## Working with Text

# Throughout this class we will work a lot with text.  First this will
# be just working with names, but it will quickly move to more complex
# text and evantually artificial intelligence over text. 


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
        
# ### Split
# Splits a string up into a list of strings based on a separator

str1 = "a:b:c"
list_of_splits = str1.split(":")
list_of_splits[1]


# ### Join
# Joins a string back together from a list.

str1 = ",".join(list_of_splits)
str1


# ### Replace
# Replaces some part of a string. 

original_str = "Item 1 | Item 2 | Item 3"
new_str = original_str.replace("|", ",")
new_str

new_str = original_str.replace("|", "")
new_str



# ### Contains
# Checks if one string contains another  

original_str = "Item 1 | Item 2 | Item 3"
contains1 = original_str.contains("Item 2")
contains1

contains2 = original_str.contains("Item 4")
contains2


# ### Conversions
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



# ## Exercises

teacher_str = "Sasha Rush,arush@cornell.edu,Roosevelt Island,NYC"


name, email, location, city =  teacher_str.split(",")


# Todo - Keyword args
