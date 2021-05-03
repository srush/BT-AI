# # Module 1 - Python Basics

# ## Goal:

# ## Working with types


# Numbers

number1 = 10
number2 = 20


number3 = number1 + number2
number3

# Strings

string1 = "Hello "
string2 = "World"


string3 = string1 + string2
string3


# Lists

list1 = [1, 2, 3]
list2 = [4, 5]

list3 = list1 + list2
list3



# For loops

for value in list3:
    print("Next value is: ", value)


# Dictionaries

dict1 = {"name1": 10, "name2": 20}
dict1




# Special Types


# Dates

import datetime as dt

date1 = dt.datetime.now()


date1.day

date1.year

date1.month


number_to_name_dict = {
    1 : "Jan",
    2 : "Feb",
    3 : "Mar",
    4 : "Apr",
    5 : "May",
    6 : "Jun",
    7 : "Jul",
    8 : "Aug",
    9 : "Sep",
    10 : "Oct",
    11 : "Nov",
    12 : "Dec"
}


number_to_name_dict[date1.month]

# ## Working with strings

# Split


str1 = "a:b:c"
a_str, b_str, c_str = str1.split(":")
b_str


# Join

str_list = [a_str, b_str, c_str]
str1 = ",".join(str_list)
str1


# Replace

original_str = "Item 1 | Item 2 | Item 3"
new_str = original_str.replace("|", ",")
new_str

new_str = original_str.replace("|", "")
new_str



# Contains

original_str = "Item 1 | Item 2 | Item 3"
contains1 = original_str.contains("Item 2")
contains1

contains2 = original_str.contains("Item 4")
contains2


# Convert

int1 = int("15")
int1

decimal1 = float("15.50")
decimal1

# Exercise

teacher_str = "Sasha Rush,arush@cornell.edu,Roosevelt Island,NYC"


name, email, location, city =  teacher_str.split(",")




# ## Functions

def add_ten(x):
    return x + 10

out = add_ten(5)
out


y = 5
out = add_ten(y)
out

y


number_list = [5, 10, 15]
# Doesn't work:  add_ten(number_list)
out = map(add_ten, number_list)

for value in out:
    print("Next Value is: ", value)


# ## Dictionaries
