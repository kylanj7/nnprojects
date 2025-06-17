import numpy as np 

##1D Arrays

a = np.array([1,2,3])
print(a)

b = np.array9([9.0,8.0,7.0],[6.0,5.0,4.0])
print(b)

my_array = np.arange(8)
print(my_array)
print(type(my_array))

my_array(1, 8) 
print(my_array)

my_array(1,8,2.1) #(start, range, step) Can specify float and negative values
print(my_array)

from_list = np.array(1,2,3) # Arrays from list
print(from_list)
print(type(from_list))#integer that takes up 64 bits
#Boolean takes up much less space than a string
#Strings take up much more storage than booleans
from_list = np.array([1,2,3], dtype=np.int8) # convert array into binary to save data space
print(type(from_list[0]))

#2D Array

from_list = np.array([[1,2,3], [4,5,6]], dtype=np.int8)
array_2D = np.array((np.arrange(9,8,2), np.arange(1,8,2)))
array_2D = array_2D.reshape((4,2))

print(array_2D)
# print("1D shape: ", my_array.shape)
print("2D shape: ", array_2D.shape)

#Create Empty Arrays

empty_array = np.zeros(2,2)#Creates a 2x2 array filled with zeros
empty_array = np.ones((2,2))#creates a 2x2 arrays filled with 1's
empty_array = np.empty((2,2))# fills the array with RANDOM values
print(empty_array)

#NumPy Eye Function
#data stucture where the main diagonal is filled with ones while all the other toles are filled with zero'
#

eye_array = np.eye(3)
print(eye_array)

eye_array = np.eye(3, k=-1) #Sets the diagonal line 1 space below the conter of the array
print(eye_array)

eye_array = np.eye(3, k=1) #k value moves the diagonal line up or down based on the value set to k.
print(eye_array)
eye_array[eye_array == 0] = 2 #filtering command that sets all tiles that are not apart of the diagonal like to 2
print(eye_array)

eye_array[0] =3 # Filtering command that sets all tiles in the index row 0 to 3
print(eye_array)

eye_array[:2] = 3 #sets all tiles in the first 2 rows to the value indicated, in this case =3
print(eye_array)

eye_array[1:] = 3 #Starts at index row 1, skipping index row 0 and sets all of the values of after to that of the operator =
print(eye_array) 
#Slicing Columns
eye_array[:, 0] =4 # Selects all of the tiles associated with the first column and set the to the value set by the = operator
print(eye_array)

eye_array[:, -1] = -1 #-1 selects the very last column in the array. 

print(eye_array)

##Sorting Arrays##

print(eye_arraym, "\n")

sorted_array = np.sort(eye_array)
print(sorted_array)# sorts the array by rows in a sequence. First item is the smallest numer and the lsat item is the largest number. This is consitant across all the rows.

sorted_array = np.sort(eye_array, axis=0)
print(sorted_array) # does the same for operations as the privious code for columns instead of rows. 

sorted_array = np.sort(eye_array, axis=0, kind='mergesort') #spicify the algorythm that worts the array with "kind=heapsort" etc.
#different sorting algorythms mays result in faster calculations with larger arrays

##Copy Arrays##
my_view = sorted_array.view()
my_copy = sorted_array.copy()
#one affects the original array while the other does not

# Select all rows and assign them to 4
my_view[:] = 4
print(my_view, "\n")
print(sorted_array)

# the original array was untouched. 
my_copy[:] = 4
print(my_view, "\n")
print(sorted_array)

my_view = my_view.reshaped((3,3,1))
print(my_copy, "\n")
print(sorted_array)
