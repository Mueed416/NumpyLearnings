from matplotlib.pyplot import angle_spectrum
import numpy as np
import time
arr1d = np.array([1,3,4,2,8])
print(f"here is our array {arr1d}")

arr2d = np.array([[1,3,2,5], [5,3,5,3]])
print(f"our 2d array is {arr2d}")

# list vs numpy array

l1 = [1,3,4,2,6]
print('list into 2 =', l1 * 2)

arr1 = np.array([1,4,3,4,353,5,9])
print(arr1 * 2)

a = time.time()
print(a)

# creating array from scratch

# zeroes array

zero = np.zeros((3, 4))
print(zero)
# its output will be like 
"""
[0, 0, 0, 0]
[0, 0, 0, 0]
[0, 0, 0, 0]
[0, 0, 0, 0]
"""

# ones array

one = np.ones((3, 4))
print(one)
# output same like 0 but a 1 instead of 0

####### CxD array###################




# to set custom

custom = np.full((3, 4), 8)
print(custom)  #output same like 0 and 1s but as 8 instead of 0 and 1



# random arrays

randm = np.random.random((3, 4))
print(randm) # will print same pattern random numbers 
# its a whole in itself so has a lotta attributes
# always between 0 and 1


# sequence array 
seq = np.arange(0,11,2)
print(seq)
# sequent array


# array # matrix # tensor

vector = np.array([1,4,5,7])
print(vector)
matrix = np.array([[3,6,9],
                   [1,4,7],
                   [2,5,8]])
print(matrix)
# vector is 1d and matrix is 2d tensor is 3d
tensor = np.array([[[4,7], [1,4],
                   [8,7], [9,4]]])
print(tensor)



# array properties
# shape .shape [shows rows and columns quantity]
#dimension .ndim [shows dimension of array]
#size .size [shows total elements in array]
#datatype .dtype [shows datatype of array]
#reshape .reshape [reshape array]
#flatten .flatten [flatten array] like from [1,2],
                                           # [3,4] to [1,2,3,4]
#reshape .reshape [vice versa of flatten]
#ravel .ravel [same as flatten]
#transpose .T [transpose array] converts [1,2,3],
 #                                       [4,5,6] to [1 4]
                      #                             [2 5]
                        #                           [3 6]
#sort .sort() to sort array
  # 2 axis (0 and 1) 0= according to row 1= according to column
#vstack .vstack() to add or append row
arrr = np.array([[1,2,3],
   [4,5,6]])
print(arrr.shape)
print(arrr.ndim)
print(arrr.size)
print(arrr.dtype)

# array reshaping ###

arr2 = np.arange(12)
print(arr2)
# output [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(arrr.flatten())

# numpy array operations

# slicing ... already understood in lists
arr7 = np.arange(11)
print(arr7[3:7])
# negetive indexing 
# will start backward counting


#specific array element
adarray = np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])
print(adarray[0, 1])
# select parameter as [rows, clolumns]
print(adarray[1]) # print whole 2nd row
print(adarray[:, 1]) # print 2,5,8



# sorting

print(np.sort(adarray))


# sort with axis
zy = np.array([[1, 6, 5],
   [4, 5, 6],
   [7, 6, 4]])
print(np.sort(zy, axis=0))
print(np.sort(zy, axis=1))
# axis 0 sorts row wise... axis 1 = column



#filter
even = zy[zy % 2 == 0]
print(even)
odd = zy[zy % 2 == 1]
print(odd)
# even and odd filter



# filter with mask
# to mask
mask = zy > 5
print("number greater than 5", zy[mask])



# fancy indexing vs .npwhere()

# where [attr]
x = np.where(zy > 5)
print(zy[x])
# to find numbers in  array > 5


# array compatibility
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([7,8,9])

print("compability", a.shape == b.shape) #true
# to check if both arrays are compatible or not(like same number of rows and columns)
d = np.array([1,2,3])
e = np.array([4,5,6,9])
f = np.array([7,8,7,9])

print("compability", d.shape == e.shape) # false


# vstack
ogrow = np.array([[1,2], [3,4]])
newrow = np.array([5,6])
final = np.vstack((ogrow, newrow))
print(final)
# makes a new array with appended row

# .hstack 
a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])

result = np.hstack((a, b))

print(result)



# to delete index from array 
arr = np.array([1,2,3,4,5,6,7,8,9])
oe = np.delete(arr, 3)
print(oe)





# working with real world data (main)
#check for
# (numpy/realdata.py)



###
#
vector1 = np.array([1,2,3])
vector2 = np.array([4,5,6])
print("vector addition", vector1 + vector2)


# dot multiplication .dot and .accros
print("dot product", np.dot(vector1, vector2))

"""
vector1:   [1   2   3]
vector2: • [4   5   6]
           ------------
           4 + 10 + 18 = 32
"""

# to get angle of vector

angle = np.arccos(np.dot(vector1, vector2) / np.linalg.norm((vector1) * np.linalg.norm(vector2)))
print(angle)


# vectorizing

cars = np.array(['LC300', 'LEXUS LX', 'M5 COMPETITION', 'BMW M4'])
uppervector = np.vectorize(str.lower)
print(uppervector(cars))
#makes normal python operations vectorized like use them like vectorized functions