# storing image into matrix and converting it into dark mode
import numpy as np
import matplotlib.pyplot as plt

arr = np.array([[1,2,3],[4,5,6]])
arr2 = np.random.rand(3, 3)
arr3 = np.zeros((3, 3))
np.save('numpy/arr.npy', arr)
np.save('numpy/arr2.npy', arr2) #to make specific files of numpy arrays
np.save('numpy/arr3.npy', arr3)

loadarr = np.load('numpy/arr.npy')
print(loadarr) #to view them

loadarr2 = np.load('numpy/arr2.npy')
print(loadarr2)


# change pixels to dark mode (white pixels to black)
# invert colours on color wheel
# try:
#   np.load('numpy/arr2.npy')
#   plt.figure(figsize = (10,6))
#   plt.subplot(121)
#   plt.imshow('numpy/arr2.npy')
#   plt.title('numpyy')
#   plt.grid(False)
# except FileNotFoundError:
#   print('numpy file not found')
#   # how to make it display above from line 23
# logo = np.load('numpy/dotaddition.jpg')
# # dark logo not avalable of repli so will be displayed as a ignore command xd
# darklogo = 1 - logo
# plt.subplot(122)
# plt.imshow(darklogo)
# plt.title('numpy dark logo')
# plt.grid(False)