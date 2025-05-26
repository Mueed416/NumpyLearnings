# DATA OF TOP 5 RESTRAUNTS IN KASHMIR


import numpy as np
import matplotlib.pyplot as plt
car = np.array([
  ["Maruti Suzuki",   1400, 1500, 1600, 1700, 1800],
  ["Hyundai",         800,  850,  900,  950, 1000],
  ["Tata Motors",     600,  700,  850,  950, 1100],
  ["Mahindra",        500,  520,  580,  620, 700],
  ["Toyota",          400,  450,  500,  550, 600]
], dtype=object)

print("Cars24 sales analysis", car.shape)
print("sample data from cars24 `of 1st 3 companies\n", car[0:3])


# total sales per year

year = np.sum(car, axis=0)
print(year)

minsales = np.min(car[:, 1:], axis=1)
print(minsales)
maxsales = np.max(car[:, 1:], axis=1)
print(maxsales)
avgsales = np.mean(car[:, 1:], axis=1)
print(avgsales)

cumsum = np.cumsum(car[:, 1:], axis=0)
print(cumsum)


# now the plot side
plt.figure(figsize=(10, 6))
plt.plot(np.mean(cumsum), axis=0)
plt.title("cumulative sales of top 5 companies")
plt.xblable("years")
plt.ylable("sales")
plt.grid(True)
plt.show()


