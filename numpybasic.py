
#Use different python packages to perform numerical calculations, statistical computations and data visualization.



import numpy as np
array = np.array([3, 2, 1, 2])
print("Original array: ", array)
print("Append (6,7,8): ",np.append(array, [6, 7, 8]))
print("Insert Specific (10,11) at third second position: ", np.insert(array, 2, [10, 11]))
print("Delete values (1,3): ", np.delete(array, [0, 2]))
print("Unique element: ", np.unique(array))
print("Sorted array: ", np.sort(array))
np.savetxt('array.txt', array)
load = np.loadtxt('array.txt')
print("Loaded from array.txt: ", load)