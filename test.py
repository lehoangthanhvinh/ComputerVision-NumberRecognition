import numpy as np

array = np.array([[3, 2, 3], [4, 7, 7], [7, 8, 9]])
array = np.array([i == i.max() for i in array])

print(array)
print('-------------------------')
for arr in array:
    if arr.sum() > 1:
        f = next(i for i, a in enumerate(arr) if a)
        for i in range(len(arr)):
            arr[i] = False if i > f else arr[i]
print(array)
