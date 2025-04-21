import numpy as np

arr = np.array([[1,2],[3,4]])
arr2 =  np.array([[3,4],[5,6]])

print(np.linalg.norm(arr - arr2, ord='fro'))
print(np.linalg.norm(arr - arr2, ord=1))
