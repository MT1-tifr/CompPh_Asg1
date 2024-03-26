import numpy as  np

# Coefficient matrix for all 4 problems
A1 = np.array([
    [3, -1, 1],
    [3, 6, 2],
    [3, 3, 7]
])
A2 = np.array([
    [10, -1, 0],
    [-1, 10, -2],
    [0, -2, 10]
])
A3 = np.array([
    [10, 5, 0, 0],
    [5, 10, -4, 0],
    [0, -4, 8, -1],
    [0, 0, -1, 5]
])
A4 = np.array([
    [4, 1, 1, 0, 1],
    [-1, -3, 1, 1, 0],
    [2, 1, 5, -1, -1],
    [-1, -1, -1, 4, 0],
    [0, 2, -1, 1, 4]
])

# Corresponding Constant vectors for all 4 problems
b1 = np.array([1, 0, 4])
b2 = np.array([9, 7, 6])
b3 = np.array([6, 25, -11, -11])
b4 = np.array([6, 6, 6, 6, 6])

# Solving the system
x1 = np.linalg.solve(A1, b1)
x2 = np.linalg.solve(A2, b2)
x3 = np.linalg.solve(A3, b3)
x4 = np.linalg.solve(A4, b4)

print("Solution vector for first system:")
print(x1)
print("Solution vector for second system:")
print(x2)
print("Solution vector for third system:")
print(x3)
print("Solution vector for fourth system:")
print(x4)
