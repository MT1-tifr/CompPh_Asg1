import numpy as np
import pygsl
from pygsl import linalg

# Define the four matrices
matrices = [np.array([[3, -1, 1], [3, 6, 2], [3, 3, 7]], dtype=float),
            np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]], dtype=float),
            np.array([[10, 5, 0, 0], [5, 10, -4, 0], [0, -4, 8, -1], [0, 0, -1, 5]], dtype=float),
            np.array([[4, 1, 1, 0, 1], [-1, -3, 1, 1, 0], [2, 1, 5, -1, -1],
               [-1, -1, -1, 4, 0], [0, 2, -1, 1, 4]], dtype=float)
            ]

# Performing LU decomposition for each matrix
for i, matrix in enumerate(matrices):
    lu, p, signum = linalg.LU_decomp(matrix)

    # Reconstructing the original matrix using LU decomposition
    reconstructed_matrix = np.dot(lu, p)

    # Checking if the reconstructed matrix is equal to the original matrix
    if np.allclose(reconstructed_matrix, matrix):
        print(f"Matrix {i + 1}: LU decomposition is correct.")
    else:
        print(f"Matrix {i + 1}: LU decomposition is incorrect.")
