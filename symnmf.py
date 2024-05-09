import sys
import math
import numpy as np
import pandas as pd
from numpy.linalg import matrix_power

MAX_ITER = 200
EPSILON = 0.001
BETA = 0.5
GOAL_PARAMS = ['symnmf', 'sym', 'ddg', 'norm']


def main():

    if len(sys.argv != 4):
        print("please enter all 3 arguments")
        exit()

    np.random.seed(0)

    # STEP a:

    # number of required clusters
    k = convert_to_number(sys.argv[1])
    
    # goal can be: (symnmf | sym | ddg | norm)
    goal = sys.argv[2]
    assert goal in GOAL_PARAMS, "goal is incorrect! can be only: 'symnmf', 'sym', 'ddg', 'norm'"
    
    file_name = sys.argv[3]
    data_table = pd.read_csv(file_name, header=None)
    X = data_table.to_numpy()

    # STEP b:
    # STEP 1.1 - The Similarity Matrix:
    A = get_similarity_matrix(X)

    # STEP 1.2 - The diagonal degree Matrix:
    D = get_diagonal_matrix(A)

    # STEP 1.3 - The normalized similarity matrix:
    tmp = matrix_power(D, -0.5)  # TODO: rename tmp variable
    W = tmp * A * tmp

    # STEP 1.4 - Algorithm for optimizing H:

    # STEP 1.4.1 - Initialize H:
    m = np.mean(W)
    H = np.full_like(W, fill_value=np.random.uniform(0, 2 * math.sqrt(m / k)))

    # STEP 1.4.2 - Update H:
    diff = 1
    iter = 0
    while diff >= EPSILON and iter <= MAX_ITER:
        next_H = H * (1 - BETA + BETA*((W*H) / (H*H.transpose()*H)))

        diff = euclidean_distance(next_H, H)

    # STEP 1.5 - Deriving a clustering solution
    vectors, clusters = np.shape(H)
    # for vector in vectors

def get_diagonal_matrix(A):
    rows, columns = np.shape(A)
    D = np.empty_like(A)

    for i in rows:
        for j in columns:

            if i != j:
                D[i][j] = 0

            else:
                D[i][j] = np.sum(A, axis=0)

    return D


def get_similarity_matrix(X):
    rows, columns = np.shape(X)
    A = np.empty_like(X)

    for i in rows:
        for j in columns:

            if i == j:
                A[i][j] = 0

            else:
                A[i][j] = math.exp(-(euclidean_distance(X[i], X[j]) ^ 2) / 2)

    return A


def euclidean_distance(vector_x, vector_y):
    sum = 0
    for i in range(vector_x):
        sum += (vector_x[i] - vector_y[i]) ^ 2
    return math.sqrt(sum)

# Function to convert a string to int
def convert_to_number(str):
    try:
        return int(str)
    except ValueError:
        print(str + " is not a whole number!")
        print("Check your arguments again")
        exit()

if __name__ == "__main__":
    main()
