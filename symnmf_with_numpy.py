import sys
import math
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
import symnmf_capi

MAX_ITER = 300
EPSILON = 1e-4  # 0.0001
BETA = 0.5
GOAL_PARAMS = ['symnmf', 'sym', 'ddg', 'norm']


def main():

    if len(sys.argv) != 4:
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
    print('file_name: ' + file_name)
    data_table = pd.read_csv(file_name, header=None)
    print("data_table: \n" + str(data_table))
    X = data_table.to_numpy()
    N, d = np.shape(X)

    print("X: \n" + str(X))

    # STEP b:
    # STEP 1.1 - The Similarity Matrix:
    A = get_similarity_matrix(X)

    print("A: \n" + str(A))

    # STEP 1.2 - The diagonal degree Matrix:
    D = get_diagonal_matrix(A)

    print("D: " + str(D))

    # STEP 1.3 - The normalized similarity matrix:
    tmp = fractional_matrix_power(D, -0.5)

    print("tmp: " + str(tmp))

    W = np.matmul(np.matmul(tmp, A), tmp)

    print("W : " + str(W))

    # STEP 1.4 - Algorithm for optimizing H:

    # STEP 1.4.1 - Initialize H:
    m = np.mean(W)
    H = np.random.uniform(low=0, high=np.random.uniform(
        0, 2 * math.sqrt(m / k)), size=(N, k))
    # H = np.full_like(W, fill_value=np.random.uniform(0, 2 * math.sqrt(m / k)))

    print("randomized H-" + str(H))

    # STEP 1.4.2 - Update H:
    diff = 1
    iter = 1
    while diff >= EPSILON and iter <= MAX_ITER:
        next_H = H * (1 - BETA + BETA*(np.matmul(W, H) /
                                       np.matmul(np.matmul(H, np.transpose(H)), H)))

        diff = get_frobenius_norm(next_H - H) ** 2
        iter += 1
        H = next_H

    print("H -" + str(H))

    # STEP 1.5 - Deriving a clustering solution

    # clusters = np.empty((k, N))
    # print("X - " + str(X))

    # for i in range(N):
    #     max_index = np.argmax(H[i])
    #     print("max_index - " + str(max_index))

    #     np.append(clusters[max_index], X[i])
    #     # clusters[max_index].append(X[i])

    # print("clusters - " + str(clusters))


def get_diagonal_matrix(A):
    rows, columns = np.shape(A)
    D = np.empty_like(A)

    for i in range(rows):
        for j in range(columns):

            if i != j:
                D[i][j] = 0

            else:
                D[i][j] = np.sum(A, axis=1)[i]

    return D


def get_similarity_matrix(X):
    rows, _columns = np.shape(X)
    print("rows - " + str(rows))
    A = np.empty(shape=(rows, rows))

    for i in range(rows):
        for j in range(rows):

            if i == j:
                A[i][j] = 0

            else:
                A[i][j] = math.exp(-(euclidean_distance(X[i], X[j]) ** 2) / 2)

    return A


def euclidean_distance(vector_x, vector_y):
    sum = 0
    for i in range(len(vector_x)):
        sum += (vector_x[i] - vector_y[i]) ** 2
    return math.sqrt(sum)


def get_frobenius_norm(H):
    return math.sqrt(np.trace(H*H))

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
