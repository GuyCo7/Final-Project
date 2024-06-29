import sys
import math
import numpy as np
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
    text_file = open(file_name, 'r')
    raw_text = text_file.read()
    text_file.close()

    vectors = raw_text.splitlines()
    data = [vector.split(',') for vector in vectors]
    X = [row[:] for row in data]  # clone data to X
    for vector_index in range(len(data)):
        for token_index in range(len(data[vector_index])):
            X[vector_index][token_index] = float(
                data[vector_index][token_index])
    N = len(X)
    d = len(X[0])

    # STEP b:
    # STEP 1.1 - The Similarity Matrix:
    A = symnmf_capi.sym(X, N, d)
    if goal == 'sym':
        print_matrix(A)

    # STEP 1.2 - The diagonal degree Matrix:
    D = symnmf_capi.ddg(A, N)
    if goal == 'ddg':
        print_matrix(D)

    # STEP 1.3 - The normalized similarity matrix:
    W = symnmf_capi.norm(A, D, N)
    if goal == 'norm':
        print_matrix(W)

    # STEP 1.4 - Algorithm for optimizing H:

    # STEP 1.4.1 - Initialize H:
    m = np.mean(W)
    H = np.random.uniform(low=0, high=2 * math.sqrt(m / k), size=(N, k))

    initial_H = H.tolist()

    final_H = symnmf_capi.symnmf(X, initial_H, N, d, k)
    if goal == 'symnmf':
        print_matrix(final_H)

    return


# Function to convert a string to int
def convert_to_number(str):
    try:
        return int(str)
    except ValueError:
        print(str + " is not a whole number!")
        print("Check your arguments again")
        exit()


def print_matrix(mat):
    rows = len(mat)
    cols = len(mat[0])
    for i in range(rows):
        for j in range(cols):
            if (j < cols - 1):
                print("{:.4f}".format(mat[i][j]), end=",")
            else:
                print("{:.4f}".format(mat[i][j]), end="")

        print()


if __name__ == "__main__":
    main()
