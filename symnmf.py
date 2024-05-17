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

    print("X:")
    print_matrix(X)

    # STEP b:
    # STEP 1.1 - The Similarity Matrix:
    A = symnmf_capi.sym(N, d, X)
    print("A:")
    print_matrix(A)

    # STEP 1.2 - The diagonal degree Matrix:
    D = symnmf_capi.ddg(N, A)
    print("D:")
    print_matrix(D)

    # STEP 1.3 - The normalized similarity matrix:
    W = symnmf_capi.norm(N, A, D)
    print("W:")
    print_matrix(W)

    # STEP 1.4 - Algorithm for optimizing H:

    # STEP 1.4.1 - Initialize H:
    m = np.mean(W)
    H = np.random.uniform(low=0, high=np.random.uniform(
        0, 2 * math.sqrt(m / k)), size=(N, k))

    # STEP 1.4.2 - Update H:
    diff = 1
    iter = 1
    while diff >= EPSILON and iter <= MAX_ITER:
        next_H = H * (1 - BETA + BETA*(np.matmul(W, H) /
                                       np.matmul(np.matmul(H, np.transpose(H)), H)))

        diff = get_frobenius_norm(next_H - H) ** 2
        iter += 1
        H = next_H

    print("H: \n" + str(H))


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
