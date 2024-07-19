import sys
import math
import numpy as np
import symnmf_capi

EPSILON = 1e-4  # 0.0001
BETA = 0.5
GOAL_PARAMS = ['symnmf', 'sym', 'ddg', 'norm']


def main():

    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        exit()

    np.random.seed(0)

    k = convert_to_number(sys.argv[1])

    goal = sys.argv[2]
    assert goal in GOAL_PARAMS, "An Error Has Occurred"

    file_name = sys.argv[3]
    text_file = open(file_name, 'r')
    raw_text = text_file.read()
    text_file.close()

    vectors = raw_text.splitlines()
    data = [vector.split(',') for vector in vectors]
    X = [row[:] for row in data]
    for i in range(len(data)):
        for j in range(len(data[i])):
            X[i][j] = float(data[i][j])
    N = len(X)
    d = len(X[0])

    assert k < N, "An Error Has Occurred"

    A = symnmf_capi.sym(X, N, d)
    if goal == 'sym':
        print_matrix(A)
        return

    D = symnmf_capi.ddg(A, N)
    if goal == 'ddg':
        print_matrix(D)
        return

    W = symnmf_capi.norm(A, D, N)
    if goal == 'norm':
        print_matrix(W)
        return

    m = np.mean(W)
    H = np.random.uniform(low=0, high=2 * math.sqrt(m / k), size=(N, k))

    initial_H = H.tolist()

    final_H = symnmf_capi.symnmf(X, initial_H, N, d, k)
    if goal == 'symnmf':
        print_matrix(final_H)

    return


# Helper Functions:

def convert_to_number(str):
    try:
        return int(str)
    except ValueError:
        print("An Error Has Occurred")
        exit()


def print_matrix(mat):
    for row in mat:
        formatted_row = ",".join("{:.4f}".format(val) for val in row)
        print(formatted_row)


if __name__ == "__main__":
    main()
