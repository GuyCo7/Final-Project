import sys
import math
import numpy as np
import sklearn.metrics
import symnmf_capi
# import kmeans_capi

MAX_ITER = 300
EPSILON = 1e-4  # 0.0001
BETA = 0.5


def main():

    k = convert_to_number(sys.argv[1])

    file_name = sys.argv[2]
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

    A = symnmf_capi.sym(X, N, d)
    D = symnmf_capi.ddg(A, N)
    W = symnmf_capi.norm(A, D, N)

    m = np.mean(W)
    print("m --- ", m)
    print("k --- ", k)
    H = np.random.uniform(low=0, high=np.random.uniform(
        0, 2 * math.sqrt(m / k)), size=(N, k))

    initial_H = H.tolist()

    H = symnmf_capi.symnmf(X, initial_H, N, d, k)

    print_matrix(H)
    
    clusters = [[] for _ in range(k)]
    
    for row_index in range(len(H)):
        cluster_index = np.argmax(H[row_index])
        clusters[cluster_index].append(X[row_index])
    
    
    print("clusters:")
    print(str(clusters))

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
