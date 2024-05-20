import sys
import symnmf_capi
import kmeans_capi
import numpy as np
import math

MAX_ITER = 300
EPSILON = 1e-4  # 0.0001
BETA = 0.5

def main():
    
    k = sys.argv[1]
    
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
    
    H = symnmf_capi.symnmf(X, N, d)
    

if __name__ == "__main__":
    main()