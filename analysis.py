import sys
import math
import numpy as np
import sklearn.metrics
import symnmf_capi

MAX_ITER = 300
EPSILON = 1e-4  # 0.0001
BETA = 0.5


def main():

    np.random.seed(0)

    k = convert_to_number(sys.argv[1])

    file_name = sys.argv[2]
    text_file = open(file_name, 'r')
    raw_text = text_file.read()
    text_file.close()

    vectors = raw_text.splitlines()
    data = [vector.split(',') for vector in vectors]
    X = [row[:] for row in data]  # clone data to X
    for i in range(len(data)):
        for j in range(len(data[i])):
            X[i][j] = float(data[i][j])
    N = len(X)
    d = len(X[0])

    A = symnmf_capi.sym(X, N, d)
    D = symnmf_capi.ddg(A, N)
    W = symnmf_capi.norm(A, D, N)

    m = np.mean(W)
    H = np.random.uniform(low=0, high=2 * math.sqrt(m / k), size=(N, k))

    initial_H = H.tolist()

    H = symnmf_capi.symnmf(X, initial_H, N, d, k)

    labels = np.argmax(H, axis=1)

    score = sklearn.metrics.silhouette_score(X, labels)
    print("nmf:", "{:.4f}".format(score))

    kmeans_labels = kmeans_fit(X, N, d, k)
    np_kmeans_labels = np.array(kmeans_labels)

    kmeans_score = sklearn.metrics.silhouette_score(X, np_kmeans_labels)
    print("kmeans:", "{:.4f}".format(kmeans_score))

    return


def convert_to_number(str):
    try:
        return int(str)
    except ValueError:
        print("An Error Has Occurred")
        exit()


def kmeans_fit(vectors_arr, n, d, k):
    # Step 1: Initialize first k vectors as centroids
    centroids = [vectors_arr[i] for i in range(k)]

    # Steps 2&5: Iterate until all delta centroids are smaller than EPSILON or max iteration is reached
    iteration = 0
    delta_centroids = [1 for _ in range(k)]
    while are_bigger_than_epsilon(delta_centroids) and iteration < MAX_ITER:

        # Step 3: Assign every vector to the closest cluster
        closest_centroid_for_vector = find_closest_centroids(
            vectors_arr, centroids)

        # Step 4: Update the centroids
        for centroid_index in range(k):
            sum = [0 for _ in range(d)]
            count = 0

            for i in range(n):
                if closest_centroid_for_vector[i] == centroid_index:
                    count += 1
                    for j in range(d):
                        sum[j] += float(vectors_arr[i][j])

            new_centroid = [(sum[i] / count) for i in range(d)]

            delta_centroids[centroid_index] = euclidean_distance(
                centroids[centroid_index], new_centroid)
            centroids[centroid_index] = new_centroid

        iteration += 1

    return closest_centroid_for_vector


def find_closest_centroids(vectors_arr, centroids):
    closest_centroid_for_vector = []
    for vector in vectors_arr:
        distances = [euclidean_distance(vector, centroid)
                     for centroid in centroids]
        closest_centroid_for_vector.append(distances.index(min(distances)))
    return closest_centroid_for_vector


def euclidean_distance(vector_x, vector_y):
    sum = 0
    for i in range(len(vector_x)):
        sum += (float(vector_x[i]) - float(vector_y[i])) ** 2

    return math.sqrt(sum)


def are_bigger_than_epsilon(arr):
    return any(value > EPSILON for value in arr)


if __name__ == "__main__":
    main()
