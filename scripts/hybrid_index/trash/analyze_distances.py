#!/usr/bin/env python3
"""
Read HDF5 hybrid dataset and compute average dense and sparse distances.
"""

import h5py
import numpy as np
import argparse
import struct


def parse_sparse_vectors(blob):
    """
    Parse binary blob into list of sparse vectors.
    Format: [len(uint32), ids(uint32[len]), vals(float32[len])]*N
    """
    vectors = []
    offset = 0
    blob_bytes = bytes(blob)

    while offset < len(blob_bytes):
        # Read length (uint32)
        if offset + 4 > len(blob_bytes):
            break

        length = struct.unpack_from('I', blob_bytes, offset)[0]
        offset += 4

        if length == 0:
            vectors.append({'ids': np.array([], dtype=np.uint32),
                            'vals': np.array([], dtype=np.float32)})
            continue

        # Read IDs (uint32 array)
        ids_size = length * 4
        if offset + ids_size > len(blob_bytes):
            break
        ids = np.frombuffer(blob_bytes, dtype=np.uint32, count=length, offset=offset)
        offset += ids_size

        # Read values (float32 array)
        vals_size = length * 4
        if offset + vals_size > len(blob_bytes):
            break
        vals = np.frombuffer(blob_bytes, dtype=np.float32, count=length, offset=offset)
        offset += vals_size

        vectors.append({'ids': ids.copy(), 'vals': vals.copy()})

    return vectors


def compute_dense_distance(vec1, vec2):
    """
    Compute dense inner product distance.
    For inner product, higher value = more similar, so distance = -inner_product
    Or we can use: distance = 1 - inner_product for normalized vectors
    Here we return the inner product itself (similarity score).
    """
    return np.dot(vec1, vec2)


def compute_sparse_distance(sparse1, sparse2):
    """
    Compute sparse inner product distance.
    Returns the inner product (similarity score).
    """
    dict1 = dict(zip(sparse1['ids'], sparse1['vals']))
    dict2 = dict(zip(sparse2['ids'], sparse2['vals']))

    sparse_ip = 0.0
    for idx in dict1:
        if idx in dict2:
            sparse_ip += dict1[idx] * dict2[idx]

    return sparse_ip


def compute_all_distances(dense_vectors, sparse_vectors):
    """
    Compute all pairwise dense and sparse distances.
    Returns average dense distance and average sparse distance.
    """
    n = len(dense_vectors)

    if n != len(sparse_vectors):
        raise ValueError(f"Mismatch: {n} dense vectors but {len(sparse_vectors)} sparse vectors")

    dense_distances = []
    sparse_distances = []

    print(f"Computing pairwise distances for {n} vectors...")

    # Compute all pairwise distances
    for i in range(n):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n}")

        for j in range(i + 1, n):  # Only compute upper triangle (symmetric)
            # Dense distance
            dense_dist = compute_dense_distance(dense_vectors[i], dense_vectors[j])
            dense_distances.append(dense_dist)

            # Sparse distance
            sparse_dist = compute_sparse_distance(sparse_vectors[i], sparse_vectors[j])
            sparse_distances.append(sparse_dist)

    return dense_distances, sparse_distances


def compute_query_distances(train_dense, train_sparse, test_dense, test_sparse, topk=10):
    """
    Compute topk distances between test queries and train vectors.
    For each query, find topk by dense distance and topk by sparse distance separately.
    Returns all topk distances combined into two lists.
    """
    n_test = len(test_dense)
    n_test = 1
    n_train = len(train_dense)

    dense_distances = []
    sparse_distances = []

    print(f"Computing query distances: {n_test} queries × {n_train} train vectors...")
    print(f"Finding top-{topk} for each query by dense and sparse distance separately")

    for i in range(n_test):
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_test}")

        # Compute all distances for this query
        query_dense = []
        query_sparse = []

        for j in range(n_train):
            # Dense distance
            dense_dist = compute_dense_distance(test_dense[i], train_dense[j])
            query_dense.append(dense_dist)

            # Sparse distance
            sparse_dist = compute_sparse_distance(test_sparse[i], train_sparse[j])
            query_sparse.append(sparse_dist)

        query_dense = np.array(query_dense)
        query_sparse = np.array(query_sparse)

        # Get topk by dense distance (highest inner product = most similar)
        d_topk_idx = np.argsort(query_dense)[-topk:][::-1]
        dense_distances.extend(query_dense[d_topk_idx].tolist())

        # Get topk by sparse distance (highest inner product = most similar)
        s_topk_idx = np.argsort(query_sparse)[-topk:][::-1]
        sparse_distances.extend(query_sparse[s_topk_idx].tolist())

    return dense_distances, sparse_distances


def main():
    parser = argparse.ArgumentParser(description="Compute average dense and sparse distances from HDF5")
    parser.add_argument("--input", required=True, help="Input HDF5 file path")
    parser.add_argument("--mode", choices=["train", "test", "query"], default="query",
                        help="Mode: 'train' (train-train), 'test' (test-test), 'query' (test-train)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample size (use subset of data for faster computation)")

    args = parser.parse_args()

    print(f"Reading HDF5 file: {args.input}")

    with h5py.File(args.input, "r") as f:
        # Read dense vectors
        train_dense = np.array(f["train"])
        test_dense = np.array(f["test"])

        # Read sparse vectors
        train_sparse_blob = np.array(f["train_sparse"])
        test_sparse_blob = np.array(f["test_sparse"])

        print(f"Loaded: {len(train_dense)} train vectors, {len(test_dense)} test vectors")
        print(f"Dense dimension: {train_dense.shape[1]}")

    # Parse sparse vectors
    print("Parsing sparse vectors...")
    train_sparse = parse_sparse_vectors(train_sparse_blob)
    test_sparse = parse_sparse_vectors(test_sparse_blob)
    print(f"Parsed: {len(train_sparse)} train sparse, {len(test_sparse)} test sparse")

    # Sample if requested
    if args.sample:
        print(f"Sampling {args.sample} vectors...")
        if args.mode == "train":
            indices = np.random.choice(len(train_dense), min(args.sample, len(train_dense)), replace=False)
            train_dense = train_dense[indices]
            train_sparse = [train_sparse[i] for i in indices]
        elif args.mode == "test":
            indices = np.random.choice(len(test_dense), min(args.sample, len(test_dense)), replace=False)
            test_dense = test_dense[indices]
            test_sparse = [test_sparse[i] for i in indices]
        else:  # query
            test_indices = np.random.choice(len(test_dense), min(args.sample, len(test_dense)), replace=False)
            test_dense = test_dense[test_indices]
            test_sparse = [test_sparse[i] for i in test_indices]

    # Compute distances based on mode
    if args.mode == "train":
        dense_dists, sparse_dists = compute_all_distances(train_dense, train_sparse)
    elif args.mode == "test":
        dense_dists, sparse_dists = compute_all_distances(test_dense, test_sparse)
    else:  # query
        dense_dists, sparse_dists = compute_query_distances(
            train_dense, train_sparse, test_dense, test_sparse
        )

    # Compute statistics
    dense_dists = np.array(dense_dists)
    sparse_dists = np.array(sparse_dists)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Total distances computed: {len(dense_dists)}")
    print()
    print("Dense Distance (Inner Product) Statistics:")
    print(f"  Mean:   {np.mean(dense_dists):.6f}")
    print(f"  Median: {np.median(dense_dists):.6f}")
    print(f"  Std:    {np.std(dense_dists):.6f}")
    print(f"  Min:    {np.min(dense_dists):.6f}")
    print(f"  Max:    {np.max(dense_dists):.6f}")
    print()
    print("Sparse Distance (Inner Product) Statistics:")
    print(f"  Mean:   {np.mean(sparse_dists):.6f}")
    print(f"  Median: {np.median(sparse_dists):.6f}")
    print(f"  Std:    {np.std(sparse_dists):.6f}")
    print(f"  Min:    {np.min(sparse_dists):.6f}")
    print(f"  Max:    {np.max(sparse_dists):.6f}")
    print()
    print(f"Non-zero sparse distances: {np.count_nonzero(sparse_dists)} ({100*np.count_nonzero(sparse_dists)/len(sparse_dists):.2f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
