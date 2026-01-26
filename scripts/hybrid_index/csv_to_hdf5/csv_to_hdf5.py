#!/usr/bin/env python3
"""
Read dense and sparse vectors from CSV and convert to HDF5 format.
"""

import h5py
import numpy as np
import pandas as pd
import argparse
import os


def parse_sparse_vector(sparse_str):
    """
    Parse sparse vector string like "335:0.11113053,465:0.09866308,..."
    Returns dict with 'ids' and 'vals' arrays
    """
    if pd.isna(sparse_str) or sparse_str == '':
        return {'ids': np.array([], dtype=np.uint32), 'vals': np.array([], dtype=np.float32)}

    pairs = sparse_str.split(',')
    ids = []
    vals = []

    for pair in pairs:
        idx, val = pair.split(':')
        ids.append(int(idx))
        vals.append(float(val))

    # Sort by IDs (ascending order, like the reference code)
    sorted_pairs = sorted(zip(ids, vals), key=lambda x: x[0])
    ids, vals = zip(*sorted_pairs) if sorted_pairs else ([], [])

    return {
        'ids': np.array(ids, dtype=np.uint32),
        'vals': np.array(vals, dtype=np.float32)
    }


def parse_dense_vector(dense_str):
    """
    Parse dense vector string (comma-separated or space-separated floats)
    Returns numpy array
    """
    if pd.isna(dense_str) or dense_str == '':
        return np.array([], dtype=np.float32)

    # Handle both comma and space separated values
    dense_str = dense_str.strip()
    if ',' in dense_str:
        vals = [float(x.strip()) for x in dense_str.split(',') if x.strip()]
    else:
        vals = [float(x.strip()) for x in dense_str.split() if x.strip()]

    return np.array(vals, dtype=np.float32)


def serialize_sparse_vectors(sparse_vectors):
    """
    Serialize sparse vectors in exact format expected by C++ parse_sparse_vectors.
    """
    blob = bytearray()

    for vec in sparse_vectors:
        len_val = len(vec['ids'])

        # Write length as uint32
        blob.extend(np.array([len_val], dtype=np.uint32).tobytes())

        if len_val > 0:
            # Write IDs as uint32 array
            ids_array = np.array(vec['ids'], dtype=np.uint32)
            blob.extend(ids_array.tobytes())

            # Write values as float32 array
            vals_array = np.array(vec['vals'], dtype=np.float32)
            blob.extend(vals_array.tobytes())

    return bytes(blob)


def compute_hybrid_score(dense1, sparse1, dense2, sparse2, alpha):
    """
    Compute hybrid similarity score (higher = more similar)
    score = alpha * dense_ip + (1 - alpha) * sparse_ip
    """
    # Dense inner product
    dense_ip = np.dot(dense1, dense2)

    # Sparse inner product
    dict1 = dict(zip(sparse1['ids'], sparse1['vals']))
    dict2 = dict(zip(sparse2['ids'], sparse2['vals']))

    sparse_ip = 0.0
    for idx in dict1:
        if idx in dict2:
            sparse_ip += dict1[idx] * dict2[idx]

    return alpha * dense_ip + (1 - alpha) * sparse_ip


def compute_ground_truth(train_dense, train_sparse, test_dense, test_sparse, k, alpha):
    """Compute ground truth neighbors and distances"""
    num_test = len(test_dense)
    num_train = len(train_dense)

    neighbors = np.zeros((num_test, k), dtype=np.int64)
    distances = np.zeros((num_test, k), dtype=np.float32)

    print(f"Computing ground truth for {num_test} queries...")
    for i in range(num_test):
        if i % 50 == 0:
            print(f"  Progress: {i}/{num_test}")

        dists = []
        for j in range(num_train):
            dist = compute_hybrid_score(
                test_dense[i], test_sparse[i],
                train_dense[j], train_sparse[j],
                alpha
            )
            dists.append((dist, j))

        dists.sort(key=lambda x: x[0], reverse=True)
        neighbors[i] = [idx for _, idx in dists[:k]]
        distances[i] = [dist for dist, _ in dists[:k]]

    return neighbors, distances


def main():
    parser = argparse.ArgumentParser(description="Convert CSV vectors to VSAG hybrid HDF5 dataset")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument("--num-train", type=int, required=True, help="Number of training vectors to read")
    parser.add_argument("--num-test", type=int, default=0, help="Number of test vectors (0=auto from remaining)")
    parser.add_argument("--dense-col", default="dense_emb", help="Dense vector column name")
    parser.add_argument("--sparse-col", default="sparse_emb", help="Sparse vector column name")
    parser.add_argument("--gt-k", type=int, default=10, help="Ground truth K (neighbors per query)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight: alpha*dense + (1-alpha)*sparse")
    parser.add_argument("--normalize", action="store_true", help="Normalize dense vectors for IP distance")
    parser.add_argument("--skip-gt", action="store_true", help="Skip ground truth computation (faster)")

    args = parser.parse_args()

    # Read CSV file
    print(f"Reading CSV file: {args.input}")
    df = pd.read_csv(args.input)

    if args.dense_col not in df.columns:
        raise ValueError(f"Dense column '{args.dense_col}' not found in CSV")
    if args.sparse_col not in df.columns:
        raise ValueError(f"Sparse column '{args.sparse_col}' not found in CSV")

    total_rows = len(df)
    print(f"Total rows in CSV: {total_rows}")

    if args.num_train > total_rows:
        raise ValueError(f"num_train ({args.num_train}) > total rows ({total_rows})")

    # Determine test size
    if args.num_test == 0:
        args.num_test = total_rows - args.num_train
        print(f"Auto-setting num_test to {args.num_test}")

    if args.num_train + args.num_test > total_rows:
        raise ValueError(f"num_train + num_test ({args.num_train + args.num_test}) > total rows ({total_rows})")

    # Parse vectors
    print("Parsing dense vectors...")
    all_dense = []
    for i in range(args.num_train + args.num_test):
        dense_vec = parse_dense_vector(df[args.dense_col].iloc[i])
        all_dense.append(dense_vec)

    all_dense = np.array(all_dense, dtype=np.float32)
    dense_dim = all_dense.shape[1]
    print(f"  Dense dimension: {dense_dim}")

    # Normalize if requested
    if args.normalize:
        print("Normalizing dense vectors...")
        norms = np.linalg.norm(all_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        all_dense = all_dense / norms

    # Split train/test
    train_dense = all_dense[:args.num_train]
    test_dense = all_dense[args.num_train:args.num_train + args.num_test]

    print("Parsing sparse vectors...")
    train_sparse_list = []
    for i in range(args.num_train):
        sparse_vec = parse_sparse_vector(df[args.sparse_col].iloc[i])
        train_sparse_list.append(sparse_vec)

    test_sparse_list = []
    for i in range(args.num_train, args.num_train + args.num_test):
        sparse_vec = parse_sparse_vector(df[args.sparse_col].iloc[i])
        test_sparse_list.append(sparse_vec)

    # Serialize sparse vectors
    print("Serializing sparse vectors...")
    train_sparse_blob = serialize_sparse_vectors(train_sparse_list)
    test_sparse_blob = serialize_sparse_vectors(test_sparse_list)

    # Compute ground truth
    if not args.skip_gt and args.num_test > 0:
        neighbors, distances = compute_ground_truth(
            train_dense, train_sparse_list,
            test_dense, test_sparse_list,
            args.gt_k, args.alpha
        )
    else:
        print("Skipping ground truth computation...")
        neighbors = np.zeros((args.num_test, args.gt_k), dtype=np.int64)
        distances = np.zeros((args.num_test, args.gt_k), dtype=np.float32)

    # Generate labels
    train_labels = np.arange(args.num_train, dtype=np.int64)
    test_labels = np.arange(args.num_test, dtype=np.int64)

    # Write to HDF5 file
    print("Writing HDF5 file...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with h5py.File(args.output, "w") as f:
        # Dense vectors (float32)
        f.create_dataset("train", data=train_dense, dtype=np.float32)
        if args.num_test > 0:
            f.create_dataset("test", data=test_dense, dtype=np.float32)

        # Sparse vectors as uint8 binary blobs
        f.create_dataset("train_sparse", data=np.frombuffer(train_sparse_blob, dtype=np.uint8))
        if args.num_test > 0:
            f.create_dataset("test_sparse", data=np.frombuffer(test_sparse_blob, dtype=np.uint8))

        # Ground truth
        if args.num_test > 0:
            f.create_dataset("neighbors", data=neighbors, dtype=np.int64)
            f.create_dataset("distances", data=distances, dtype=np.float32)

        # Labels
        f.create_dataset("train_labels", data=train_labels, dtype=np.int64)
        if args.num_test > 0:
            f.create_dataset("test_labels", data=test_labels, dtype=np.int64)

        # Attributes
        f.attrs["type"] = "hybrid"
        f.attrs["distance"] = "ip"
        f.attrs["alpha"] = args.alpha

    print(f"✓ Hybrid dataset saved to {args.output}")
    print(f"  Train vectors: {args.num_train}")
    print(f"  Test vectors: {args.num_test}")
    print(f"  Dense dimension: {dense_dim}")
    print(f"  File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
