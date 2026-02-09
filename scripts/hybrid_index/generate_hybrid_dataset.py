#!/usr/bin/env python3
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate HDF5 dataset compatible with VSAG's EvalDataset::Load() for HYBRID vectors.

This script exactly mimics the behavior of the C++ GenerateSparseVectors function:
- Generates random unique IDs in range [0, max_id]
- Ensures max_dim <= max_id
- Sorts IDs in ascending order
- Uses the same distribution ranges
"""

import h5py
import numpy as np
import struct
import argparse
import os


def generate_sparse_vector(max_dim, max_id, min_val, max_val, rng):
    """
    Generate one sparse vector exactly like the C++ version:
    - len ~ Uniform(max_dim/2, max_dim)
    - IDs ~ Uniform(0, max_id) with uniqueness
    - Values ~ Uniform(min_val, max_val)
    - IDs are sorted in ascending order
    """
    if max_dim > max_id:
        raise ValueError(f"generate sparse vectors failed, max_dim ({max_dim}) > max_id ({max_id})")

    # Generate length: Uniform(max_dim/2, max_dim)
    min_len = max_dim // 2
    actual_dim = rng.integers(min_len, max_dim + 1)

    # Generate unique random IDs in range [0, max_id]
    if actual_dim > max_id + 1:
        raise ValueError(f"Cannot generate {actual_dim} unique IDs from range [0, {max_id}]")

    # Use choice without replacement to get unique IDs
    ids = rng.choice(np.arange(0, max_id + 1), size=actual_dim, replace=False)
    ids.sort()  # Sort in ascending order (like C++ std::sort)

    # Generate random values
    vals = rng.uniform(min_val, max_val, size=actual_dim).astype(np.float32)

    return {'ids': ids.astype(np.uint32), 'vals': vals}


def serialize_sparse_vectors(sparse_vectors):
    """
    Serialize sparse vectors in exact format expected by C++ parse_sparse_vectors.
    Uses system native byte order (matches C++ behavior).
    """
    blob = bytearray()

    for vec in sparse_vectors:
        len_val = len(vec['ids'])

        # Write length as uint32 (system native byte order)
        blob.extend(np.array([len_val], dtype=np.uint32).tobytes())

        if len_val > 0:
            # Write IDs as uint32 array (system native)
            ids_array = np.array(vec['ids'], dtype=np.uint32)
            blob.extend(ids_array.tobytes())

            # Write values as float32 array (system native)
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
    parser = argparse.ArgumentParser(description="Generate VSAG hybrid HDF5 dataset")
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument("--num-train", type=int, default=1000, help="Number of training vectors")
    parser.add_argument("--num-test", type=int, default=100, help="Number of test vectors")
    parser.add_argument("--dense-dim", type=int, default=128, help="Dense vector dimension")
    parser.add_argument("--sparse-max-dim", type=int, default=100, help="Max sparse dimension per vector")
    parser.add_argument("--sparse-max-id", type=int, default=10000, help="Max ID value for sparse vectors")
    parser.add_argument("--min-val", type=float, default=0.0, help="Min value for sparse vector elements")
    parser.add_argument("--max-val", type=float, default=10.0, help="Max value for sparse vector elements")
    parser.add_argument("--gt-k", type=int, default=10, help="Ground truth K (neighbors per query)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight: alpha*dense + (1-alpha)*sparse")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Validate parameters (like C++ version)
    if args.sparse_max_dim > args.sparse_max_id:
        raise ValueError(f"sparse_max_dim ({args.sparse_max_dim}) > sparse_max_id ({args.sparse_max_id})")

    # Set random seed
    rng = np.random.default_rng(args.seed)
    print(f"Generating hybrid dataset (IP distance, alpha={args.alpha})...")
    print(f"  Train vectors: {args.num_train}")
    print(f"  Test vectors: {args.num_test}")
    print(f"  Dense dim: {args.dense_dim}")
    print(f"  Sparse: max_dim={args.sparse_max_dim}, max_id={args.sparse_max_id}")
    print(f"  Sparse values: [{args.min_val}, {args.max_val}]")
    print(f"  Output: {args.output}")

    # Generate dense vectors (L2 normalized for IP distance)
    train_dense = rng.random((args.num_train, args.dense_dim)).astype(np.float32)
    test_dense = rng.random((args.num_test, args.dense_dim)).astype(np.float32)

    # Normalize for inner product (cosine similarity)
    train_dense /= np.linalg.norm(train_dense, axis=1, keepdims=True)
    test_dense /= np.linalg.norm(test_dense, axis=1, keepdims=True)

    # Generate sparse vectors (exactly like C++ version)
    print("Generating sparse vectors...")
    train_sparse_list = []
    for i in range(args.num_train):
        vec = generate_sparse_vector(
            args.sparse_max_dim,
            args.sparse_max_id,
            args.min_val,
            args.max_val,
            rng
        )
        train_sparse_list.append(vec)

    test_sparse_list = []
    for i in range(args.num_test):
        vec = generate_sparse_vector(
            args.sparse_max_dim,
            args.sparse_max_id,
            args.min_val,
            args.max_val,
            rng
        )
        test_sparse_list.append(vec)

    # Serialize sparse vectors to binary blobs
    print("Serializing sparse vectors...")
    train_sparse_blob = serialize_sparse_vectors(train_sparse_list)
    test_sparse_blob = serialize_sparse_vectors(test_sparse_list)

    # Compute ground truth
    neighbors, distances = compute_ground_truth(
        train_dense, train_sparse_list,
        test_dense, test_sparse_list,
        args.gt_k, args.alpha
    )

    # Generate labels (0, 1, 2, ...)
    train_labels = np.arange(args.num_train, dtype=np.int64)
    test_labels = np.arange(args.num_test, dtype=np.int64)

    # Write to HDF5 file
    print("Writing HDF5 file...")
    with h5py.File(args.output, "w") as f:
        # Dense vectors (float32)
        f.create_dataset("train", data=train_dense, dtype=np.float32)
        f.create_dataset("test", data=test_dense, dtype=np.float32)

        # Sparse vectors as uint8 binary blobs
        f.create_dataset("train_sparse", data=np.frombuffer(train_sparse_blob, dtype=np.uint8))
        f.create_dataset("test_sparse", data=np.frombuffer(test_sparse_blob, dtype=np.uint8))

        # Ground truth
        f.create_dataset("neighbors", data=neighbors, dtype=np.int64)
        f.create_dataset("distances", data=distances, dtype=np.float32)

        # Labels
        f.create_dataset("train_labels", data=train_labels, dtype=np.int64)
        f.create_dataset("test_labels", data=test_labels, dtype=np.int64)

        # Attributes
        f.attrs["type"] = "hybrid"
        f.attrs["distance"] = "ip"

    with open('/tbase-project/vsag/scripts/debug_python_train.bin', 'wb') as f:
        f.write(train_sparse_blob)
    print(f"Python debug file: debug_python_train.bin ({len(train_sparse_blob)} bytes)")

    print(f" Hybrid dataset saved to {args.output}")
    print(f"   File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")


    # print("train: ")
    # for i in range(args.num_train):
    #     print(f"{i}: ")
    #     print(f"dense: {train_dense[i]}")
    #     print(f"sparse: {train_sparse_list[i]}")
    #
    # print("test: ")
    # for i in range(args.num_test):
    #     print(f"{i}: ")
    #     print(f"dense: {test_dense[i]}")
    #     print(f"sparse: {test_sparse_list[i]}")
    #     print("ground_truth: ")
    #     for j in range(args.gt_k):
    #         print(f"num{j}: ")
    #         print(f"label: {neighbors[i][j]}")
    #         print(f"dis: {distances[i][j]}")
    #     print()


if __name__ == "__main__":
    main()
