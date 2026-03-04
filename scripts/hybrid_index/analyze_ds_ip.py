#!/usr/bin/env python3
import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm


def deserialize_sparse_vectors(sparse_blob, num_vectors):
    """反序列化稀疏向量"""
    sparse_vectors = []
    offset = 0
    blob_bytes = bytes(sparse_blob)

    for _ in range(num_vectors):
        len_val = np.frombuffer(blob_bytes[offset:offset+4], dtype=np.uint32)[0]
        offset += 4

        if len_val > 0:
            ids = np.frombuffer(blob_bytes[offset:offset+len_val*4], dtype=np.uint32)
            offset += len_val * 4
            vals = np.frombuffer(blob_bytes[offset:offset+len_val*4], dtype=np.float32)
            offset += len_val * 4
            sparse_vectors.append({'ids': ids, 'vals': vals})
        else:
            sparse_vectors.append({'ids': np.array([], dtype=np.uint32),
                                   'vals': np.array([], dtype=np.float32)})

    return sparse_vectors


def compute_dense_score(dense1, dense2):
    """计算稠密向量内积"""
    return np.dot(dense1, dense2)


def compute_sparse_score(sparse1, sparse2):
    """计算稀疏向量内积"""
    dict1 = dict(zip(sparse1['ids'], sparse1['vals']))
    dict2 = dict(zip(sparse2['ids'], sparse2['vals']))
    score = 0.0
    for idx in dict1:
        if idx in dict2:
            score += dict1[idx] * dict2[idx]
    return score


def get_topk_scores(query_dense, query_sparse, doc_dense_list, doc_sparse_list, k):
    """获取某个query的topk稠密和稀疏分数

    Returns:
        dense_topk_scores: topk个稠密分数
        sparse_topk_scores: topk个稀疏分数
        dense_topk_indices: 稠密topk的索引
        sparse_topk_indices: 稀疏topk的索引
    """
    num_docs = len(doc_dense_list)

    # 计算所有分数
    dense_scores = []
    sparse_scores = []

    for i in range(num_docs):
        dense_score = compute_dense_score(query_dense, doc_dense_list[i])
        sparse_score = compute_sparse_score(query_sparse, doc_sparse_list[i])
        dense_scores.append(dense_score)
        sparse_scores.append(sparse_score)

    dense_scores = np.array(dense_scores)
    sparse_scores = np.array(sparse_scores)

    # 获取topk
    dense_topk_indices = np.argsort(dense_scores)[-k:][::-1]
    sparse_topk_indices = np.argsort(sparse_scores)[-k:][::-1]

    dense_topk_scores = dense_scores[dense_topk_indices]
    sparse_topk_scores = sparse_scores[sparse_topk_indices]

    return dense_topk_scores, sparse_topk_scores, dense_topk_indices, sparse_topk_indices


def analyze_topk_overlap(dense_indices, sparse_indices):
    """分析topk的重叠情况"""
    dense_set = set(dense_indices)
    sparse_set = set(sparse_indices)
    overlap = len(dense_set & sparse_set)
    return overlap


def analyze_train_train_topk(train_dense, train_sparse, num_samples, k):
    """分析 train 内部topk相似度"""
    print(f"\n[1] Train内部topk={k}相似度分析 (采样{num_samples}条作为query)")

    # 随机采样作为query
    query_indices = np.random.choice(len(train_dense), min(num_samples, len(train_dense)), replace=False)

    dense_topk_all = []
    sparse_topk_all = []
    overlap_counts = []

    print(f"  对每个query在所有train中找topk...")
    for i in tqdm(range(len(query_indices)), desc="  Train-Train"):
        query_idx = query_indices[i]

        # 排除自己
        doc_indices = [j for j in range(len(train_dense)) if j != query_idx]
        doc_dense_list = [train_dense[j] for j in doc_indices]
        doc_sparse_list = [train_sparse[j] for j in doc_indices]

        dense_topk, sparse_topk, dense_idx, sparse_idx = get_topk_scores(
            train_dense[query_idx], train_sparse[query_idx],
            doc_dense_list, doc_sparse_list, k
        )

        dense_topk_all.extend(dense_topk)
        sparse_topk_all.extend(sparse_topk)
        overlap = analyze_topk_overlap(dense_idx, sparse_idx)
        overlap_counts.append(overlap)

    dense_topk_all = np.array(dense_topk_all)
    sparse_topk_all = np.array(sparse_topk_all)

    print(f"\n  【结果 - 仅统计TopK中的分数】")
    print(f"    稠密TopK均值:  {np.mean(dense_topk_all):.6f}")
    print(f"    稀疏TopK均值:  {np.mean(sparse_topk_all):.6f}")
    print(f"    稠密/稀疏:     {np.mean(dense_topk_all) / (np.mean(sparse_topk_all) + 1e-10):.4f}")
    print(f"    稠密标准差:    {np.std(dense_topk_all):.6f}")
    print(f"    稀疏标准差:    {np.std(sparse_topk_all):.6f}")
    print(f"    TopK重叠度:    {np.mean(overlap_counts):.2f}/{k} ({np.mean(overlap_counts)/k*100:.1f}%)")

    return dense_topk_all, sparse_topk_all


def analyze_query_train_topk(test_dense, test_sparse, train_dense, train_sparse,
                             num_queries, k):
    """分析 query 和 train 的topk相似度"""
    print(f"\n[2] Query-Train topk={k}相似度分析 (采样{num_queries}个query)")

    # 随机采样query
    query_indices = np.random.choice(len(test_dense), min(num_queries, len(test_dense)), replace=False)

    dense_topk_all = []
    sparse_topk_all = []
    overlap_counts = []

    print(f"  对每个query在所有train中找topk...")
    for i in tqdm(range(len(query_indices)), desc="  Query-Train"):
        query_idx = query_indices[i]

        dense_topk, sparse_topk, dense_idx, sparse_idx = get_topk_scores(
            test_dense[query_idx], test_sparse[query_idx],
            train_dense, train_sparse, k
        )

        dense_topk_all.extend(dense_topk)
        sparse_topk_all.extend(sparse_topk)
        overlap = analyze_topk_overlap(dense_idx, sparse_idx)
        overlap_counts.append(overlap)

    dense_topk_all = np.array(dense_topk_all)
    sparse_topk_all = np.array(sparse_topk_all)

    print(f"\n  【结果 - 仅统计TopK中的分数】")
    print(f"    稠密TopK均值:  {np.mean(dense_topk_all):.6f}")
    print(f"    稀疏TopK均值:  {np.mean(sparse_topk_all):.6f}")
    print(f"    稠密/稀疏:     {np.mean(dense_topk_all) / (np.mean(sparse_topk_all) + 1e-10):.4f}")
    print(f"    稠密标准差:    {np.std(dense_topk_all):.6f}")
    print(f"    稀疏标准差:    {np.std(sparse_topk_all):.6f}")
    print(f"    TopK重叠度:    {np.mean(overlap_counts):.2f}/{k} ({np.mean(overlap_counts)/k*100:.1f}%)")

    return dense_topk_all, sparse_topk_all


def main():
    parser = argparse.ArgumentParser(description="统计混合向量TopK中稠密和稀疏IP的比例")
    parser.add_argument("--input", required=True, help="输入HDF5文件路径")
    parser.add_argument("--num-train", type=int, default=50, help="train采样数量作为query")
    parser.add_argument("--num-query", type=int, default=50, help="query采样数量")
    parser.add_argument("--topk", type=int, default=10, help="TopK值")

    args = parser.parse_args()

    print("=" * 80)
    print("混合向量TopK稠密/稀疏IP比例分析")
    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"Train采样: {args.num_train}")
    print(f"Query采样: {args.num_query}")
    print(f"TopK: {args.topk}")
    print("=" * 80)

    # 读取HDF5文件
    print("\n加载数据...")
    with h5py.File(args.input, "r") as f:
        train_dense = f["train"][:]
        test_dense = f["test"][:]
        train_sparse_blob = f["train_sparse"][:]
        test_sparse_blob = f["test_sparse"][:]

        print(f"  ✓ Train: {len(train_dense)} 条")
        print(f"  ✓ Test:  {len(test_dense)} 条")

    # 反序列化稀疏向量
    print("\n反序列化稀疏向量...")
    train_sparse = deserialize_sparse_vectors(train_sparse_blob, len(train_dense))
    test_sparse = deserialize_sparse_vectors(test_sparse_blob, len(test_dense))
    print(f"  ✓ 完成")

    # 分析1: Train内部topk
    train_dense_topk, train_sparse_topk = analyze_train_train_topk(
        train_dense, train_sparse, args.num_train, args.topk
    )

    # 分析2: Query-Train topk
    query_dense_topk, query_sparse_topk = analyze_query_train_topk(
        test_dense, test_sparse, train_dense, train_sparse,
        args.num_query, args.topk
    )

    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"\n【Train内部 (doc-doc) - TopK={args.topk}】")
    print(f"  稠密TopK均值: {np.mean(train_dense_topk):.6f}")
    print(f"  稀疏TopK均值: {np.mean(train_sparse_topk):.6f}")
    print(f"  比例:         {np.mean(train_dense_topk) / (np.mean(train_sparse_topk) + 1e-10):.4f} : 1")

    print(f"\n【Query-Train (query-doc) - TopK={args.topk}】")
    print(f"  稠密TopK均值: {np.mean(query_dense_topk):.6f}")
    print(f"  稀疏TopK均值: {np.mean(query_sparse_topk):.6f}")
    print(f"  比例:         {np.mean(query_dense_topk) / (np.mean(query_sparse_topk) + 1e-10):.4f} : 1")

    print(f"\n【建议的Alpha权重参考 (基于TopK分数)】")
    # 基于topk均值比例给出建议
    train_ratio = np.mean(train_dense_topk) / (np.mean(train_sparse_topk) + 1e-10)
    query_ratio = np.mean(query_dense_topk) / (np.mean(query_sparse_topk) + 1e-10)

    suggested_alpha_train = 1.0 / (1.0 + train_ratio)
    suggested_alpha_query = 1.0 / (1.0 + query_ratio)

    print(f"  基于Train TopK，建议alpha ≈ {suggested_alpha_train:.2f} (使两项贡献相当)")
    print(f"  基于Query TopK，建议alpha ≈ {suggested_alpha_query:.2f} (使两项贡献相当)")
    print(f"  综合建议: alpha ≈ {(suggested_alpha_train + suggested_alpha_query) / 2:.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
