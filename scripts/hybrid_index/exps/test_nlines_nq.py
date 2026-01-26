#!/usr/bin/env python3
import h5py
import numpy as np
from tqdm import tqdm
import argparse
import os


def deserialize_sparse_vectors(blob, num_vectors, num_train):
    """反序列化稀疏向量"""
    sparse_vectors = []
    offset = 0
    blob_bytes = bytes(blob)

    for _ in range(num_vectors):
        # 读取长度
        len_val = np.frombuffer(blob_bytes[offset:offset+4], dtype=np.uint32)[0]
        offset += 4

        if len_val > 0:
            # 读取 ids
            ids = np.frombuffer(blob_bytes[offset:offset+len_val*4], dtype=np.uint32)
            offset += len_val * 4

            # 读取 vals
            vals = np.frombuffer(blob_bytes[offset:offset+len_val*4], dtype=np.float32)
            offset += len_val * 4

            sparse_vectors.append({'ids': ids, 'vals': vals})
        else:
            sparse_vectors.append({'ids': np.array([], dtype=np.uint32),
                                   'vals': np.array([], dtype=np.float32)})

    num = num_train if num_train else len(sparse_vectors)
    return sparse_vectors[0: num]


def compute_sparse_ip(sparse1, sparse2):
    """计算稀疏向量的内积"""
    dict1 = dict(zip(sparse1['ids'], sparse1['vals']))
    dict2 = dict(zip(sparse2['ids'], sparse2['vals']))

    sparse_ip = 0.0
    for idx in dict1:
        if idx in dict2:
            sparse_ip += dict1[idx] * dict2[idx]

    return sparse_ip


def compute_hybrid_distances_exclude_self(query_idx, query_dense, query_sparse,
                                          train_dense, train_sparse):
    """
    计算查询点到其它所有训练点（排除自身）的混合距离参数
    返回: (n-1) x 2 的数组，每行是 [a, b]，表示距离函数 d(alpha) = a * alpha + b
    其中 a = dense_ip - sparse_ip, b = sparse_ip
    """
    n = len(train_dense)
    lines = []

    for i in range(n):
        if i == query_idx:  # 跳过自身
            continue

        # 计算稠密IP
        dense_ip = np.dot(train_dense[i], query_dense)

        # 计算稀疏IP
        sparse_ip = compute_sparse_ip(query_sparse, train_sparse[i])

        # 混合距离 = alpha * dense_ip + (1-alpha) * sparse_ip
        #          = alpha * (dense_ip - sparse_ip) + sparse_ip
        a = dense_ip - sparse_ip  # a = dense_ip - sparse_ip
        b = sparse_ip             # b = sparse_ip

        lines.append([a, b])

    return np.array(lines, dtype=np.float64)


def get_decimal_places(x_step):
    """根据步长确定需要的小数位数"""
    step_str = f"{x_step:.10f}".rstrip('0')
    if '.' in step_str:
        return len(step_str.split('.')[1])
    return 0


def convert_to_intervals(x_list, x_step, decimal_places):
    """将x值列表转换为左闭右开的区间列表，并合并连续区间"""
    if not x_list:
        return []

    x_list = sorted(x_list)

    # 对于每个x值，生成对应的区间
    raw_intervals = []
    for x in x_list:
        if abs(x - 1.0) < x_step/10:  # x = 1.0 的特殊情况
            left = 1.0
            right = 1.0
        else:
            left = round(x, decimal_places)
            right = round(min(x + x_step, 1.0), decimal_places)
        raw_intervals.append([left, right])

    # 合并连续区间
    intervals = []
    if raw_intervals:
        current_left, current_right = raw_intervals[0]

        for i in range(1, len(raw_intervals)):
            left, right = raw_intervals[i]

            # 如果当前区间的左端点等于前一个区间的右端点，则合并
            if abs(left - current_right) < x_step/10:
                current_right = right
            else:
                # 不连续，保存当前区间，开始新区间
                intervals.append([round(current_left, decimal_places),
                                  round(current_right, decimal_places)])
                current_left, current_right = left, right

        # 添加最后一个区间
        intervals.append([round(current_left, decimal_places),
                          round(current_right, decimal_places)])

    return intervals


def track_topk_survival_intervals(lines, k, x_step=0.01):
    """
    跟踪每条边进入top-k的生存区间

    参数:
        lines: (n-1, 2) 数组，每行 [a, b] 表示直线 y = a*x + b
        k: top-k的k值
        x_step: alpha的步长

    返回:
        survival_intervals: 每条边的生存区间列表
        num_all_neighbors: 每个alpha对应的邻居集合的并集的size
    """
    n = len(lines)
    decimal_places = get_decimal_places(x_step)

    # 初始化每条边的alpha值列表
    topk_alpha_values = [[] for _ in range(n)]

    # alpha从0到1变化
    alpha = 0.0
    alpha_values = []
    while alpha <= 1.0 + x_step/2:
        alpha_values.append(round(alpha, decimal_places))
        alpha += x_step

    all_neighbors = []
    # 对每个alpha值
    for alpha in alpha_values:
        # 计算所有边在当前alpha处的距离值
        distances = []
        for i, (a, b) in enumerate(lines):
            dist = a * alpha + b
            distances.append((dist, i))

        # 按距离降序排序，取top-k
        distances.sort(reverse=True, key=lambda item: item[0])
        topk_indices = [idx for _, idx in distances[:k]]

        # 记录当前alpha
        for idx in topk_indices:
            topk_alpha_values[idx].append(alpha)
            all_neighbors.append(idx)

    all_neighbors_unique = list(set(all_neighbors))
    num_all_neighbors = len(all_neighbors_unique)

    # 将alpha值列表转换为区间列表
    survival_intervals = []
    for alpha_list in topk_alpha_values:
        intervals = convert_to_intervals(alpha_list, x_step, decimal_places)
        survival_intervals.append(intervals)

    return survival_intervals, num_all_neighbors


def analyze_single_point(query_idx, train_dense, train_sparse, k, x_step):
    """分析单个训练点的生存区间"""
    # 计算到其它所有训练点（排除自身）的距离参数
    lines = compute_hybrid_distances_exclude_self(
        query_idx,
        train_dense[query_idx],
        train_sparse[query_idx],
        train_dense,
        train_sparse
    )

    # 跟踪生存区间
    survival_intervals, num_all_neighbors = track_topk_survival_intervals(lines, k, x_step)

    # 统计区间个数
    interval_counts = [len(intervals) for intervals in survival_intervals]

    return survival_intervals, interval_counts, num_all_neighbors

def main():
    parser = argparse.ArgumentParser(description="分析混合向量的Top-K生存区间")
    parser.add_argument("--input", required=True, help="输入 HDF5 文件路径")
    parser.add_argument("--k", type=int, default=10, help="Top-K 的 K 值")
    parser.add_argument("--step", type=float, default=0.01, help="Alpha 步长")
    parser.add_argument("--num-train", type=int, default=None, help="读取的数据数量（默认全部）")
    parser.add_argument("--num-points", type=int, default=None,
                        help="分析的训练点数量（默认全部）")
    parser.add_argument("--output", type=str, default="survival_analysis.txt",
                        help="输出文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    parser.add_argument("--save-details", action="store_true",
                        help="保存每个点的详细区间信息")

    args = parser.parse_args()

    print("=" * 80)
    print("Top-K 生存区间分析（训练数据）")
    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"K 值: {args.k}")
    print(f"Alpha 步长: {args.step}")
    print("=" * 80)

    # 读取数据
    print("\n[1/3] 读取 HDF5 数据...")
    with h5py.File(args.input, 'r') as f:
        train_dense = f['train'][:args.num_train]
        train_sparse_blob = f['train_sparse'][:]

        print(f"  训练集: {train_dense.shape[0]} 个向量")
        print(f"  稠密维度: {train_dense.shape[1]}")

    # 反序列化稀疏向量
    print("\n[2/3] 反序列化稀疏向量...")
    train_sparse = deserialize_sparse_vectors(train_sparse_blob, len(train_dense), args.num_train)
    print(f"  ✓ 完成")

    # 确定分析的点数
    num_points = args.num_points if args.num_points else len(train_dense)
    num_points = min(num_points, len(train_dense))

    print(f"\n[3/3] 分析 {num_points} 个训练点...")
    print(f"  每个点需要计算到其它 {len(train_dense)-1} 个点的距离")

    all_statistics = []

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Top-{args.k} 生存区间分析（训练数据）\n")
        f.write(f"训练集大小: {len(train_dense)}\n")
        f.write(f"分析点数: {num_points}\n")
        f.write(f"Alpha 步长: {args.step}\n")
        f.write("=" * 80 + "\n\n")

        for point_idx in tqdm(range(num_points), desc="处理训练点"):
            survival_intervals, interval_counts, num_all_neighbors = analyze_single_point(
                point_idx,
                train_dense,
                train_sparse,
                args.k,
                args.step
            )

            # 统计信息
            total_intervals = sum(interval_counts)
            max_intervals = max(interval_counts) if interval_counts else 0
            avg_intervals = np.mean(interval_counts) if interval_counts else 0
            zero_count = interval_counts.count(0)

            # 计算总生存时长
            total_lengths = []
            for intervals in survival_intervals:
                length = sum(right - left for left, right in intervals)
                total_lengths.append(length)

            stats = {
                'point_idx': point_idx,
                'total_intervals': total_intervals,
                'max_intervals': max_intervals,
                'avg_intervals': avg_intervals,
                'zero_count': zero_count,
                'interval_counts': interval_counts,
                'max_survival_length': max(total_lengths) if total_lengths else 0,
                'avg_survival_length': np.mean(total_lengths) if total_lengths else 0,
                'num_all_neighbors': num_all_neighbors
            }
            all_statistics.append(stats)

            # 写入文件
            if args.save_details:
                f.write(f"训练点 {point_idx}:\n")
                f.write(f"  总区间数: {total_intervals}\n")
                f.write(f"  最大区间数（单条边）: {max_intervals}\n")
                f.write(f"  平均区间数: {avg_intervals:.2f}\n")
                f.write(f"  从未进入 top-{args.k} 的边数: {zero_count} / {len(train_dense)-1}\n")
                f.write(f"  最长生存时长: {stats['max_survival_length']:.4f}\n")
                f.write(f"  平均生存时长: {stats['avg_survival_length']:.4f}\n")
                f.write(f"  邻居总数: {stats['num_all_neighbors']}\n")

                if args.verbose:
                    # 输出前10条边的生存区间
                    f.write(f"\n  前10条边的生存区间:\n")
                    for i in range(min(10, len(survival_intervals))):
                        if survival_intervals[i]:
                            intervals_str = ', '.join([f"[{left:.2f}, {right:.2f})"
                                                       if left != right
                                                       else f"[{left:.2f}]"
                                                       for left, right in survival_intervals[i]])
                            f.write(f"    边 {i}: {intervals_str}\n")
                        else:
                            f.write(f"    边 {i}: []\n")

                f.write("\n" + "-" * 80 + "\n\n")

        # 总体统计
        f.write("\n" + "=" * 80 + "\n")
        f.write("总体统计:\n")
        f.write("=" * 80 + "\n")

        total_intervals_all = [s['total_intervals'] for s in all_statistics]
        max_intervals_all = [s['max_intervals'] for s in all_statistics]
        avg_intervals_all = [s['avg_intervals'] for s in all_statistics]
        zero_count_all = [s['zero_count'] for s in all_statistics]
        max_survival_all = [s['max_survival_length'] for s in all_statistics]
        avg_survival_all = [s['avg_survival_length'] for s in all_statistics]
        avg_num_neighbors = [s['num_all_neighbors'] for s in all_statistics]

        f.write(f"分析的训练点数量: {num_points}\n")
        f.write(f"每个点的邻居候选数: {len(train_dense)-1}\n\n")

        f.write(f"总区间数统计:\n")
        f.write(f"  最小值: {min(total_intervals_all)}\n")
        f.write(f"  最大值: {max(total_intervals_all)}\n")
        f.write(f"  平均值: {np.mean(total_intervals_all):.2f}\n")
        f.write(f"  中位数: {np.median(total_intervals_all):.2f}\n\n")

        f.write(f"  最多邻居数: {max(avg_num_neighbors)}\n")
        f.write(f"  最少邻居数: {min(avg_num_neighbors)}\n")
        f.write(f"  中位数: {np.median(avg_num_neighbors)}\n")
        f.write(f"  平均每个点的邻居的去重并集的大小: {np.mean(avg_num_neighbors):.2f}\n")

if __name__ == "__main__":
    main()