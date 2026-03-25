#!/usr/bin/env python3
import h5py
import sys

def read_query0_neighbors(hdf5_path: str, output_path: str = "hdf5_topk.txt"):
    with h5py.File(hdf5_path, "r") as f:
        if "neighbors" not in f:
            print("❌ 不存在 'neighbors' 数据集")
            sys.exit(1)

        ids = f["neighbors"][0]   # 只取 query 0，shape: (k,)
        print(f"✅ query 0 top-{len(ids)} ids: {ids[:10]} ...")

    with open(output_path, "w", encoding="utf-8") as out:
        ids_str = ",".join(str(int(v)) for v in ids)
        out.write(f"neighbors_top{len(ids)}_ids: {ids_str}\n")

    print(f"✅ 已保存至 {output_path}")


if __name__ == "__main__":
    hdf5_path   = sys.argv[1] if len(sys.argv) > 1 else "/tbase-project/vsag/scripts/hybrid_index/data/models/data/msmarco_gt/msmarco_alpha_0_4_k_200.hdf5"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "hdf5_topk.txt"
    read_query0_neighbors(hdf5_path, output_path)
