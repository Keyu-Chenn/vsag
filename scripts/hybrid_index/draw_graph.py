import matplotlib.pyplot as plt
import numpy as np

# ── 数据 ──────────────────────────────────────────────────────────────
recall_sindi  = [0.996312, 0.993594, 0.987688, 0.965125]
qps_sindi     = [154,      204,      290,      481     ]

recall_random = [0.982625, 0.975344, 0.963,    0.930094]
qps_random    = [171,      226,      313,      529     ]

recall_base   = [0.990531, 0.993625, 0.994719, 0.99675,  0.998156]
qps_base      = [379,      352,      309,      232,      229     ]

# ── 绘图 ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

datasets = [
        ("sindi eps",  recall_sindi,  qps_sindi,  "#E74C3C", "o", "-"),
        ("random ep",  recall_random, qps_random, "#3498DB", "s", "--"),
        ("baseline",   recall_base,   qps_base,   "#2ECC71", "^", "-."),
]

for label, recall, qps, color, marker, ls in datasets:
        # 按 QPS 升序排列，曲线走向从左到右
        order  = np.argsort(qps)
        r_sort = np.array(recall)[order]
        q_sort = np.array(qps)[order]

        ax.plot(q_sort, r_sort,
                color=color, marker=marker, linestyle=ls,
                linewidth=2, markersize=8, label=label)

        # 标注每个点
        for r, q in zip(recall, qps):
                ax.annotate(f"({q}, {r:.4f})",
                            xy=(q, r), xytext=(6, -13),
                            textcoords="offset points",
                            fontsize=7.5, color=color)

ax.set_xlabel("QPS (Queries Per Second)", fontsize=12)
ax.set_ylabel("Recall", fontsize=12)
ax.set_title("Recall vs QPS (alpha = 0.4)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_ylim(0.92, 1.005)

plt.tight_layout()
plt.savefig("recall_vs_qps.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ 图片已保存为 recall_vs_qps.png")
