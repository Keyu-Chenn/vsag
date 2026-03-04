import matplotlib.pyplot as plt

# alpha = 1 混合图数据
hybrid_ef     = [200, 100, 50, 25, 10]
hybrid_recall = [0.999469, 0.998969, 0.995875, 0.98968, 0.9334]
hybrid_qps    = [244, 406, 683, 1074, 1771]

# baseline (4k) 数据
base4k_ef     = [200, 100, 50, 25, 10]
base4k_recall = [0.984375, 0.973625, 0.973625, 0.973375, 0.973906]
base4k_qps    = [406, 496, 499, 495, 482]

# baseline (2k) 数据
base2k_ef     = [200, 100, 50, 25, 10]
base2k_recall = [0.983719, 0.963437, 0.939844, 0.940375, 0.940125]
base2k_qps    = [493, 675, 761, 786, 798]


fig, ax = plt.subplots(figsize=(10, 6))

# 绘制三条曲线
ax.plot(hybrid_recall, hybrid_qps,
        'o-', color='tomato', linewidth=2, markersize=6,
        label='Hybrid (alpha=1)')

ax.plot(base4k_recall, base4k_qps,
        's--', color='steelblue', linewidth=2, markersize=6,
        label='Baseline (4k)')

ax.plot(base2k_recall, base2k_qps,
        '^-.', color='seagreen', linewidth=2, markersize=6,
        label='Baseline (2k)')


# 坐标轴设置
ax.set_xlabel('Recall@10', fontsize=13)
ax.set_ylabel('QPS (queries/sec)', fontsize=13)
ax.set_title('QPS-Recall Curve: Hybrid(alpha=1) vs Baseline(4k) vs Baseline(2k)', fontsize=13)

ax.set_xlim(0.65, 1.005)
ax.set_ylim(0, 2000)

ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('qps_recall_alpha1.png', dpi=150)
plt.show()
