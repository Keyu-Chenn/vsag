import matplotlib.pyplot as plt

# 数据
baseline_recall = [0.85507, 0.977709, 0.994549, 0.99819]
baseline_qps = [572.99, 346.141, 252.213, 199.175]

hybrid_bkef_recall = [0.941878, 0.990649, 0.996539, 0.998559]
hybrid_bkef_qps = [1065.39, 614.489, 405.43, 304.287]

hybrid_cset_recall = [0.78911, 0.82422, 0.86545, 0.88818, 0.90862, 0.933999]
hybrid_cset_qps = [1655.47, 1566.49, 1438.86, 1354.42, 1252.72, 1086.33]

labels_bf = ['ef=100', 'ef=200', 'ef=300', 'ef=400']
labels_cset = ['c_set=1', 'c_set=2', 'c_set=5', 'c_set=10', 'c_set=20', 'c_set=50']

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(baseline_recall, baseline_qps, 'o-', label='Baseline', color='blue')
ax.plot(hybrid_bkef_recall, hybrid_bkef_qps, 's-', label='Hybrid', color='red')
ax.plot(hybrid_cset_recall, hybrid_cset_qps, '^-', label='Hybrid fix_cset', color='green')


ax.set_xlabel('Recall')
ax.set_ylabel('QPS')
ax.set_title('QPS-Recall Curve  (α=0.2)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('qps_recall.png', dpi=150)
plt.show()
