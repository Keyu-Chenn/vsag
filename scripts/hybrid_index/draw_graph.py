import matplotlib.pyplot as plt
import numpy as np

# Data
# baseline (hgraph + sindi)
baseline_sindi_recall = [0.933302, 0.969097, 0.983803, 0.990154]
baseline_sindi_qps = [75.3852, 66.0382, 59.7803, 53.9087]

# baseline (hgraph + hgraph)
baseline_hgraph_recall = [0.920968, 0.955085, 0.969032, 0.976428]
baseline_hgraph_qps = [47.7082, 40.384, 35.0847, 31.3367]

# hybrid graph (random ep)
hybrid_random_recall = [0.936117, 0.948621, 0.956679, 0.962804]
hybrid_random_qps = [80.792, 69.4339, 61.4063, 53.8778]

# hybrid graph (sindi eps)
hybrid_sindi_recall = [0.971875, 0.981554, 0.98684, 0.99042]
hybrid_sindi_qps = [62.8004, 54.9526, 49.4673, 44.174]

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot each line with markers
ax.plot(baseline_sindi_recall, baseline_sindi_qps,
        marker='o', linewidth=2, markersize=8,
        label='baseline (hgraph + sindi)', color='#2196F3', linestyle='-')

ax.plot(baseline_hgraph_recall, baseline_hgraph_qps,
        marker='s', linewidth=2, markersize=8,
        label='baseline (hgraph + hgraph)', color='#F44336', linestyle='-')

ax.plot(hybrid_random_recall, hybrid_random_qps,
        marker='^', linewidth=2, markersize=8,
        label='hybrid graph (random ep)', color='#4CAF50', linestyle='-')

ax.plot(hybrid_sindi_recall, hybrid_sindi_qps,
        marker='D', linewidth=2, markersize=8,
        label='hybrid graph (sindi eps)', color='#FF9800', linestyle='-')

# Labels and title
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('QPS', fontsize=13)
ax.set_title('Recall vs QPS (k=500)', fontsize=15, fontweight='bold')

# Grid
ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax.legend(fontsize=11, loc='upper right')

# Axis formatting
ax.set_xlim([0.91, 1.00])
ax.set_ylim([25, 90])

plt.tight_layout()
plt.savefig('recall_vs_qps_k500.png', dpi=150, bbox_inches='tight')
plt.show()
