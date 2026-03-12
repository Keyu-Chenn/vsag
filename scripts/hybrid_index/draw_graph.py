import matplotlib.pyplot as plt
import numpy as np

# Data - alpha = 0.8
recall_hybrid = [0.9904, 0.9878, 0.9833, 0.9765, 0.962625]
qps_hybrid = [158, 182, 216, 256, 373]

recall_baseline = [0.9927, 0.9905, 0.9871, 0.9813, 0.9775]
qps_baseline = [127, 139, 173, 227, 246]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(recall_hybrid, qps_hybrid,
        color='#2196F3', marker='o', markersize=8,
        linewidth=2.5, label='Hybrid Graph (α=0.8)',
        markerfacecolor='white', markeredgewidth=2.5)

ax.plot(recall_baseline, qps_baseline,
        color='#FF5722', marker='s', markersize=8,
        linewidth=2.5, label='Baseline',
        markerfacecolor='white', markeredgewidth=2.5)

# Add data point annotations - hybrid (上方)
for i, (r, q) in enumerate(zip(recall_hybrid, qps_hybrid)):
    ax.annotate(f'({r:.4f}, {q})',
                xy=(r, q),
                xytext=(5, 8),
                textcoords='offset points',
                fontsize=8, color='#2196F3')

# Add data point annotations - baseline (下方)
for i, (r, q) in enumerate(zip(recall_baseline, qps_baseline)):
    ax.annotate(f'({r:.4f}, {q})',
                xy=(r, q),
                xytext=(5, -15),
                textcoords='offset points',
                fontsize=8, color='#FF5722')

# Formatting
ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
ax.set_ylabel('QPS (Queries Per Second)', fontsize=13, fontweight='bold')
ax.set_title('QPS vs Recall Curve\n(α = 0.8)', fontsize=15, fontweight='bold', pad=15)

ax.legend(fontsize=11, loc='upper right', framealpha=0.9,
          edgecolor='gray', fancybox=True)

ax.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')

# Set axis ranges with padding
ax.set_xlim(0.958, 0.997)
ax.set_ylim(100, 410)

# Minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.savefig('qps_recall_curve_alpha08.png', dpi=150, bbox_inches='tight')
plt.show()

print("图表已保存为 qps_recall_curve_alpha08.png")
