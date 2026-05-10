import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# ── Data ──────────────────────────────────────────────────────────────────────
data = {
    'α=0.2 (prune=0.5)': {
        'baseline': {'recall': [0.85496,  0.977679, 0.994519, 0.99821],
                     'qps':    [565,      346.256,  243.787,  195.492]},
        'hybrid':   {'recall': [0.941878, 0.990649, 0.996539, 0.998559],
                     'qps':    [1070.43,  584.582,  397.878,  295.764]}
    },
    'α=0.3 (prune=0.5)': {
        'baseline': {'recall': [0.83169,  0.964059, 0.988879, 0.995549],
                     'qps':    [554.568,  342.346,  250.646,  197.5]},
        'hybrid':   {'recall': [0.918109, 0.982208, 0.993468, 0.996939],
                     'qps':    [864.181,  459.363,  316.498,  240.791]}
    },
    'α=0.5 (prune=0.6)': {
        'baseline': {'recall': [0.84587,  0.967478, 0.987218, 0.993058],
                     'qps':    [548.095,  333.068,  248.707,  198.074]},
        'hybrid':   {'recall': [0.942968, 0.980268, 0.990029, 0.993979],
                     'qps':    [530.265,  325.17,   249.55,   205.68]}
    },
    'α=0.7 (prune=0.6)': {
        'baseline': {'recall': [0.88642,  0.971948, 0.984688, 0.990248],
                     'qps':    [553.886,  336.885,  246.621,  196.488]},
        'hybrid':   {'recall': [0.929619, 0.971758, 0.984848, 0.990098],
                     'qps':    [577.095,  352.017,  257.9,    210.262]}
    },
    'α=0.8 (prune=0.6)': {
        'baseline': {'recall': [0.902599, 0.969538, 0.982818, 0.988859],
                     'qps':    [560.52,   346.111,  252.154,  200.452]},
        'hybrid':   {'recall': [0.924109, 0.968548, 0.982488, 0.988878],
                     'qps':    [585.402,  349.129,  259.788,  210.034]}
    }
}

style_handles = [
    Line2D([0], [0], color='#e41a1c', linestyle='-',  linewidth=2,
           marker='o', markersize=8,
           markerfacecolor='white', markeredgewidth=2,
           label='Baseline'),
    Line2D([0], [0], color='#377eb8', linestyle='--', linewidth=2,
           marker='*', markersize=12,
           label='Hybrid Graph'),
]

# ── One figure per alpha ───────────────────────────────────────────────────────
for i, (alpha, vals) in enumerate(data.items()):
    fig, ax = plt.subplots(figsize=(7, 5))

    b_recall = vals['baseline']['recall']
    b_qps    = vals['baseline']['qps']
    h_recall = vals['hybrid']['recall']
    h_qps    = vals['hybrid']['qps']

    # Baseline
    ax.plot(b_recall, b_qps,
            color='#e41a1c', linestyle='-', linewidth=2,
            marker='o', markersize=8,
            markerfacecolor='white', markeredgewidth=2,
            label='Baseline', zorder=3)

    # Hybrid Graph
    ax.plot(h_recall, h_qps,
            color='#377eb8', linestyle='--', linewidth=2,
            marker='*', markersize=12,
            label='Hybrid Graph', zorder=3)

    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_title(f'Recall vs QPS  ({alpha})', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    fname = f'recall_qps_{alpha.replace("=","").replace(" ","_").replace("(","").replace(")","").replace(".","")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存：{fname}")
