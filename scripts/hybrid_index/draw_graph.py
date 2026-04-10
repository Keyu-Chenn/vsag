import matplotlib.pyplot as plt

# Data
alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
recall = [0.962533, 0.969417, 0.964658, 0.961659, 0.966908, 0.974648, 0.979383, 0.979892, 0.978503, 0.976883, 0.975522]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(alpha, recall, marker='o', linewidth=2, markersize=6, label='baseline')

# Labels and title
plt.xlabel('Alpha')
plt.ylabel('Recall')
plt.title('baseline')
plt.legend()
plt.grid(True)
plt.xticks(alpha)

plt.tight_layout()
plt.savefig('baseline_recall.png', dpi=300, bbox_inches='tight')
plt.show()
