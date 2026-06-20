import matplotlib.pyplot as plt
import numpy as np

# =========================
# Confusion Matrix
# =========================
cm = np.array([
    [797, 1142],
    [147, 3218]
])

labels = ['Fake', 'TRUE']

# =========================
# Create Figure
# =========================
fig, ax = plt.subplots(figsize=(6, 5))

# White background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot confusion matrix
im = ax.imshow(
    cm,
    cmap='Pastel1',
    vmin=0,
    vmax=6000
)

# Labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

ax.set_xticklabels(labels, fontsize=13)
ax.set_yticklabels(labels, fontsize=13)

ax.set_xlabel('Predicted Label', fontsize=15)
ax.set_ylabel('Actual Label', fontsize=15)

ax.set_title('Confusion Matrix', fontsize=17)

# Add values inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i,
            f'{cm[i, j]}',
            ha='center',
            va='center',
            color='black',
            fontsize=15,
            fontweight='bold'
        )

# =========================
# Add grid lines between squares
# =========================
ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)

ax.grid(
    which='minor',
    color='black',
    linestyle='-',
    linewidth=2
)

ax.tick_params(which='minor', bottom=False, left=False)

# Color bar
cbar = fig.colorbar(
    im,
    fraction=0.046,
    pad=0.04
)

# Layout
plt.tight_layout()

# Save figure
plt.savefig(
    "confusion_matrix_only.png",
    dpi=300,
    bbox_inches='tight'
)

# Show plot
plt.show()