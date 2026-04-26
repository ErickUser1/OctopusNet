"""
OctopusNet architecture diagram.
Run: python plot_architecture.py
Output: architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Colors
C_MOD = ['#3b82f6', '#22c55e', '#f97316', '#a855f7']  # blue, green, orange, purple
C_NERVE = '#eab308'    # yellow
C_COORD = '#ef4444'    # red
C_INPUT = '#94a3b8'    # gray
C_TEXT = '#f1f5f9'
C_DIM = '#94a3b8'

def box(ax, x, y, w, h, color, label, sublabel=None, alpha=0.85, radius=0.15):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=1.5, edgecolor=color,
                          facecolor=color + '33', zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.18, label, ha='center', va='center',
                color=C_TEXT, fontsize=9, fontweight='bold', zorder=4)
        ax.text(x, y - 0.22, sublabel, ha='center', va='center',
                color=C_DIM, fontsize=7.5, zorder=4)
    else:
        ax.text(x, y, label, ha='center', va='center',
                color=C_TEXT, fontsize=9, fontweight='bold', zorder=4)

def arrow(ax, x1, y1, x2, y2, color, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, connectionstyle='arc3,rad=0.0'),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.12, my, label, color=C_DIM, fontsize=7, va='center', zorder=5)

# ── Input image ──────────────────────────────────────────────
box(ax, 1.4, 5.0, 1.6, 0.8, C_INPUT, 'Input Image', 'CIFAR-10\n32×32 RGB')

# ── 4 Modules ─────────────────────────────────────────────────
module_labels = [
    ('Module 1', 'kernel 3×3\nres 32×32'),
    ('Module 2', 'kernel 5×5\nres 16×16'),
    ('Module 3', 'kernel 7×7\nres 8×8'),
    ('Module 4', 'kernel 9×9\nres 4×4'),
]
module_ys = [8.0, 6.2, 4.2, 2.4]
mod_x = 5.2

for i, (lbl, sub) in enumerate(module_labels):
    box(ax, mod_x, module_ys[i], 2.2, 1.1, C_MOD[i], lbl, sub)
    # FF loss badge
    ax.text(mod_x + 1.35, module_ys[i] + 0.55, 'FF loss ✓',
            color=C_MOD[i], fontsize=6.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#0d1117',
                      edgecolor=C_MOD[i], linewidth=0.8), zorder=5)
    # Arrow from input
    arrow(ax, 2.2, 5.0, mod_x - 1.1, module_ys[i], C_MOD[i])

# ── Nerve Ring ────────────────────────────────────────────────
nerve_x = 9.5
nerve_y = 5.0
box(ax, nerve_x, nerve_y, 2.4, 1.2, C_NERVE, 'Nerve Ring', 'Cross-Attention')

for i in range(4):
    arrow(ax, mod_x + 1.1, module_ys[i], nerve_x - 1.2, nerve_y,
          C_MOD[i], f'h_{i+1}  (64d)')

# ── Coordinator ───────────────────────────────────────────────
coord_x = 13.0
coord_y = 5.0
box(ax, coord_x, coord_y, 2.2, 1.8, C_COORD,
    'Central\nCoordinator',
    'attention α\nh_agg = Σαᵢ·hᵢ\'')

arrow(ax, nerve_x + 1.2, nerve_y, coord_x - 1.1, coord_y, C_NERVE, "h'₁..h'₄")

# ── Classification head ───────────────────────────────────────
out_x = 15.2
out_y = 5.0
box(ax, out_x, out_y, 1.2, 0.7, C_COORD, '10 classes')
arrow(ax, coord_x + 1.1, coord_y, out_x - 0.6, out_y, C_COORD, 'logits')

# ── "No global backprop" brace ────────────────────────────────
ax.annotate('', xy=(6.4, 1.4), xytext=(3.1, 1.4),
            arrowprops=dict(arrowstyle='<->', color='#475569', lw=1.2))
ax.text(4.75, 1.15, 'local FF loss only\n(no global backprop)',
        ha='center', color='#475569', fontsize=7.5, style='italic')

# ── Legend ────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_MOD[0]+'33', edgecolor=C_MOD[0], label='Module 1 (3×3)'),
    mpatches.Patch(facecolor=C_MOD[1]+'33', edgecolor=C_MOD[1], label='Module 2 (5×5)'),
    mpatches.Patch(facecolor=C_MOD[2]+'33', edgecolor=C_MOD[2], label='Module 3 (7×7)'),
    mpatches.Patch(facecolor=C_MOD[3]+'33', edgecolor=C_MOD[3], label='Module 4 (9×9)'),
    mpatches.Patch(facecolor=C_NERVE+'33', edgecolor=C_NERVE, label='Nerve Ring'),
    mpatches.Patch(facecolor=C_COORD+'33', edgecolor=C_COORD, label='Coordinator'),
]
leg = ax.legend(handles=legend_items, loc='lower right',
                framealpha=0.15, facecolor='#1e293b',
                edgecolor='#334155', labelcolor=C_TEXT,
                fontsize=8, handlelength=1.2)

# ── Title ─────────────────────────────────────────────────────
ax.text(8.0, 9.5, 'OctopusNet Architecture',
        ha='center', color=C_TEXT, fontsize=14, fontweight='bold')
ax.text(8.0, 9.1, 'Modular Forward-Forward Network  |  CIFAR-10: 53.16%',
        ha='center', color=C_DIM, fontsize=9)

plt.tight_layout()
plt.savefig('/mnt/c/octopusnet/architecture.png', dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: architecture.png")
