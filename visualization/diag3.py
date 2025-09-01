import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Set up the figure with a light background
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
ax = plt.gca()
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Add title
plt.title('Multi-Domain Explainability Pipeline for Vibration-Based Fault Detection',
          fontsize=16, fontweight='bold', y=0.98)

# Create domain columns
domains = ['Time Domain', 'Frequency Domain', 'Time-Frequency Domain']
domain_x = [1.5, 6, 10.5]  # X-center of each domain
domain_width = 3
domain_height = 7

# Draw domain backgrounds
for i, domain in enumerate(domains):
    # Domain background
    rect = patches.Rectangle((domain_x[i] - domain_width / 2, 0.5),
                             domain_width, domain_height,
                             linewidth=1, edgecolor='gray',
                             facecolor='#f5f5f5', alpha=0.7,
                             zorder=0)
    ax.add_patch(rect)

    # Domain title
    plt.text(domain_x[i], 7.8, domain,
             ha='center', va='center',
             fontsize=14, fontweight='bold')


# Function to add a box with text
def add_box(x, y, text, box_type='normal', width=2.5, height=0.8):
    colors = {
        'normal': ('#ffffff', '#000000'),  # white bg, black edge
        'model': ('#e6f3ff', '#0066cc'),  # light blue
        'transform': ('#fff2e6', '#ff6600'),  # light orange
        'method': ('#e6ffe6', '#008800')  # light green
    }

    bg_color, edge_color = colors.get(box_type, colors['normal'])

    rect = patches.Rectangle((x - width / 2, y - height / 2),
                             width, height,
                             linewidth=1, edgecolor=edge_color,
                             facecolor=bg_color, alpha=0.9,
                             zorder=1)
    ax.add_patch(rect)

    # Split text by newline and position each line
    lines = text.split('\n')
    line_height = 0.25
    start_y = y + (len(lines) - 1) * line_height / 2

    for i, line in enumerate(lines):
        line_y = start_y - i * line_height
        plt.text(x, line_y, line,
                 ha='center', va='center',
                 fontsize=10, color='black')


# Function to add an arrow between boxes
def add_arrow(start_x, start_y, end_x, end_y, label='', arrow_type='normal'):
    colors = {
        'normal': '#666666',
        'transform': '#ff6600'
    }
    color = colors.get(arrow_type, colors['normal'])

    # Create arrow with a curve if needed
    if start_x != end_x:
        # Curved arrow for cross-domain connections
        verts = [
            (start_x + 0.5, start_y),  # start point (right side of box)
            (start_x + 1.0, start_y),  # control point
            ((start_x + 1.0 + end_x - 0.5) / 2, (start_y + end_y) / 2),  # control point
            (end_x - 0.5, end_y),  # end point (left side of box)
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        # Straight arrow for same-domain connections
        verts = [
            (start_x, start_y - 0.4),  # start point (bottom of box)
            (start_x, end_y + 0.4),  # end point (top of box)
        ]
        codes = [Path.MOVETO, Path.LINETO]

    path = Path(verts, codes)
    #patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=1.5, arrowstyle='->', zorder=1)


    # ax.add_patch(patch)

    # Create the path
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                              lw=1.5, zorder=1)
    ax.add_patch(patch)

    # Add arrow head
    if start_x != end_x:
        # For curved paths - add arrow at the end
        arrow = patches.FancyArrowPatch(verts[-2], verts[-1],
                                        connectionstyle="arc3",
                                        arrowstyle='->',
                                        color=color,
                                        lw=1.5,
                                        mutation_scale=15,
                                        zorder=2)
    else:
        # For straight lines
        arrow = patches.FancyArrowPatch(verts[0], verts[1],
                                        arrowstyle='->',
                                        color=color,
                                        lw=1.5,
                                        mutation_scale=15,
                                        zorder=2)
    ax.add_patch(arrow)



    # Add label if provided
    if label and start_x != end_x:
        mid_x = (verts[0][0] + verts[-1][0]) / 2
        mid_y = (verts[0][1] + verts[-1][1]) / 2
        plt.text(mid_x, mid_y + 0.2, label,
                 ha='center', va='center',
                 fontsize=10, color=color)


# Add components to the time domain
add_box(domain_x[0], 7, "Input Vibration Signal\nx(t) = [xₓ(t), x_y(t), x_z(t)]")
add_box(domain_x[0], 5.5, "CNN1D-Wide Model\nConvolutional layers + Pooling", box_type='model')
add_box(domain_x[0], 4, "Layer-wise Relevance Propagation\nR_i^(l) = ∑_j (z_ij/∑_i z_ij) R_j^(l+1)", box_type='method')
add_box(domain_x[0], 2.5, "Time-Domain Attribution\nR_time(t)")
add_box(domain_x[0], 1.5, "Temporal patterns:\nWhen the fault occurs")

# Add components to the frequency domain
add_box(domain_x[1], 7, "DFT Virtual Layer\nX(f) = ℱ{x(t)}", box_type='transform')
add_box(domain_x[1], 5.5, "Frequency Representation\nMagnitude & Phase Spectrum")
add_box(domain_x[1], 4, "DFT-LRP\nR_freq(f) = R_time(t) ⊙ X(f)/|x(t)|", box_type='method')
add_box(domain_x[1], 2.5, "Frequency-Domain Attribution\nR_freq(f)")
add_box(domain_x[1], 1.5, "Spectral patterns:\nWhich frequencies indicate faults")

# Add components to the time-frequency domain
add_box(domain_x[2], 7, "STFT Virtual Layer\nX(t,f) = STFT{x(t)}", box_type='transform')
add_box(domain_x[2], 5.5, "Time-Frequency Representation\nSpectrogram")
add_box(domain_x[2], 4, "STFT-LRP\nJoint Time-Frequency Relevance", box_type='method')
add_box(domain_x[2], 2.5, "Time-Frequency Attribution\nR_tf(t,f)")
add_box(domain_x[2], 1.5, "Joint patterns:\nWhen and which frequencies")

# Connect components within domains with arrows
for x in domain_x:
    add_arrow(x, 7, x, 5.5)
    add_arrow(x, 5.5, x, 4)
    add_arrow(x, 4, x, 2.5)
    add_arrow(x, 2.5, x, 1.5)

# Connect domains with transformation arrows
add_arrow(domain_x[0], 2.5, domain_x[1], 7, "DFT", arrow_type='transform')
add_arrow(domain_x[0], 2.5, domain_x[2], 7, "STFT", arrow_type='transform')

# Save and show the figure
plt.savefig("xai_pipeline.png", dpi=300, bbox_inches='tight')
plt.savefig("xai_pipeline.pdf", bbox_inches='tight')
plt.show()