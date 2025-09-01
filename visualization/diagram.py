import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors
from scipy import signal as sg
import matplotlib as mpl

# Set style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Roboto']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14

# Create figure with appropriate size for academic paper
fig = plt.figure(figsize=(12, 10))
# Adjust the GridSpec to create more space between rows
gs = GridSpec(16, 3, figure=fig)

# Define colors
time_color = '#70AD47'  # green
freq_color = '#ED7D31'  # orange
time_freq_color = '#7030A0'  # purple
box_alpha = 0.15
arrow_alpha = 0.8

# Generate sample data for visualization
np.random.seed(42)
t = np.linspace(0, 2, 1000)

# Create more realistic and distinct signals for each axis
# X-axis: higher frequency components
signal_x = np.sin(2 * np.pi * 8 * t) + 0.5 * np.sin(2 * np.pi * 15 * t) + 0.3 * np.sin(2 * np.pi * 25 * t)
signal_x += 0.2 * np.random.normal(0, 1, size=len(t))
signal_x *= 1.7  # Amplify

# Y-axis: mid-range frequency with more noise
signal_y = 0.3 * np.sin(2 * np.pi * 4 * t) + 0.8 * np.sin(2 * np.pi * 7 * t)
signal_y += 0.4 * np.random.normal(0, 1, size=len(t))

# Z-axis: lower frequencies with transients
signal_z = np.sin(2 * np.pi * 3 * t) + 0.4 * np.sin(2 * np.pi * 10 * t)
# Add some transient effects
signal_z[300:350] += 1.5 * np.sin(2 * np.pi * 20 * t[300:350])
signal_z[700:750] += 1.2 * np.sin(2 * np.pi * 15 * t[700:750])
signal_z += 0.15 * np.random.normal(0, 1, size=len(t))
signal_z *= 1.3  # Amplify

# Choose signal_z for attribution visualization
signal = signal_z

# Relevance in time domain (simulated LRP output)
relevance_time = np.zeros_like(signal)
relevance_time[300:350] = np.abs(signal_z[300:350]) * 0.8
relevance_time[700:750] = np.abs(signal_z[700:750]) * 1.0

# Add some sparse relevance elsewhere
random_spots = np.random.choice(range(len(t)), 20)
for spot in random_spots:
    if spot not in range(300, 350) and spot not in range(700, 750):
        relevance_time[max(0, spot - 5):min(len(t), spot + 5)] = 0.3 * np.abs(
            signal[max(0, spot - 5):min(len(t), spot + 5)])

# Make some relevance negative to match your example
negative_spots = np.random.choice(range(len(t)), int(len(t) * 0.3))
relevance_time[negative_spots] *= -1

# Frequency domain
freqs = np.fft.rfftfreq(len(t), t[1] - t[0])
signal_fft = np.abs(np.fft.rfft(signal))
signal_fft = signal_fft / np.max(signal_fft) * 0.8

# Create frequency domain relevance with MUCH THINNER spikes (more like your example)
relevance_freq = np.zeros_like(signal_fft)

# Create very thin, spike-like relevance
peak_freqs = [3, 10, 15, 20]
for freq in peak_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    # Use much smaller width for thinner spikes
    width = max(1, int(len(freqs) * 0.001))  # Reduced from 0.005 to 0.001

    if freq == 3:
        relevance_freq[idx - width:idx + width + 1] = -0.003 * np.ones(2 * width + 1)  # Constant thin line
        relevance_freq[idx] = -0.004  # Spike at center
    elif freq == 10:
        relevance_freq[idx - width:idx + width + 1] = 0.003 * np.ones(2 * width + 1)  # Constant thin line
        relevance_freq[idx] = 0.004  # Spike at center
    elif freq == 15:
        relevance_freq[idx - width:idx + width + 1] = 0.003 * np.ones(2 * width + 1)  # Constant thin line
        relevance_freq[idx] = 0.004  # Spike at center
    elif freq == 20:
        relevance_freq[idx - width:idx + width + 1] = -0.003 * np.ones(2 * width + 1)  # Constant thin line
        relevance_freq[idx] = -0.004  # Spike at center

# Add some very small background noise
noise_indices = np.random.choice(range(len(relevance_freq)), size=int(len(relevance_freq) * 0.1))
relevance_freq[noise_indices] = np.random.uniform(-0.0005, 0.0005, size=len(noise_indices))

# Time-frequency representation
# Create time-frequency data for signal
f, t_spec, Sxx = sg.spectrogram(signal, fs=500, nperseg=128, noverlap=64)
Sxx = np.log1p(Sxx)  # Log scale for better visibility
Sxx = Sxx / np.max(Sxx)

# Create time-frequency relevance with horizontal bands (more like your example)
rel_tf = np.zeros_like(Sxx) - 0.05  # Start with very light negative base (light blue)

# Map from 0-30 Hz range to corresponding indices in f
max_freq_idx = np.argmin(np.abs(f - 30))
freq_indices = np.linspace(0, max_freq_idx, 30, dtype=int)

# Add horizontal bands of relevance - make them THINNER to match the reference image
# Band around 5Hz - positive relevance
band_idx = np.argmin(np.abs(f - 5))
rel_tf[band_idx:band_idx + 1, :] = 0.8  # Thinner positive band

# Band around 15Hz - positive relevance
band_idx = np.argmin(np.abs(f - 15))
rel_tf[band_idx:band_idx + 1, :] = 0.9  # Thinner positive band

# Band around 25Hz - positive relevance
band_idx = np.argmin(np.abs(f - 25))
rel_tf[band_idx:band_idx + 1, :] = 0.7  # Thinner positive band

# Add subtle negative regions
band_idx = np.argmin(np.abs(f - 10))
rel_tf[band_idx:band_idx + 1, :] = -0.3  # Thinner negative band

# 1. Input data plots - tri-axial signals in 3 rows with ALIGNED Y-AXIS LABELS
label_x_pos = -0.15  # Consistent position for all axis labels

ax_input_x = fig.add_subplot(gs[0, 0])
ax_input_x.plot(t, signal_x, 'b-', linewidth=1)
ax_input_x.set_title('Input: Tri-axial Vibration Signal')
ax_input_x.set_ylabel('X-axis', labelpad=15)
ax_input_x.yaxis.set_label_coords(label_x_pos, 0.5)  # Align all y-axis labels horizontally
ax_input_x.set_xlim(0, 2)
ax_input_x.set_xticklabels([])

ax_input_y = fig.add_subplot(gs[1, 0])
ax_input_y.plot(t, signal_y, 'g-', linewidth=1)
ax_input_y.set_ylabel('Y-axis', labelpad=15)
ax_input_y.yaxis.set_label_coords(label_x_pos, 0.5)  # Align all y-axis labels horizontally
ax_input_y.set_xlim(0, 2)
ax_input_y.set_xticklabels([])

ax_input_z = fig.add_subplot(gs[2, 0])
ax_input_z.plot(t, signal_z, 'r-', linewidth=1)
ax_input_z.set_ylabel('Z-axis', labelpad=15)
ax_input_z.yaxis.set_label_coords(label_x_pos, 0.5)  # Align all y-axis labels horizontally
ax_input_z.set_xlabel('Time (s)')
ax_input_z.set_xlim(0, 2)

# 2. CNN Model visualization
ax_model = fig.add_subplot(gs[4:6, 0])
ax_model.axis('off')
# Draw a simplified CNN architecture
h = 0.6
w = 0.8
ax_model.add_patch(patches.Rectangle((0.1, 0.2), w, h, linewidth=2, edgecolor='black',
                                     facecolor='#4472C4', alpha=0.2))

# Draw CNN layers as vertical lines of different heights
n_layers = 7
layer_positions = np.linspace(0.2, 0.8, n_layers)
layer_heights = np.array([0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2])
layer_centers = h / 2 + 0.2

for i, (pos, height) in enumerate(zip(layer_positions, layer_heights)):
    ax_model.add_patch(patches.Rectangle((pos - 0.02, layer_centers - height / 2), 0.04, height,
                                         linewidth=1, edgecolor='black', facecolor='#4472C4', alpha=0.7))

ax_model.text(0.5, 0.05, '1D-CNN Model', horizontalalignment='center',
              verticalalignment='center', fontsize=12, fontweight='bold')

# 3. Attribution Methods box with TITLE POSITIONED HIGHER
ax_attr = fig.add_subplot(gs[7:9, 0])
attr_methods = ['LRP', 'Occlusion', 'SmoothGrad', 'Gradient×Input']
y_positions = np.linspace(0.2, 0.8, len(attr_methods))
ax_attr.axis('off')

# Move title higher up from the box
ax_attr.text(0.5, 1.05, 'Attribution Methods', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')

for i, method in enumerate(attr_methods):
    ax_attr.add_patch(patches.FancyBboxPatch((0.3, y_positions[i] - 0.08), 0.4, 0.16,
                                             boxstyle=patches.BoxStyle("Round", pad=0.04),
                                             facecolor='lightgray', alpha=0.3, edgecolor='black'))
    ax_attr.text(0.5, y_positions[i], method, horizontalalignment='center',
                 verticalalignment='center', fontsize=11)

# 4. Time Domain Attribution with SMALLER LEGEND
ax_time = fig.add_subplot(gs[0:3, 1])
ax_time.plot(t, signal, 'k-', alpha=0.7, linewidth=1, label='Signal')
pos_rel = np.copy(relevance_time)
neg_rel = np.copy(relevance_time)
pos_rel[pos_rel < 0] = 0
neg_rel[neg_rel > 0] = 0

ax_time.fill_between(t, 0, pos_rel, color='r', alpha=0.5, label='Positive Attribution')
ax_time.fill_between(t, 0, neg_rel, color='b', alpha=0.5, label='Negative Attribution')
ax_time.set_title('Time Domain Attribution')
ax_time.set_ylabel('Attribution Strength')
ax_time.set_xlabel('Time (s)', labelpad=-1)

# Make legend smaller
ax_time.legend(loc='upper right', fontsize=6, frameon=False)  # Smaller font size and no frame
ax_time.set_xlim(0, 2)

# 5. DFT Virtual Layer
ax_dft = fig.add_subplot(gs[4:6, 1])
ax_dft.axis('off')
# Make sure the title is positioned correctly to avoid overlap with time domain x-axis
ax_dft.set_title('DFT Virtual Layer', pad=-1)

# Draw DFT transformation diagram - shifted down by adjusting y positions
rect = patches.Rectangle((0.2, 0.2), 0.6, 0.6, linewidth=2, edgecolor='black',
                         facecolor='lightgray', alpha=0.3)
ax_dft.add_patch(rect)
ax_dft.text(0.5, 0.5, 'DFT Virtual\nTransformation', horizontalalignment='center',
            verticalalignment='center', fontsize=12, fontweight='bold')

# 6. Frequency Domain Signal with SMALLER LEGEND
ax_freq_sig = fig.add_subplot(gs[7, 1])
ax_freq_sig.plot(freqs[:len(freqs) // 4], signal_fft[:len(freqs) // 4], 'k-', alpha=0.7, linewidth=1, label='Magnitude')
ax_freq_sig.set_title('Frequency Domain Signal')
ax_freq_sig.set_ylabel('Magnitude', labelpad=1)
ax_freq_sig.set_xlim(0, 30)
ax_freq_sig.set_xticklabels([])
ax_freq_sig.legend(loc='upper right', fontsize=6, frameon=False)  # Smaller font size and no frame

# Frequency Domain Attribution with SMALLER LEGEND
ax_freq_attr = fig.add_subplot(gs[8, 1])
pos_freq_rel = np.copy(relevance_freq[:len(freqs) // 4])
neg_freq_rel = np.copy(relevance_freq[:len(freqs) // 4])
pos_freq_rel[pos_freq_rel < 0] = 0
neg_freq_rel[neg_freq_rel > 0] = 0

# Plot with thinner lines to match reference image
ax_freq_attr.plot(freqs[:len(freqs) // 4], np.zeros_like(freqs[:len(freqs) // 4]), 'k-', linewidth=0.5)  # Zero line
ax_freq_attr.plot(freqs[:len(freqs) // 4], pos_freq_rel, 'r-', linewidth=0.5, label='Positive Attribution')
ax_freq_attr.plot(freqs[:len(freqs) // 4], neg_freq_rel, 'b-', linewidth=0.5, label='Negative Attribution')
ax_freq_attr.set_title('Frequency Domain Attribution')
ax_freq_attr.set_ylabel('Attribution')
ax_freq_attr.set_xlabel('Frequency (Hz)')
ax_freq_attr.set_xlim(0, 30)
ax_freq_attr.legend(loc='upper right', fontsize=6, frameon=False)  # Smaller font size and no frame

# Adjust y limits for better visibility of the thin spikes
ax_freq_attr.set_ylim(-0.005, 0.005)

# 7. STFT Virtual Layer
ax_stft = fig.add_subplot(gs[0:3, 2])
ax_stft.axis('off')
ax_stft.set_title('SDTFT Virtual Layer')

# Draw STFT transformation diagram
rect = patches.Rectangle((0.2, 0.2), 0.6, 0.6, linewidth=2, edgecolor='black',
                         facecolor='lightgray', alpha=0.3)
ax_stft.add_patch(rect)
ax_stft.text(0.5, 0.5, 'SDTFT Virtual\nTransformation', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')

# 8. Time-Frequency Domain Signal
ax_tf_sig = fig.add_subplot(gs[4:6, 2])
im = ax_tf_sig.pcolormesh(t_spec, f[:max_freq_idx], Sxx[:max_freq_idx], shading='gouraud', cmap='viridis')
ax_tf_sig.set_title('Time-Frequency Domain Signal')
ax_tf_sig.set_ylabel('Frequency (Hz)')
ax_tf_sig.set_xticklabels([])
ax_tf_sig.set_ylim(0, 30)
cbar = plt.colorbar(im, ax=ax_tf_sig, fraction=0.046, pad=0.04)  # Make colorbar smaller
cbar.set_label('Power', fontsize=8)
cbar.ax.tick_params(labelsize=6)  # Smaller tick labels on colorbar

# 9. Time-Frequency Domain Attribution with smaller colorbar
ax_tf = fig.add_subplot(gs[7:9, 2])

# Create custom colormap for better contrast with vivid red lines on light blue background
colors = [(0.2, 0.4, 0.8), (0.9, 0.9, 1.0), (1.0, 0.0, 0.0)]  # light blue, light gray, vivid red
positions = [0, 0.5, 1]
cmap_custom = mpl.colors.LinearSegmentedColormap.from_list("CustomMap", list(zip(positions, colors)))

# Use the custom colormap for visualization
im = ax_tf.pcolormesh(t_spec, f[:max_freq_idx], rel_tf[:max_freq_idx],
                      shading='gouraud', cmap=cmap_custom, vmin=-0.3, vmax=1)
ax_tf.set_title('Time-Frequency Domain Attribution')
ax_tf.set_ylabel('Frequency (Hz)')
ax_tf.set_xlabel('Time (s)')
ax_tf.set_ylim(0, 30)
cbar = plt.colorbar(im, ax=ax_tf, fraction=0.046, pad=0.04)  # Make colorbar smaller
cbar.set_label('Attribution Strength', fontsize=8)
cbar.ax.tick_params(labelsize=6)  # Smaller tick labels on colorbar

# 10. Domain Comparison Row - MOVED CLOSER TO VISUALIZATIONS
ax_compare = fig.add_subplot(gs[10:12, :])  # Changed from 11:13 to 10:12 to move boxes up
ax_compare.axis('off')

# Create domain comparison text
domains = ['Time Domain', 'Frequency Domain', 'Time-Frequency Domain']
descriptions = [
    'Identifies when in the signal\nfault patterns occur',
    'Reveals characteristic\nfrequencies of faults',
    'Shows how spectral patterns\nevolve over time'
]
colors = [time_color, freq_color, time_freq_color]

for i, (domain, desc, color) in enumerate(zip(domains, descriptions, colors)):
    x_pos = 0.16 + 0.33 * i
    rect = patches.Rectangle((x_pos - 0.12, 0.2), 0.24, 0.6, linewidth=1,
                             edgecolor='black', facecolor=color, alpha=box_alpha)
    ax_compare.add_patch(rect)
    ax_compare.text(x_pos, 0.65, domain, horizontalalignment='center',
                    verticalalignment='center', fontsize=12, fontweight='bold')
    ax_compare.text(x_pos, 0.4, desc, horizontalalignment='center',
                    verticalalignment='center', fontsize=10)

# Keep the arrow positioning as you have it in your code
# 1. Arrow from Z-axis to CNN model - vertical black arrow
fig.add_artist(patches.FancyArrowPatch(
    (0.17, gs[2].get_position(fig).y1 - 0.11),
    (0.17, gs[4].get_position(fig).y1 - 0.12),
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

# 2. Arrow from CNN Model to Attribution Methods - vertical black arrow
fig.add_artist(patches.FancyArrowPatch(
    (0.17, gs[5].get_position(fig).y1 - 0.20),
    (0.17, gs[7].get_position(fig).y1 - 0.21),
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

# 3. Arrow from Z-axis to Time Domain Attribution - curved red arrow
fig.add_artist(patches.FancyArrowPatch(
    (0.3, gs[2].get_position(fig).y0),  # Starting from Z-axis subplot
    (0.4, gs[1].get_position(fig).y0),  # To middle of time domain attribution
    connectionstyle="arc3,rad=-0.3",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='r',  # Red arrow matching Z-axis
    alpha=arrow_alpha
))

# 4. Arrow from Time Domain to DFT Virtual Layer - vertical black arrow
fig.add_artist(patches.FancyArrowPatch(
    (0.5, gs[2].get_position(fig).y1 - 0.09),  # Start below time domain
    (0.5, gs[4].get_position(fig).y1 - 0.10),  # End at top of DFT box
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

# 5. Arrow from DFT Virtual Layer to Frequency Domain - vertical black arrow
fig.add_artist(patches.FancyArrowPatch(
    (0.5, gs[5].get_position(fig).y1 - 0.20),  # Start below DFT
    (0.5, gs[7].get_position(fig).y1 - 0.21),  # End at top of frequency domain
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

# 6. Arrow from Time Domain to STFT Virtual Layer - curved purple arrow
fig.add_artist(patches.FancyArrowPatch(
    (0.6, gs[1].get_position(fig).y0 + 0.05),  # From time domain
    (0.8, gs[1].get_position(fig).y0),  # To STFT virtual layer
    connectionstyle="arc3,rad=0.3",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color=time_freq_color,  # Purple for time-frequency
    alpha=arrow_alpha
))

# 7. Arrow from STFT Virtual Layer to Time-Frequency Domain - vertical black arrow
fig.add_artist(patches.FancyArrowPatch(
    (0.85, gs[2].get_position(fig).y1 - 0.05),  # From bottom of STFT
    (0.85, gs[4].get_position(fig).y1 - 0.07),  # To top of time-frequency domain
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

plt.tight_layout()
plt.subplots_adjust(hspace=0.8, wspace=0.3)
plt.savefig('multi_domain_explainability_pipeline.png', dpi=1200, bbox_inches='tight')
plt.savefig('multi_domain_explainability_pipeline.pdf', bbox_inches='tight')
plt.show()