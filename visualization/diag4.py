import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy import signal as sg

# Set style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
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
signal_x = np.sin(2*np.pi*8*t) + 0.5*np.sin(2*np.pi*15*t) + 0.3*np.sin(2*np.pi*25*t)
signal_x += 0.2 * np.random.normal(0, 1, size=len(t))
signal_x *= 1.7  # Amplify

# Y-axis: mid-range frequency with more noise
signal_y = 0.3*np.sin(2*np.pi*4*t) + 0.8*np.sin(2*np.pi*7*t)
signal_y += 0.4 * np.random.normal(0, 1, size=len(t))

# Z-axis: lower frequencies with transients
signal_z = np.sin(2*np.pi*3*t) + 0.4*np.sin(2*np.pi*10*t)
# Add some transient effects
signal_z[300:350] += 1.5 * np.sin(2*np.pi*20*t[300:350])
signal_z[700:750] += 1.2 * np.sin(2*np.pi*15*t[700:750])
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
        relevance_time[max(0, spot-5):min(len(t), spot+5)] = 0.3 * np.abs(signal[max(0, spot-5):min(len(t), spot+5)])

# Make some relevance negative to match your example
negative_spots = np.random.choice(range(len(t)), int(len(t)*0.3))
relevance_time[negative_spots] *= -1

# Frequency domain
freqs = np.fft.rfftfreq(len(t), t[1]-t[0])
signal_fft = np.abs(np.fft.rfft(signal))
signal_fft = signal_fft / np.max(signal_fft) * 0.8

# Create frequency domain relevance with positive and negative components
relevance_freq = np.zeros_like(signal_fft)

# Add peaks at characteristic frequencies
peak_freqs = [3, 10, 15, 20]
for freq in peak_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    width = max(1, int(len(freqs) * 0.005))
    if freq == 3:
        relevance_freq[idx-width:idx+width+1] = -0.3 * np.exp(-0.5 * ((np.arange(-width, width+1) / (width/2))**2))
    elif freq == 10:
        relevance_freq[idx-width:idx+width+1] = 0.8 * np.exp(-0.5 * ((np.arange(-width, width+1) / (width/2))**2))
    elif freq == 15:
        relevance_freq[idx-width:idx+width+1] = 0.5 * np.exp(-0.5 * ((np.arange(-width, width+1) / (width/2))**2))
    elif freq == 20:
        relevance_freq[idx-width:idx+width+1] = -0.4 * np.exp(-0.5 * ((np.arange(-width, width+1) / (width/2))**2))

# Add some background noise to the relevance
noise_indices = np.random.choice(range(len(relevance_freq)), size=int(len(relevance_freq)*0.1))
relevance_freq[noise_indices] = np.random.uniform(-0.1, 0.1, size=len(noise_indices))

# Time-frequency representation - improved version with better resolution
nperseg = 128  # Window length
noverlap = 64  # Overlap between windows

# Create time-frequency data for all signals
f, t_spec, Sxx_x = sg.spectrogram(signal_x, fs=500, nperseg=nperseg, noverlap=noverlap)
_, _, Sxx_y = sg.spectrogram(signal_y, fs=500, nperseg=nperseg, noverlap=noverlap)
_, _, Sxx_z = sg.spectrogram(signal_z, fs=500, nperseg=nperseg, noverlap=noverlap)

# Apply log scaling for better visibility
Sxx_x = np.log1p(Sxx_x)
Sxx_y = np.log1p(Sxx_y)
Sxx_z = np.log1p(Sxx_z)

# Normalize for visualization
Sxx_x = Sxx_x / np.max(Sxx_x)
Sxx_y = Sxx_y / np.max(Sxx_y)
Sxx_z = Sxx_z / np.max(Sxx_z)

# Create time-frequency relevance with clearer patterns
rel_tf = np.zeros_like(Sxx_z)

# Map from 0-30 Hz range to corresponding indices in f
max_freq_idx = np.argmin(np.abs(f - 30))
freq_indices = np.linspace(0, max_freq_idx, 30, dtype=int)

# Add localized patterns of relevance (both positive and negative)
# Pattern around 3Hz (low frequency) - negative relevance
band_idx = np.argmin(np.abs(f - 3))
rel_tf[band_idx-1:band_idx+2, :] = -0.3

# Pattern around 10Hz at t=0.6s - positive relevance
band_idx = np.argmin(np.abs(f - 10))
time_idx = np.argmin(np.abs(t_spec - 0.6))
width_t = 3
width_f = 2
rel_tf[band_idx-width_f:band_idx+width_f+1, time_idx-width_t:time_idx+width_t+1] = 0.8

# Pattern around 15Hz at t=1.4s - positive relevance
band_idx = np.argmin(np.abs(f - 15))
time_idx = np.argmin(np.abs(t_spec - 1.4))
width_t = 3
width_f = 2
rel_tf[band_idx-width_f:band_idx+width_f+1, time_idx-width_t:time_idx+width_t+1] = 0.9

# Pattern around 20Hz at t=0.3s - negative relevance
band_idx = np.argmin(np.abs(f - 20))
time_idx = np.argmin(np.abs(t_spec - 0.3))
width_t = 2
width_f = 2
rel_tf[band_idx-width_f:band_idx+width_f+1, time_idx-width_t:time_idx+width_t+1] = -0.6

# Add title with subtitle
fig.suptitle('Multi-Domain Explainability Pipeline for Vibration-Based Fault Detection', fontsize=16, y=0.98)
plt.figtext(0.5, 0.94, 'Transforming attributions across time, frequency, and time-frequency domains',
            ha='center', fontsize=12, style='italic')

# 1. Input data plots - tri-axial signals in 3 rows
ax_input_x = fig.add_subplot(gs[0, 0])
ax_input_x.plot(t, signal_x, 'b-', linewidth=1)
ax_input_x.set_title('Input: Tri-axial Vibration Signal')
ax_input_x.set_ylabel('X-axis')
ax_input_x.set_xlim(0, 2)
ax_input_x.set_xticklabels([])

ax_input_y = fig.add_subplot(gs[1, 0])
ax_input_y.plot(t, signal_y, 'g-', linewidth=1)
ax_input_y.set_ylabel('Y-axis')
ax_input_y.set_xlim(0, 2)
ax_input_y.set_xticklabels([])

ax_input_z = fig.add_subplot(gs[2, 0])
ax_input_z.plot(t, signal_z, 'r-', linewidth=1)
ax_input_z.set_ylabel('Z-axis')
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
layer_centers = h/2 + 0.2

for i, (pos, height) in enumerate(zip(layer_positions, layer_heights)):
    ax_model.add_patch(patches.Rectangle((pos-0.02, layer_centers-height/2), 0.04, height,
                                 linewidth=1, edgecolor='black', facecolor='#4472C4', alpha=0.7))

ax_model.text(0.5, 0.05, '1D-CNN-Wide Model', horizontalalignment='center',
          verticalalignment='center', fontsize=12, fontweight='bold')

# 3. Attribution Methods box
ax_attr = fig.add_subplot(gs[7:9, 0])
attr_methods = ['LRP', 'Occlusion', 'SmoothGrad', 'Gradient×Input']
y_positions = np.linspace(0.2, 0.8, len(attr_methods))
ax_attr.axis('off')
ax_attr.text(0.5, 0.9, 'Attribution Methods', horizontalalignment='center',
         verticalalignment='center', fontsize=12, fontweight='bold')

for i, method in enumerate(attr_methods):
    ax_attr.add_patch(patches.FancyBboxPatch((0.3, y_positions[i]-0.08), 0.4, 0.16,
                                    boxstyle=patches.BoxStyle("Round", pad=0.04),
                                    facecolor='lightgray', alpha=0.3, edgecolor='black'))
    ax_attr.text(0.5, y_positions[i], method, horizontalalignment='center',
             verticalalignment='center', fontsize=11)

# 4. Time Domain Attribution
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
ax_time.set_xlabel('Time (s)')
ax_time.legend(loc='upper right', fontsize=8)
ax_time.set_xlim(0, 2)

# 5. DFT Virtual Layer - SHIFTED DOWN
ax_dft = fig.add_subplot(gs[4:6, 1])
ax_dft.axis('off')
# Make sure the title is positioned correctly to avoid overlap with time domain x-axis
ax_dft.set_title('DFT Virtual Layer', pad=20)

# Draw DFT transformation diagram - shifted down by adjusting y positions
rect = patches.Rectangle((0.2, 0.2), 0.6, 0.6, linewidth=2, edgecolor='black',
                         facecolor='lightgray', alpha=0.3)
ax_dft.add_patch(rect)
# Add equation
ax_dft.text(0.5, 0.65, 'DFT Virtual Layer', horizontalalignment='center',
            verticalalignment='center', fontsize=12, fontweight='bold')
ax_dft.text(0.5, 0.45, r'$X_k = \sum_{n=0}^{N-1} x_n e^{-i2\pi kn/N}$', horizontalalignment='center',
            verticalalignment='center', fontsize=10)
ax_dft.text(0.5, 0.25, r'$R_{freq} = R_{time} \odot \frac{DFT(signal)}{|signal|}$', horizontalalignment='center',
            verticalalignment='center', fontsize=10)

# 6. Frequency Domain Signal and Attribution
ax_freq_sig = fig.add_subplot(gs[7, 1])
ax_freq_sig.plot(freqs[:len(freqs)//4], signal_fft[:len(freqs)//4], 'k-', alpha=0.7, linewidth=1, label='Magnitude')
ax_freq_sig.set_title('Frequency Domain Signal')
ax_freq_sig.set_ylabel('Magnitude')
ax_freq_sig.set_xlim(0, 30)
ax_freq_sig.set_xticklabels([])
ax_freq_sig.legend(loc='upper right', fontsize=8)

# Frequency Domain Attribution (positive/negative like time domain)
ax_freq_attr = fig.add_subplot(gs[8, 1])
pos_freq_rel = np.copy(relevance_freq[:len(freqs)//4])
neg_freq_rel = np.copy(relevance_freq[:len(freqs)//4])
pos_freq_rel[pos_freq_rel < 0] = 0
neg_freq_rel[neg_freq_rel > 0] = 0

ax_freq_attr.plot(freqs[:len(freqs)//4], np.zeros_like(freqs[:len(freqs)//4]), 'k-', linewidth=0.5)  # Zero line
ax_freq_attr.fill_between(freqs[:len(freqs)//4], 0, pos_freq_rel, color='r', alpha=0.6, label='Positive Attribution')
ax_freq_attr.fill_between(freqs[:len(freqs)//4], 0, neg_freq_rel, color='b', alpha=0.6, label='Negative Attribution')
ax_freq_attr.set_title('Frequency Domain Attribution')
ax_freq_attr.set_ylabel('Attribution')
ax_freq_attr.set_xlabel('Frequency (Hz)')
ax_freq_attr.set_xlim(0, 30)
ax_freq_attr.legend(loc='upper right', fontsize=8)

# 7. STFT Virtual Layer
ax_stft = fig.add_subplot(gs[0:3, 2])
ax_stft.axis('off')
ax_stft.set_title('STFT Virtual Layer')

# Draw STFT transformation diagram
rect = patches.Rectangle((0.2, 0.2), 0.6, 0.6, linewidth=2, edgecolor='black',
                         facecolor='lightgray', alpha=0.3)
ax_stft.add_patch(rect)
ax_stft.text(0.5, 0.65, 'STFT Virtual Layer', horizontalalignment='center',
            verticalalignment='center', fontsize=12, fontweight='bold')
ax_stft.text(0.5, 0.45, r'STFT$(x)[m,k] = \sum_{n} x[n]w[n-m]e^{-i2\pi kn/N}$', horizontalalignment='center',
            verticalalignment='center', fontsize=10)
ax_stft.text(0.5, 0.25, 'Window size=128, overlap=64', horizontalalignment='center',
            verticalalignment='center', fontsize=9)

# 8. Time-Frequency Domain Signal - improved visualization
ax_tf_sig = fig.add_subplot(gs[4:6, 2])
im = ax_tf_sig.pcolormesh(t_spec, f[:max_freq_idx], Sxx_z[:max_freq_idx],
                          shading='gouraud', cmap='viridis')
ax_tf_sig.set_title('Time-Frequency Domain Signal')
ax_tf_sig.set_ylabel('Frequency (Hz)')
ax_tf_sig.set_xticklabels([])
ax_tf_sig.set_ylim(0, 30)
cbar = plt.colorbar(im, ax=ax_tf_sig)
cbar.set_label('Power')

# 9. Time-Frequency Domain Attribution (with red-blue color map for better contrast)
ax_tf = fig.add_subplot(gs[7:9, 2])
# Create a custom red-blue colormap similar to your example
cmap_rb = plt.cm.RdBu_r
im = ax_tf.pcolormesh(t_spec, f[:max_freq_idx], rel_tf[:max_freq_idx],
                     shading='gouraud', cmap=cmap_rb, vmin=-1, vmax=1)
ax_tf.set_title('Time-Frequency Domain Attribution')
ax_tf.set_ylabel('Frequency (Hz)')
ax_tf.set_xlabel('Time (s)')
ax_tf.set_ylim(0, 30)
cbar = plt.colorbar(im, ax=ax_tf)
cbar.set_label('Attribution Strength')

# 10. Domain Comparison Row
ax_compare = fig.add_subplot(gs[11:13, :])
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
    rect = patches.Rectangle((x_pos-0.12, 0.2), 0.24, 0.6, linewidth=1,
                            edgecolor='black', facecolor=color, alpha=box_alpha)
    ax_compare.add_patch(rect)
    ax_compare.text(x_pos, 0.65, domain, horizontalalignment='center',
                  verticalalignment='center', fontsize=12, fontweight='bold')
    ax_compare.text(x_pos, 0.4, desc, horizontalalignment='center',
                  verticalalignment='center', fontsize=10)

# Add arrows connecting components
# Input to Model
fig.add_artist(patches.FancyArrowPatch(
    (0.16, gs[2].get_position(fig).y0),
    (0.16, gs[4].get_position(fig).y1),
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

# Model to Attribution
fig.add_artist(patches.FancyArrowPatch(
    (0.16, gs[6].get_position(fig).y0),
    (0.16, gs[7].get_position(fig).y1),
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

# Attribution to Time Domain (highlighted for Z axis)
fig.add_artist(patches.FancyArrowPatch(
    (0.25, gs[7].get_position(fig).y0),
    (0.4, gs[2].get_position(fig).y0),
    connectionstyle="arc3,rad=0.3",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='r',  # Using red to match Z-axis
    alpha=arrow_alpha
))

# Time Domain to DFT - ADJUSTED
fig.add_artist(patches.FancyArrowPatch(
    (0.5, gs[3].get_position(fig).y0),
    (0.5, gs[4].get_position(fig).y1),  # Connect to top of DFT box
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

# DFT to Frequency
fig.add_artist(patches.FancyArrowPatch(
    (0.5, gs[6].get_position(fig).y0),
    (0.5, gs[7].get_position(fig).y1),
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

# Attribution to STFT
fig.add_artist(patches.FancyArrowPatch(
    (0.22, gs[7].get_position(fig).y0),
    (0.7, gs[2].get_position(fig).y0),
    connectionstyle="arc3,rad=-0.3",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color=time_freq_color,
    alpha=arrow_alpha
))

# STFT to Time-Frequency
fig.add_artist(patches.FancyArrowPatch(
    (0.83, gs[3].get_position(fig).y0),
    (0.83, gs[4].get_position(fig).y1),
    connectionstyle="arc3,rad=0",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=2,
    color='black',
    alpha=arrow_alpha
))

plt.tight_layout()
plt.subplots_adjust(hspace=0.8, wspace=0.3, top=0.92)
plt.savefig('multi_domain_explainability_pipeline.png', dpi=600, bbox_inches='tight')
plt.savefig('multi_domain_explainability_pipeline.pdf', bbox_inches='tight')
plt.show()