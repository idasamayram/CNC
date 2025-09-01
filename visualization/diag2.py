import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patches as patches
from scipy import signal
import matplotlib.colors as colors

# Set styling for professional appearance
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14
})

# Create figure with proper dimensions
fig = plt.figure(figsize=(15, 8))
gs = GridSpec(7, 3, height_ratios=[0.3, 1, 0.5, 0.3, 1, 1, 0.5], width_ratios=[1, 1, 1])
fig.suptitle('Multi-Domain Explainability Pipeline for Vibration-Based Fault Detection', fontsize=16)

# Generate sample data for visualization
np.random.seed(42)
t = np.linspace(0, 5, 1000)
# Create tri-axial vibration signal with a fault pattern
x_axis = np.sin(2 * np.pi * t) + 0.2 * np.random.randn(len(t))
y_axis = np.sin(2 * np.pi * t + np.pi / 4) + 0.2 * np.random.randn(len(t))
z_axis = np.sin(2 * np.pi * t + np.pi / 2) + 0.2 * np.random.randn(len(t))

# Add a fault pattern in a specific region
fault_region = (t > 2) & (t < 3)
x_axis[fault_region] += 0.8 * np.sin(2 * np.pi * 10 * t[fault_region])
y_axis[fault_region] += 0.6 * np.sin(2 * np.pi * 10 * t[fault_region])
z_axis[fault_region] += 0.7 * np.sin(2 * np.pi * 10 * t[fault_region])

# Create mock relevance scores based on the fault region
time_relevance_x = np.zeros_like(x_axis)
time_relevance_y = np.zeros_like(y_axis)
time_relevance_z = np.zeros_like(z_axis)

time_relevance_x[fault_region] = 0.8 + 0.2 * np.random.rand(np.sum(fault_region))
time_relevance_y[fault_region] = 0.7 + 0.2 * np.random.rand(np.sum(fault_region))
time_relevance_z[fault_region] = 0.6 + 0.2 * np.random.rand(np.sum(fault_region))

# Compute frequency domain representations
fs = len(t) / t[-1]  # Sampling frequency
freqs_x = np.fft.rfftfreq(len(x_axis), 1 / fs)
fft_x = np.fft.rfft(x_axis)
fft_y = np.fft.rfft(y_axis)
fft_z = np.fft.rfft(z_axis)

# Create frequency domain relevance (focused around the 10Hz component)
freq_relevance_x = np.zeros_like(freqs_x)
freq_relevance_y = np.zeros_like(freqs_x)
freq_relevance_z = np.zeros_like(freqs_x)

fault_freq_idx = np.abs(freqs_x - 10).argmin()
freq_window = np.exp(-0.5 * ((freqs_x - 10) / 1) ** 2)
freq_relevance_x = freq_window
freq_relevance_y = 0.9 * freq_window
freq_relevance_z = 0.8 * freq_window

# Compute time-frequency representations
nperseg = 128
f_tf, t_tf, Sxx_x = signal.spectrogram(x_axis, fs, nperseg=nperseg)
_, _, Sxx_y = signal.spectrogram(y_axis, fs, nperseg=nperseg)
_, _, Sxx_z = signal.spectrogram(z_axis, fs, nperseg=nperseg)

# Create time-frequency relevance maps
tf_relevance_x = np.zeros_like(Sxx_x)
tf_relevance_y = np.zeros_like(Sxx_y)
tf_relevance_z = np.zeros_like(Sxx_z)

# Find time indices corresponding to fault region
t_fault_indices = np.where((t_tf >= 2) & (t_tf <= 3))[0]
# Find frequency indices around 10Hz
f_fault_indices = np.where((f_tf >= 8) & (f_tf <= 12))[0]

for i in t_fault_indices:
    for j in f_fault_indices:
        tf_relevance_x[j, i] = 0.8 + 0.2 * np.random.rand()
        tf_relevance_y[j, i] = 0.7 + 0.2 * np.random.rand()
        tf_relevance_z[j, i] = 0.6 + 0.2 * np.random.rand()

# Create colormap for relevance visualization
cmap = plt.cm.RdBu_r

# ----- TIME DOMAIN SECTION -----
# Title for time domain column
ax_title_time = fig.add_subplot(gs[0, 0])
ax_title_time.text(0.5, 0.5, 'Time Domain', ha='center', va='center', fontweight='bold')
ax_title_time.axis('off')

# Input signal visualization
ax_input = fig.add_subplot(gs[1, 0])
ax_input.plot(t, x_axis, 'r-', label='X-axis', alpha=0.8)
ax_input.plot(t, y_axis, 'g-', label='Y-axis', alpha=0.8)
ax_input.plot(t, z_axis, 'b-', label='Z-axis', alpha=0.8)
ax_input.set_title('Input Vibration Signal')
ax_input.set_ylabel('Amplitude')
ax_input.set_xlabel('Time (s)')
ax_input.legend(loc='upper right', frameon=False)
ax_input.grid(alpha=0.3)

# CNN Model visualization (simplified)
ax_model = fig.add_subplot(gs[2, 0])
# Draw boxes for CNN layers
layer_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
layer_labels = ['Conv1D', 'Conv1D', 'Conv1D', 'Linear']
for i, (color, label) in enumerate(zip(layer_colors, layer_labels)):
    rect = Rectangle((i * 0.25, 0.2), 0.2, 0.6, facecolor=color, alpha=0.7)
    ax_model.add_patch(rect)
    ax_model.text(i * 0.25 + 0.1, 0.1, label, ha='center', fontsize=9)
ax_model.text(0.5, 0.9, 'CNN1D-Wide Model', ha='center', fontweight='bold')
ax_model.set_xlim(0, 1)
ax_model.set_ylim(0, 1)
ax_model.set_xticks([])
ax_model.set_yticks([])
ax_model.spines['top'].set_visible(False)
ax_model.spines['right'].set_visible(False)
ax_model.spines['bottom'].set_visible(False)
ax_model.spines['left'].set_visible(False)

# Time domain attribution visualization
ax_time_attr = fig.add_subplot(gs[4, 0])
ax_time_attr.plot(t, time_relevance_x, 'r-', label='X-axis', alpha=0.8)
ax_time_attr.plot(t, time_relevance_y, 'g-', label='Y-axis', alpha=0.8)
ax_time_attr.plot(t, time_relevance_z, 'b-', label='Z-axis', alpha=0.8)
ax_time_attr.set_title('Time-Domain Attribution')
ax_time_attr.set_ylabel('Relevance')
ax_time_attr.set_xlabel('Time (s)')
ax_time_attr.grid(alpha=0.3)
ax_time_attr.set_ylim(0, 1.1)

# ----- FREQUENCY DOMAIN SECTION -----
# Title for frequency domain column
ax_title_freq = fig.add_subplot(gs[0, 1])
ax_title_freq.text(0.5, 0.5, 'Frequency Domain', ha='center', va='center', fontweight='bold')
ax_title_freq.axis('off')

# DFT Virtual Layer visualization
ax_dft = fig.add_subplot(gs[1, 1])
ax_dft.text(0.5, 0.7, 'DFT Virtual Layer', ha='center', fontweight='bold', fontsize=12)
ax_dft.text(0.5, 0.4, r'$X_k = \sum_{n=0}^{N-1} x_n e^{-i2\pi kn/N}$', ha='center', fontsize=12)
ax_dft.text(0.5, 0.2, 'Transformation Matrix', ha='center', fontsize=10)
ax_dft.set_xlim(0, 1)
ax_dft.set_ylim(0, 1)
ax_dft.set_xticks([])
ax_dft.set_yticks([])
ax_dft.spines['top'].set_visible(False)
ax_dft.spines['right'].set_visible(False)
ax_dft.spines['bottom'].set_visible(False)
ax_dft.spines['left'].set_visible(False)

# DFT Relevance Propagation
ax_dft_prop = fig.add_subplot(gs[2, 1])
ax_dft_prop.text(0.5, 0.7, 'DFT-LRP', ha='center', fontweight='bold', fontsize=12)
ax_dft_prop.text(0.5, 0.4, r'$R_{freq} = R_{time} \odot \frac{DFT(signal)}{|signal|}$', ha='center', fontsize=10)
ax_dft_prop.set_xlim(0, 1)
ax_dft_prop.set_ylim(0, 1)
ax_dft_prop.set_xticks([])
ax_dft_prop.set_yticks([])
ax_dft_prop.spines['top'].set_visible(False)
ax_dft_prop.spines['right'].set_visible(False)
ax_dft_prop.spines['bottom'].set_visible(False)
ax_dft_prop.spines['left'].set_visible(False)

# Frequency domain signal visualization
ax_freq = fig.add_subplot(gs[4, 1])
ax_freq.plot(freqs_x, np.abs(fft_x), 'r-', label='X-axis', alpha=0.8)
ax_freq.plot(freqs_x, np.abs(fft_y), 'g-', label='Y-axis', alpha=0.8)
ax_freq.plot(freqs_x, np.abs(fft_z), 'b-', label='Z-axis', alpha=0.8)
ax_freq.set_title('Frequency Domain Signal')
ax_freq.set_ylabel('Magnitude')
ax_freq.set_xlabel('Frequency (Hz)')
ax_freq.legend(loc='upper right', frameon=False)
ax_freq.set_xlim(0, 20)  # Focus on the relevant frequency range
ax_freq.grid(alpha=0.3)

# Frequency domain attribution visualization
ax_freq_attr = fig.add_subplot(gs[5, 1])
ax_freq_attr.plot(freqs_x, freq_relevance_x, 'r-', label='X-axis', alpha=0.8)
ax_freq_attr.plot(freqs_x, freq_relevance_y, 'g-', label='Y-axis', alpha=0.8)
ax_freq_attr.plot(freqs_x, freq_relevance_z, 'b-', label='Z-axis', alpha=0.8)
ax_freq_attr.set_title('Frequency-Domain Attribution')
ax_freq_attr.set_ylabel('Relevance')
ax_freq_attr.set_xlabel('Frequency (Hz)')
ax_freq_attr.set_xlim(0, 20)  # Focus on the relevant frequency range
ax_freq_attr.grid(alpha=0.3)
ax_freq_attr.set_ylim(0, 1.1)

# ----- TIME-FREQUENCY DOMAIN SECTION -----
# Title for time-frequency domain column
ax_title_tf = fig.add_subplot(gs[0, 2])
ax_title_tf.text(0.5, 0.5, 'Time-Frequency Domain', ha='center', va='center', fontweight='bold')
ax_title_tf.axis('off')

# STFT Virtual Layer visualization
ax_stft = fig.add_subplot(gs[1, 2])
ax_stft.text(0.5, 0.7, 'STFT Virtual Layer', ha='center', fontweight='bold', fontsize=12)
ax_stft.text(0.5, 0.4, r'STFT$(x)[m,k] = \sum_{n} x[n]w[n-m]e^{-i2\pi kn/N}$', ha='center', fontsize=10)
ax_stft.text(0.5, 0.2, 'Window Parameters: size=128, shift=64', ha='center', fontsize=9)
ax_stft.set_xlim(0, 1)
ax_stft.set_ylim(0, 1)
ax_stft.set_xticks([])
ax_stft.set_yticks([])
ax_stft.spines['top'].set_visible(False)
ax_stft.spines['right'].set_visible(False)
ax_stft.spines['bottom'].set_visible(False)
ax_stft.spines['left'].set_visible(False)

# STFT Relevance Propagation
ax_stft_prop = fig.add_subplot(gs[2, 2])
ax_stft_prop.text(0.5, 0.7, 'STFT-LRP', ha='center', fontweight='bold', fontsize=12)
ax_stft_prop.text(0.5, 0.4, r'Joint Time-Frequency Relevance', ha='center', fontsize=10)
ax_stft_prop.text(0.5, 0.2, r'When & Which Frequencies', ha='center', fontsize=9)
ax_stft_prop.set_xlim(0, 1)
ax_stft_prop.set_ylim(0, 1)
ax_stft_prop.set_xticks([])
ax_stft_prop.set_yticks([])
ax_stft_prop.spines['top'].set_visible(False)
ax_stft_prop.spines['right'].set_visible(False)
ax_stft_prop.spines['bottom'].set_visible(False)
ax_stft_prop.spines['left'].set_visible(False)

# Time-frequency signal visualization (X-axis)
ax_tf_x = fig.add_subplot(gs[4, 2])
im_x = ax_tf_x.pcolormesh(t_tf, f_tf, 10 * np.log10(Sxx_x + 1e-10), shading='gouraud', cmap='viridis')
ax_tf_x.set_title('Time-Frequency Representation (X-axis)')
ax_tf_x.set_ylabel('Frequency (Hz)')
ax_tf_x.set_xlabel('Time (s)')
ax_tf_x.set_ylim(0, 20)  # Focus on the relevant frequency range
plt.colorbar(im_x, ax=ax_tf_x, label='Power (dB)')

# Time-frequency attribution visualization
ax_tf_attr = fig.add_subplot(gs[5, 2])
im_attr = ax_tf_attr.pcolormesh(t_tf, f_tf, tf_relevance_x, shading='gouraud', cmap=cmap, vmin=0, vmax=1)
ax_tf_attr.set_title('Time-Frequency Domain Attribution (X-axis)')
ax_tf_attr.set_ylabel('Frequency (Hz)')
ax_tf_attr.set_xlabel('Time (s)')
ax_tf_attr.set_ylim(0, 20)  # Focus on the relevant frequency range
plt.colorbar(im_attr, ax=ax_tf_attr, label='Relevance')


# Add arrows connecting the sections
def add_arrow(fig, start_ax, end_ax, label="", offset_x=0, offset_y=0, connectionstyle="arc3,rad=0.2"):
    posA = start_ax.get_position()
    posB = end_ax.get_position()

    fig.add_artist(FancyArrowPatch(
        (posA.x1 + offset_x, posA.y0 + posA.height / 2 + offset_y),
        (posB.x0 - 0.02 + offset_x, posB.y0 + posB.height / 2 + offset_y),
        transform=fig.transFigure,
        connectionstyle=connectionstyle,
        arrowstyle='-|>',
        lw=1.5,
        color='black'
    ))

    if label:
        fig.text(
            (posA.x1 + posB.x0) / 2 + offset_x,
            posA.y0 + posA.height / 2 + offset_y + 0.02,
            label,
            ha='center',
            fontsize=9
        )


# Add arrows between sections
add_arrow(fig, ax_time_attr, ax_dft, label="DFT", offset_y=-0.1)
add_arrow(fig, ax_time_attr, ax_stft, label="STFT", offset_y=-0.15, connectionstyle="arc3,rad=0.3")

# Add a legend for the three domains
ax_legend = fig.add_subplot(gs[6, :])
ax_legend.axis('off')
ax_legend.text(0.17, 0.5, "Temporal patterns:\nWhen the fault occurs", ha='center', fontsize=10)
ax_legend.text(0.5, 0.5, "Spectral patterns:\nWhich frequencies indicate faults", ha='center', fontsize=10)
ax_legend.text(0.83, 0.5, "Joint patterns:\nWhen and which frequencies", ha='center', fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.5)
plt.savefig("xai_pipeline.png", dpi=300, bbox_inches='tight')
plt.savefig("xai_pipeline.pdf", bbox_inches='tight')
plt.show()