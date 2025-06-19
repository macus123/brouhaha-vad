import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# Define paths
results_dir = Path("results")
audio_file = "data/paris_walk.wav"
rttm_file = results_dir / "rttm_files" / "paris_walk.rttm"
snr_file = results_dir / "detailed_snr_labels" / "paris_walk.npy"
c50_file = results_dir / "c50" / "paris_walk.npy"

# Load the audio file
y, sr = librosa.load(audio_file, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
times = np.linspace(0, duration, len(y))

# Load speech segments from RTTM and create binary mask
speech_segments = []
with open(rttm_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        start = float(parts[3])
        dur = float(parts[4])
        speech_segments.append((start, start + dur))

# Create time points array for all metrics
time_resolution = 0.01  # 10ms
time_points = np.arange(0, duration, time_resolution)

# Create binary VAD signal (0/1)
vad_signal = np.zeros_like(time_points)
for start, end in speech_segments:
    mask_start = int(start / time_resolution)
    mask_end = min(int(end / time_resolution), len(vad_signal))
    if mask_start < len(vad_signal):
        vad_signal[mask_start:mask_end] = 1

# Load SNR and C50 data
snr_data = np.load(snr_file)
c50_data = np.load(c50_file)

# Create time arrays for SNR and C50
snr_time = np.linspace(0, duration, len(snr_data))
c50_time = np.linspace(0, duration, len(c50_data))

# Normalize audio waveform for visualization
y_norm = y / np.max(np.abs(y)) * 0.3  # Scaled to not overwhelm the plot

# Create the figure and axes
fig, ax = plt.subplots(figsize=(16, 8))

# Plot audio waveform in grey
ax.plot(times, y_norm, color='grey', linewidth=0.5, alpha=0.5, label='Audio Waveform')

# Plot VAD as solid green line
ax.plot(time_points, vad_signal, color='green', linewidth=2, label='VAD (Speech Detection)')

# Create a single twin axis for both SNR and C50
ax_db = ax.twinx()

# Plot SNR and C50 on the same dB axis
ax_db.plot(snr_time, snr_data, color='blue', linestyle=':', linewidth=2, label='SNR (dB)')
ax_db.plot(c50_time, c50_data, color='orange', linestyle=':', linewidth=2, label='C50 (dB)')

# Set axis labels
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Amplitude / VAD Status', fontsize=12)
ax_db.set_ylabel('Level (dB)', fontsize=12)

# Set the y-axis limits for VAD and audio
ax.set_ylim(-0.5, 1.5)

# Create a background shading for speech regions
for start, end in speech_segments:
    ax.axvspan(start, end, color='lightgreen', alpha=0.1)

# Combine all legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_db.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Add grid
ax.grid(True, alpha=0.3)

# Add title
plt.title('Audio Analysis Timeline', fontsize=14, fontweight='bold')

# Add key statistics in a text box
speech_percentage = 100 * np.mean(vad_signal)
plt.figtext(0.02, 0.02, 
            f"Audio duration: {duration:.1f}s | Speech: {speech_percentage:.1f}% | " +
            f"Avg SNR: {np.mean(snr_data):.1f}dB | Avg C50: {np.mean(c50_data):.1f}dB", 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

# Adjust layout and save
plt.tight_layout()
plt.savefig('unified_db_timeline.png', dpi=300, bbox_inches='tight')
plt.show()