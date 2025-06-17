import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def merge_speech_with_noise(speech_dir, noise_library_dir, output_dir, 
                           padding_ratio=0.5, target_snr=5):
    """
    Merge speech files with noise by padding speech with noise.
    
    Args:
        speech_dir: Directory containing speech WAV files
        noise_library_dir: Directory containing noise library files
        output_dir: Directory to save merged files
        padding_ratio: Amount of padding relative to speech duration (0.5 = 50% padding)
        target_snr: Target SNR for the noise in dB
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find speech and noise files
    speech_files = list(Path(speech_dir).glob("*.wav"))
    noise_files = list(Path(noise_library_dir).glob("*.wav"))
    
    if not speech_files:
        print(f"No speech files found in {speech_dir}")
        return
        
    if not noise_files:
        print(f"No noise files found in {noise_library_dir}")
        return
    
    print(f"Found {len(speech_files)} speech files and {len(noise_files)} noise files")
    
    # Create subdirectory for visualization
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Process each speech file
    for speech_file in tqdm(speech_files, desc="Merging speech with noise"):
        try:
            # Load speech audio
            speech, sr = librosa.load(speech_file, sr=None)
            speech_duration = len(speech) / sr
            
            # Calculate padding duration
            pad_duration = speech_duration * padding_ratio
            pad_samples = int(pad_duration * sr)
            
            # Choose a random noise file
            noise_file = random.choice(noise_files)
            noise, noise_sr = librosa.load(noise_file, sr=None)
            
            # Resample noise if needed
            if noise_sr != sr:
                noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)
            
            # Make sure noise is long enough by looping if necessary
            if len(noise) < (pad_samples * 2 + len(speech)):
                repeats_needed = int(np.ceil((pad_samples * 2 + len(speech)) / len(noise)))
                noise = np.tile(noise, repeats_needed)
            
            # Get noise segments for before and after
            noise_before = noise[:pad_samples]
            noise_after = noise[pad_samples:pad_samples*2]
            
            # Get noise for mixing with speech
            noise_during = noise[pad_samples*2:pad_samples*2+len(speech)]
            
            # Calculate speech power
            speech_power = np.mean(speech ** 2) + 1e-10
            
            # Calculate and apply scaling for noise during speech
            noise_power = np.mean(noise_during ** 2) + 1e-10
            scaling = np.sqrt(speech_power / (10**(target_snr/10) * noise_power))
            mixed_speech = speech + scaling * noise_during
            
            # Create the final merged audio
            merged_audio = np.concatenate([noise_before, mixed_speech, noise_after])
            
            # Normalize to prevent clipping
            if np.max(np.abs(merged_audio)) > 0.98:
                merged_audio = merged_audio / np.max(np.abs(merged_audio)) * 0.98
            
            # Save the result
            output_file = output_dir / f"{speech_file.stem}_with_padding.wav"
            sf.write(output_file, merged_audio, sr)
            
            # Create visualization for some files (e.g., first 5)
            if speech_files.index(speech_file) < 5:
                create_visualization(speech, noise_before, mixed_speech, noise_after, 
                                    sr, viz_dir / f"{speech_file.stem}_visualization.png")
            
        except Exception as e:
            print(f"Error processing {speech_file}: {e}")
    
    # Create summary of the merged files
    with open(output_dir / "merge_summary.txt", "w") as f:
        f.write("SPEECH-NOISE MERGE SUMMARY\n")
        f.write("========================\n\n")
        f.write(f"Total files processed: {len(speech_files)}\n")
        f.write(f"Padding ratio: {padding_ratio * 100}% of speech duration\n")
        f.write(f"Target SNR: {target_snr} dB\n\n")
        f.write("The merged files contain:\n")
        f.write(f"1. Noise padding before speech ({padding_ratio * 100}% of speech duration)\n")
        f.write(f"2. Speech mixed with noise at {target_snr} dB SNR\n")
        f.write(f"3. Noise padding after speech ({padding_ratio * 100}% of speech duration)\n")
    
    print(f"\nMerged {len(speech_files)} files with noise padding")
    print(f"Results saved to {output_dir}")
    print(f"Visualizations saved to {viz_dir}")

def create_visualization(speech, noise_before, mixed_speech, noise_after, sr, output_file):
    """Create a visualization of the merged audio components."""
    plt.figure(figsize=(12, 8))
    
    # Plot waveforms
    plt.subplot(4, 1, 1)
    plt.plot(noise_before)
    plt.title("Noise Before (Padding)")
    plt.ylabel("Amplitude")
    
    plt.subplot(4, 1, 2)
    plt.plot(speech)
    plt.title("Original Speech")
    plt.ylabel("Amplitude")
    
    plt.subplot(4, 1, 3)
    plt.plot(mixed_speech)
    plt.title("Speech + Noise")
    plt.ylabel("Amplitude")
    
    plt.subplot(4, 1, 4)
    plt.plot(noise_after)
    plt.title("Noise After (Padding)")
    plt.ylabel("Amplitude")
    plt.xlabel("Samples")
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    # Example usage
    merge_speech_with_noise(
        speech_dir="VAD_Output/TEST/speech",
        noise_library_dir="Noise_Library",
        output_dir="Merged_Audio",
        padding_ratio=0.5,  # 50% padding before and after
        target_snr=5       # 5dB SNR
    )