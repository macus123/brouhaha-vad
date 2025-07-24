import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random

def create_noise_library(target_duration=60, target_snr_levels=[-20, -10, 0, 10]):
    """
    Create a comprehensive noise library with different SNR levels 
    by combining the extracted noise profiles
    """
    noise_dir = Path("Noise_Profiles")
    output_dir = Path("Noise_Library")
    output_dir.mkdir(exist_ok=True)
    
    # Find all noise profiles
    noise_files = []
    for category_dir in noise_dir.glob("*"):
        if category_dir.is_dir():
            noise_files.extend(list(category_dir.glob("*.wav")))
    
    if not noise_files:
        print("No noise profiles found. Run extract_noise_profiles() first.")
        return
    
    print(f"Found {len(noise_files)} noise profiles")
    
    # Load all noise files
    noise_data = []
    for file_path in tqdm(noise_files, desc="Loading noise files"):
        try:
            audio, sr = librosa.load(file_path, sr=None)
            noise_data.append({
                "audio": audio,
                "sr": sr,
                "path": file_path,
                "category": file_path.parent.name
            })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Create combined noise library at different SNR levels
    for target_snr in target_snr_levels:
        print(f"\nCreating noise library at {target_snr} dB SNR")
        
        # Create a clean signal (silence) for reference
        reference_sr = 16000  # Standard sample rate
        reference_signal = np.zeros(int(target_duration * reference_sr))
        
        # Generate noise at target SNR
        noise_signal = generate_noise_at_snr(reference_signal, reference_sr, 
                                           noise_data, target_snr)
        
        # Save the result
        output_file = output_dir / f"standard_noise_{target_snr}dB.wav"
        sf.write(output_file, noise_signal, reference_sr)
        print(f"Created noise library at {target_snr} dB: {output_file}")
        
    # Create a mixed library with varying SNR
    print("\nCreating mixed noise library with varying SNR")
    mixed_noise = np.zeros(int(target_duration * reference_sr))
    section_duration = target_duration / len(target_snr_levels)
    
    for i, snr in enumerate(target_snr_levels):
        start_idx = int(i * section_duration * reference_sr)
        end_idx = int((i + 1) * section_duration * reference_sr)
        
        section_reference = np.zeros(end_idx - start_idx)
        section_noise = generate_noise_at_snr(section_reference, reference_sr, 
                                             noise_data, snr)
        
        mixed_noise[start_idx:end_idx] = section_noise
    
    # Save the mixed result
    output_file = output_dir / "mixed_snr_noise_library.wav"
    sf.write(output_file, mixed_noise, reference_sr)
    print(f"Created mixed SNR noise library: {output_file}")
    
    # Create a README file
    with open(output_dir / "README.txt", "w") as f:
        f.write("NOISE LIBRARY\n")
        f.write("============\n\n")
        f.write("This directory contains standardized noise profiles at different SNR levels.\n\n")
        
        for snr in target_snr_levels:
            f.write(f"standard_noise_{snr}dB.wav: Noise at {snr} dB SNR\n")
        
        f.write("\nmixed_snr_noise_library.wav: A single file with sections at different SNR levels\n")
        f.write(f"  Sections: {', '.join([f'{snr} dB' for snr in target_snr_levels])}\n")
        
    return output_dir

def generate_noise_at_snr(reference_signal, sr, noise_data, target_snr):
    """Generate noise at a specific SNR level relative to the reference signal."""
    # Combine different noise profiles randomly
    combined_noise = np.zeros_like(reference_signal)
    
    # Select noise samples to use (at least one of each category if available)
    categories = set(n["category"] for n in noise_data)
    selected_noise = []
    
    # First, try to get at least one from each category
    for category in categories:
        category_noise = [n for n in noise_data if n["category"] == category]
        if category_noise:
            selected_noise.append(random.choice(category_noise))
    
    # If we need more, add random ones
    while len(selected_noise) < min(5, len(noise_data)):
        sample = random.choice(noise_data)
        if sample not in selected_noise:
            selected_noise.append(sample)
    
    # Mix the selected noise samples
    for i, noise in enumerate(selected_noise):
        # Resample if needed
        if noise["sr"] != sr:
            noise_audio = librosa.resample(noise["audio"], orig_sr=noise["sr"], target_sr=sr)
        else:
            noise_audio = noise["audio"]
            
        # Make the noise the right length through looping or concatenation
        if len(noise_audio) < len(reference_signal):
            repeats_needed = int(np.ceil(len(reference_signal) / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats_needed)[:len(reference_signal)]
        else:
            # Take a random section
            start = random.randint(0, len(noise_audio) - len(reference_signal))
            noise_audio = noise_audio[start:start + len(reference_signal)]
        
        # Scale for mixing (giving less weight to later samples)
        weight = 0.5 ** i  # Exponential decay of importance
        combined_noise += noise_audio * weight
    
    # Normalize the combined noise
    if np.abs(combined_noise).max() > 0:
        combined_noise = combined_noise / np.abs(combined_noise).max()
    
    # Set the noise level based on target SNR
    ref_power = np.mean(reference_signal ** 2) + 1e-10  # Avoid division by zero
    
    # If reference is silence, use a standard reference level
    if ref_power < 1e-8:
        ref_power = 0.01  # Standard reference power
        
    noise_power = np.mean(combined_noise ** 2) + 1e-10
    
    # Calculate scaling factor to achieve desired SNR
    k = np.sqrt(ref_power / (10 ** (target_snr / 10) * noise_power))
    
    # Scale the noise
    scaled_noise = combined_noise * k
    
    return scaled_noise

if __name__ == "__main__":
    library_dir = create_noise_library()
    print(f"\nNoise library creation complete! Files saved to {library_dir}")