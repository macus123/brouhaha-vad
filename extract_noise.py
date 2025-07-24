import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def extract_noise_profiles():
    """Extract noise profiles from the bottom 20% SNR non-speech segments."""
    
    # Set paths
    snr_results_dir = Path("SNR_Results")
    bottom_20_dir = snr_results_dir / "bottom_20_percent"
    noise_output_dir = Path("Noise_Profiles")
    noise_output_dir.mkdir(exist_ok=True)
    
    # Ensure we have the bottom 20% files
    if not bottom_20_dir.exists():
        print(f"Error: Directory {bottom_20_dir} not found. Run SNR filtering first.")
        return
    
    # Directories for different types of noise
    noise_categories = {
        "low_freq_noise": noise_output_dir / "low_frequency",  # 20Hz-200Hz
        "mid_freq_noise": noise_output_dir / "mid_frequency",  # 200Hz-2kHz
        "high_freq_noise": noise_output_dir / "high_frequency", # 2kHz-20kHz
        "broadband_noise": noise_output_dir / "broadband",    # Across spectrum
    }
    
    # Create category directories
    for category_dir in noise_categories.values():
        category_dir.mkdir(exist_ok=True)
    
    # Load SNR results to get the noisiest files first
    try:
        snr_df = pd.read_csv(snr_results_dir / "all_snr_results.csv")
        bottom_files = snr_df[snr_df["snr"] <= snr_df.iloc[int(len(snr_df) * 0.2)]["snr"]]
        files_to_process = [Path(row["path"]) for _, row in bottom_files.iterrows() 
                            if Path(row["path"]).exists()]
    except Exception as e:
        print(f"Error loading SNR results: {e}")
        # Fallback: just use all files in the bottom_20_percent directory
        files_to_process = list(bottom_20_dir.glob("*.wav"))
    
    print(f"Processing {len(files_to_process)} files for noise extraction")
    
    # Analysis stats
    noise_profiles = {category: [] for category in noise_categories}
    
    # Process each file
    for file_path in tqdm(files_to_process, desc="Extracting noise profiles"):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=None)
            
            # Compute spectrum to analyze noise characteristics
            S = np.abs(librosa.stft(audio))
            freq_bands = librosa.fft_frequencies(sr=sr)
            
            # Calculate energy in different frequency bands
            low_freq_energy = np.sum(S[freq_bands < 200, :])
            mid_freq_energy = np.sum(S[(freq_bands >= 200) & (freq_bands < 2000), :])
            high_freq_energy = np.sum(S[freq_bands >= 2000, :])
            total_energy = np.sum(S)
            
            # Determine dominant noise type
            energies = [low_freq_energy, mid_freq_energy, high_freq_energy]
            max_energy_index = np.argmax(energies)
            broadband_threshold = 0.5  # If no frequency band has >50% energy, consider it broadband
            
            if energies[max_energy_index] / total_energy > broadband_threshold:
                if max_energy_index == 0:
                    category = "low_freq_noise"
                elif max_energy_index == 1:
                    category = "mid_freq_noise"
                else:
                    category = "high_freq_noise"
            else:
                category = "broadband_noise"
            
            # Store this noise profile
            noise_profiles[category].append({
                "file": file_path,
                "audio": audio,
                "sr": sr,
                "duration": len(audio)/sr,
                "energy_distribution": [
                    low_freq_energy/total_energy, 
                    mid_freq_energy/total_energy,
                    high_freq_energy/total_energy
                ]
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Generate standardized noise profiles
    for category, profiles in noise_profiles.items():
        if not profiles:
            print(f"No profiles found for category: {category}")
            continue
            
        print(f"\nProcessing {len(profiles)} files for {category}")
        output_dir = noise_categories[category]
        
        # Create standard length noise profiles (1s, 2s, 5s)
        for duration in [1, 2, 5]:
            # Create concatenated noise for this duration and category
            create_standard_noise_profile(profiles, output_dir, duration)
            
    # Create a noise summary
    create_noise_summary(noise_profiles, noise_output_dir)
    
    return noise_output_dir

def create_standard_noise_profile(profiles, output_dir, target_duration):
    """Create standardized noise profile of specified duration."""
    if not profiles:
        return
        
    # Use sample rate from first profile
    sr = profiles[0]["sr"]
    
    # Sort profiles by duration for easier processing
    profiles = sorted(profiles, key=lambda x: x["duration"])
    
    # Method 1: Concatenate short samples to reach target duration
    concat_audio = np.array([])
    remaining_duration = target_duration
    
    # First try to use whole profiles
    for profile in profiles:
        if profile["duration"] <= remaining_duration:
            concat_audio = np.concatenate([concat_audio, profile["audio"]])
            remaining_duration -= profile["duration"]
        
        if remaining_duration <= 0:
            # Trim to exact duration
            concat_audio = concat_audio[:int(target_duration * sr)]
            break
    
    # If we didn't get enough, loop the last one
    if remaining_duration > 0 and len(concat_audio) > 0:
        repeat_audio = np.tile(concat_audio, int(np.ceil(target_duration * sr / len(concat_audio))))
        concat_audio = repeat_audio[:int(target_duration * sr)]
    
    # If we still don't have audio, use the longest available
    if len(concat_audio) == 0 and profiles:
        longest_profile = max(profiles, key=lambda x: len(x["audio"]))
        # Loop if needed
        repeat_count = int(np.ceil(target_duration * sr / len(longest_profile["audio"])))
        concat_audio = np.tile(longest_profile["audio"], repeat_count)[:int(target_duration * sr)]
    
    # Save the concatenated noise profile
    if len(concat_audio) > 0:
        output_file = output_dir / f"standard_{target_duration}s_concat.wav"
        sf.write(output_file, concat_audio, sr)
        print(f"Created standardized {target_duration}s noise profile: {output_file}")
    
    # Method 2: Create synthetic noise with similar spectral characteristics
    if len(profiles) >= 3:  # Need multiple samples for good estimation
        # Average the spectrum of all samples for more stable characteristics
        avg_spectrum = np.zeros(1024)  # Use fixed size for consistency
        count = 0
        
        for profile in profiles:
            spectrum = np.abs(librosa.stft(profile["audio"]))
            if spectrum.shape[0] >= 1024:
                avg_spectrum += np.mean(spectrum[:1024, :], axis=1)
                count += 1
        
        if count > 0:
            avg_spectrum /= count
            
            # Generate synthetic noise with this spectral shape
            synthetic_noise = generate_shaped_noise(avg_spectrum, target_duration, sr)
            output_file = output_dir / f"standard_{target_duration}s_synthetic.wav"
            sf.write(output_file, synthetic_noise, sr)
            print(f"Created synthetic {target_duration}s noise profile: {output_file}")

def generate_shaped_noise(spectrum_shape, duration, sr):
    """Generate noise with a specific spectral shape."""
    # Generate white noise
    white_noise = np.random.normal(0, 1, int(duration * sr))
    
    # Shape the noise in frequency domain
    noise_spectrum = librosa.stft(white_noise)
    
    # Apply the target spectral shape (keeping phase information)
    shaped_spectrum = noise_spectrum.copy()
    for i in range(min(noise_spectrum.shape[0], len(spectrum_shape))):
        shaped_spectrum[i, :] = shaped_spectrum[i, :] * (spectrum_shape[i] / (np.abs(shaped_spectrum[i, :]).mean() + 1e-8))
    
    # Convert back to time domain
    shaped_noise = librosa.istft(shaped_spectrum)
    
    # Normalize
    shaped_noise = shaped_noise / np.max(np.abs(shaped_noise))
    
    return shaped_noise

def create_noise_summary(noise_profiles, output_dir):
    """Create summary information and visualizations of noise profiles."""
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    categories = list(noise_profiles.keys())
    counts = [len(profiles) for profiles in noise_profiles.values()]
    
    plt.subplot(2, 1, 1)
    plt.bar(categories, counts)
    plt.title('Noise Profiles by Category')
    plt.ylabel('Number of Samples')
    
    # Plot average energy distribution for each category
    plt.subplot(2, 1, 2)
    x_labels = ['Low Freq', 'Mid Freq', 'High Freq']
    
    for i, (category, profiles) in enumerate(noise_profiles.items()):
        if not profiles:
            continue
            
        # Average energy distribution
        avg_distribution = np.mean([p["energy_distribution"] for p in profiles], axis=0)
        plt.bar([x + i*0.2 for x in range(3)], avg_distribution, width=0.2, 
                label=category, alpha=0.7)
    
    plt.xlabel('Frequency Range')
    plt.ylabel('Energy Proportion')
    plt.title('Energy Distribution by Noise Category')
    plt.xticks(range(3), x_labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "noise_summary.png")
    
    # Create text summary
    with open(output_dir / "noise_summary.txt", "w") as f:
        f.write("NOISE PROFILE SUMMARY\n")
        f.write("====================\n\n")
        
        for category, profiles in noise_profiles.items():
            f.write(f"{category}: {len(profiles)} samples\n")
            
            if profiles:
                total_duration = sum(p["duration"] for p in profiles)
                f.write(f"  Total duration: {total_duration:.2f} seconds\n")
                f.write(f"  Average duration: {total_duration/len(profiles):.2f} seconds\n")
                
                # Energy distribution
                avg_distribution = np.mean([p["energy_distribution"] for p in profiles], axis=0)
                f.write(f"  Average energy distribution:\n")
                f.write(f"    Low frequency (20-200Hz): {avg_distribution[0]*100:.1f}%\n")
                f.write(f"    Mid frequency (200Hz-2kHz): {avg_distribution[1]*100:.1f}%\n")
                f.write(f"    High frequency (2kHz+): {avg_distribution[2]*100:.1f}%\n")
            
            f.write("\n")

if __name__ == "__main__":
    noise_dir = extract_noise_profiles()
    print(f"\nNoise extraction complete! Profiles saved to {noise_dir}")