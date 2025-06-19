import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
from tqdm import tqdm
import shutil
from brouhaha_pipeline import process_audio

def filter_non_speech_by_snr():
    """
    Process all non-speech files with Brouhaha model and filter out 
    the top and bottom 20% based on SNR predictions
    """
    # Set up paths
    output_base_dir = Path("VAD_Output")
    splits = ["DEV", "TEST", "TRAIN"]
    results_dir = Path("SNR_Results")
    results_dir.mkdir(exist_ok=True)
    
    # Create output directories for results
    keep_dir = results_dir / "keep_middle_60_percent"
    top_dir = results_dir / "top_20_percent"
    bottom_dir = results_dir / "bottom_20_percent"
    brouhaha_results_dir = results_dir / "brouhaha_output"
    
    for directory in [keep_dir, top_dir, bottom_dir, brouhaha_results_dir]:
        directory.mkdir(exist_ok=True)
    
    # Find all non-speech files
    print("Searching for non-speech files...")
    all_files = []
    for split in splits:
        non_speech_dir = output_base_dir / split / "non_speech"
        if non_speech_dir.exists():
            print(f"Found non-speech directory for {split}")
            for file_path in non_speech_dir.glob("*.wav"):
                all_files.append(str(file_path))
    
    print(f"Found {len(all_files)} non-speech files to process")
    
    if len(all_files) == 0:
        print("No non-speech files found. Please make sure the VAD_Output directory structure is correct.")
        return
    
    # Process files with Brouhaha to get SNR values
    print("\nProcessing files with Brouhaha model...")
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100  # Adjust based on your system's capacity
    results = {}
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(all_files)-1)//batch_size + 1} ({len(batch_files)} files)")
        
        batch_results = process_audio(
            audio_paths=batch_files,
            out_dir=str(brouhaha_results_dir),
            skip_c50=True,  # We only need SNR values
            verbose=True,
            num_chunks=1    # Use smaller number of chunks for shorter files
        )
        
        results.update(batch_results)
    
    # Extract SNR values and compile results
    print("\nExtracting SNR values...")
    snr_results = []
    
    for file_path, file_result in results.items():
        snr_value = file_result.get('mean_snr', 0.0)
        
        # Get split and filename info
        path = Path(file_path)
        filename = path.name
        parent_parts = path.parts
        split = "UNKNOWN"
        
        # Find which split this file belongs to
        for split_name in splits:
            if split_name in parent_parts:
                split = split_name
                break
        
        snr_results.append({
            "path": file_path,
            "split": split,
            "filename": filename,
            "snr": snr_value
        })
    
    # Create DataFrame and sort by SNR
    df = pd.DataFrame(snr_results)
    df = df.sort_values(by="snr")
    
    # Save complete results
    df.to_csv(results_dir / "all_snr_results.csv", index=False)
    
    # Calculate thresholds for filtering
    num_files = len(df)
    bottom_threshold_idx = int(num_files * 0.2)
    top_threshold_idx = int(num_files * 0.8)
    
    bottom_threshold = df.iloc[bottom_threshold_idx]["snr"]
    top_threshold = df.iloc[top_threshold_idx]["snr"]
    
    # Split the files into three categories
    bottom_files = df[df["snr"] <= bottom_threshold]
    top_files = df[df["snr"] >= top_threshold]
    keep_files = df[(df["snr"] > bottom_threshold) & (df["snr"] < top_threshold)]
    
    # Create lists of files in each category
    with open(results_dir / "keep_files.txt", "w") as f:
        for _, row in keep_files.iterrows():
            f.write(f"{row['path']},{row['snr']}\n")
            
    with open(results_dir / "top_20_percent.txt", "w") as f:
        for _, row in top_files.iterrows():
            f.write(f"{row['path']},{row['snr']}\n")
            
    with open(results_dir / "bottom_20_percent.txt", "w") as f:
        for _, row in bottom_files.iterrows():
            f.write(f"{row['path']},{row['snr']}\n")
    
    # Copy files to their respective directories
    print("\nCopying files to filtered directories...")
    
    for _, row in tqdm(keep_files.iterrows(), desc="Copying keep files (middle 60%)"):
        shutil.copy2(row["path"], keep_dir / row["filename"])
    
    for _, row in tqdm(top_files.iterrows(), desc="Copying top 20% files"):
        shutil.copy2(row["path"], top_dir / row["filename"])
    
    for _, row in tqdm(bottom_files.iterrows(), desc="Copying bottom 20% files"):
        shutil.copy2(row["path"], bottom_dir / row["filename"])
    
    # Generate summary statistics
    print("\n=== SNR Filtering Summary ===")
    print(f"Total non-speech files: {num_files}")
    print(f"Bottom 20% threshold: SNR ≤ {bottom_threshold:.2f} dB ({len(bottom_files)} files)")
    print(f"Top 20% threshold: SNR ≥ {top_threshold:.2f} dB ({len(top_files)} files)")
    print(f"Middle 60% (keeping): {bottom_threshold:.2f} < SNR < {top_threshold:.2f} dB ({len(keep_files)} files)")
    
    # Create visualization
    print("\nGenerating SNR distribution plot...")
    plt.figure(figsize=(10, 6))
    plt.hist(df["snr"], bins=50, alpha=0.7)
    plt.axvline(bottom_threshold, color='r', linestyle='--', 
                label=f'Bottom 20% threshold: {bottom_threshold:.2f} dB')
    plt.axvline(top_threshold, color='g', linestyle='--', 
                label=f'Top 20% threshold: {top_threshold:.2f} dB')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Number of files")
    plt.title("Distribution of SNR in Non-Speech Files")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "snr_distribution.png")
    
    print(f"\nResults saved to {results_dir}")
    print(f"SNR distribution plot saved to {results_dir / 'snr_distribution.png'}")

if __name__ == "__main__":
    filter_non_speech_by_snr()