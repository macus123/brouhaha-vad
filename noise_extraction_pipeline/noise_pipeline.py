#!/usr/bin/env python3

import sys
from filter_non_speech_by_snr import filter_non_speech_by_snr
from extract_noise import extract_noise_profiles
from create_noise_library import create_noise_library
from noise_merge import merge_speech_with_noise

def run_pipeline(stages="all", target_snr=5, speech_dir="VAD_Output/TEST/speech"):
    """Run noise processing pipeline in sequence"""
    
    if stages == "all":
        stages = ["filter", "extract", "library", "merge"]
    else:
        stages = [s.strip() for s in stages.split(",")]
    
    print("\n=== NOISE PROCESSING PIPELINE ===\n")
    
    if "filter" in stages:
        print("Step 1: Filtering non-speech by SNR...")
        filter_non_speech_by_snr()
    
    if "extract" in stages:
        print("\nStep 2: Extracting noise profiles...")
        extract_noise_profiles()
    
    if "library" in stages:
        print("\nStep 3: Creating noise library...")
        create_noise_library()
    
    if "merge" in stages:
        print("\nStep 4: Merging speech with noise...")
        merge_speech_with_noise(
            speech_dir=speech_dir, 
            noise_library_dir="Noise_Library",
            output_dir="Merged_Audio", 
            target_snr=target_snr
        )
    
    print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    # Simple command-line parsing
    stages = "all"
    target_snr = 5
    speech_dir = "VAD_Output/TEST/speech"
    
    # Very basic argument handling
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--stages" and i+1 < len(args):
            stages = args[i+1]
        elif arg == "--target-snr" and i+1 < len(args):
            target_snr = float(args[i+1])
        elif arg == "--speech-dir" and i+1 < len(args):
            speech_dir = args[i+1]
    
    run_pipeline(stages, target_snr, speech_dir)