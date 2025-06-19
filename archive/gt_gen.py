from pathlib import Path
import os
import sys
from brouhaha_pipeline import process_audio
import librosa
import shutil

def generate_ground_truth(
    audio_path, 
    output_dir="gt_gen_data", 
    num_chunks=5, 
    verbose=True
):
    """
    Generate VAD ground truth for an audio file using Brouhaha.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Base directory for output (defaults to gt_gen_data)
        num_chunks: Number of chunks for processing (fewer for shorter files)
        verbose: Whether to show detailed output
        
    Returns:
        Dictionary with paths to the created files and information
    """
    # Create the output structure
    output_path = Path(output_dir)
    audio_output_path = output_path / "audio"
    ground_truth_path = output_path / "ground_truth"
    brouhaha_output_path = output_path / "brouhaha_output"
    
    # Get the file name
    file_path = Path(audio_path)
    file_name = file_path.stem
    
    # Create necessary directories
    audio_output_path.mkdir(parents=True, exist_ok=True)
    ground_truth_path.mkdir(parents=True, exist_ok=True)
    brouhaha_output_path.mkdir(parents=True, exist_ok=True)
    
    # Target audio path in our structure
    target_audio_path = audio_output_path / file_path.name
    
    if verbose:
        print(f"Processing {audio_path} with Brouhaha VAD...")
    
    # Process audio with Brouhaha, storing outputs in brouhaha_output folder
    results = process_audio(
        audio_paths=audio_path,
        out_dir=str(brouhaha_output_path),
        num_chunks=num_chunks,
        verbose=verbose
    )
    
    # Get audio duration
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Get speech segments from results
    speech_segments = results[audio_path]["speech_segments"]
    
    # Create ground truth file in the required format
    gt_file_path = ground_truth_path / f"{file_name}.txt"
    
    with open(gt_file_path, "w") as f:
        for i, (start, end) in enumerate(speech_segments):
            f.write(f"{start:.6f}\t{end:.6f}\tSpeech segment {i+1}\n")
    
    # Copy the audio file to our structure
    shutil.copy2(audio_path, target_audio_path)
    
    # Calculate speech and non-speech stats
    total_speech = sum(end - start for start, end in speech_segments)
    speech_percentage = (total_speech / duration) * 100
    
    if verbose:
        print(f"Created ground truth file: {gt_file_path}")
        print(f"Copied audio to: {target_audio_path}")
        print(f"Found {len(speech_segments)} speech segments")
        print(f"Speech: {total_speech:.2f}s ({speech_percentage:.1f}% of audio)")
        print(f"Non-speech: {duration - total_speech:.2f}s ({100-speech_percentage:.1f}% of audio)")
    
    return {
        "audio_path": str(target_audio_path),
        "ground_truth_path": str(gt_file_path),
        "speech_segments": len(speech_segments),
        "speech_duration": total_speech,
        "non_speech_duration": duration - total_speech,
        "total_duration": duration,
        "speech_percentage": speech_percentage
    }

def batch_generate_ground_truth(audio_dir, output_dir="gt_gen_data", verbose=True):
    """
    Process all audio files in a directory and create ground truth files.
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Base directory for output files
        verbose: Whether to show detailed output
    
    Returns:
        List of processed file results
    """
    audio_dir_path = Path(audio_dir)
    processed_files = []
    
    # Find all audio files
    audio_files = []
    for ext in ['wav', 'mp3', 'flac', 'ogg']:
        audio_files.extend(list(audio_dir_path.glob(f"*.{ext}")))
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return []
    
    if verbose:
        print(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    for audio_file in audio_files:
        result = generate_ground_truth(
            audio_path=str(audio_file),
            output_dir=output_dir,
            verbose=verbose
        )
        processed_files.append(result)
    
    return processed_files

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     input_path = sys.argv[1]
    #     output_dir = sys.argv[2] if len(sys.argv) > 2 else "gt_gen_data"
        
    #     if os.path.isdir(input_path):
    #         results = batch_generate_ground_truth(input_path, output_dir)
    #         print(f"\nProcessed {len(results)} files")
    #     else:
    #         result = generate_ground_truth(input_path, output_dir)
    #         print("\nProcessing complete!")
    # else:
    #     print("Usage: python gt_gen.py <audio_file_or_directory> [output_directory]")
    #     print("Example: python gt_gen.py my_audio.wav gt_gen_data")


    # Process a longer audio file with custom parameters
    result = generate_ground_truth(
        audio_path="my_gt_data/Recompiled_Output/recording_balanced_0.1h.wav",
        output_dir="my_gt_data_test",            # Custom output directory
        num_chunks=10,                      # More chunks for longer files
        verbose=True                        # Show detailed progress
    )

    # Detailed result access
    print(f"Audio file: {result['audio_path']}")
    print(f"Ground truth file: {result['ground_truth_path']}")
    print(f"Found {result['speech_segments']} speech segments")
    print(f"Speech duration: {result['speech_duration']:.2f}s")
    print(f"Non-speech duration: {result['non_speech_duration']:.2f}s")
    print(f"Total duration: {result['total_duration']:.2f}s")
    print(f"Speech percentage: {result['speech_percentage']:.1f}%")