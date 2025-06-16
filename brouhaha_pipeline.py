import os
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import librosa
import soundfile as sf
import tempfile
from tqdm import tqdm

# Make sure to import the necessary components
from pyannote.audio import Model
from brouhaha.pipeline import RegressiveActivityDetectionPipeline

def process_audio_in_chunks(
    audio_path: str,
    out_dir: Path,
    model: Model,
    pipeline: RegressiveActivityDetectionPipeline,
    num_chunks: int = 10,
    skip_snr: bool = False,
    skip_c50: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Process a single audio file in chunks and consolidate results.
    
    Args:
        audio_path: Path to audio file
        out_dir: Output directory
        model: Loaded brouhaha model
        pipeline: Configured pipeline
        num_chunks: Number of chunks to split the audio into
        skip_snr: Whether to skip SNR estimation
        skip_c50: Whether to skip C50 estimation
        verbose: Whether to display progress information
        
    Returns:
        Dictionary with consolidated results
    """
    # Load audio to determine duration
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    chunk_duration = duration / num_chunks
    
    file_uri = Path(audio_path).stem
    
    # Prepare folders
    rttm_folder = out_dir / "rttm_files"
    snr_folder = out_dir / "detailed_snr_labels"
    c50_folder = out_dir / "c50"
    
    # Prepare result containers
    all_speech_segments = []
    all_snr_values = []
    all_c50_values = []
    
    # Create a temporary directory for chunk files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Process each chunk
        chunk_progress = range(num_chunks)
        if verbose:
            chunk_progress = tqdm(chunk_progress, desc="Processing chunks")
            
        for i in chunk_progress:
            # Calculate chunk boundaries
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)
            
            # Extract chunk samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_audio = y[start_sample:end_sample]
            
            # Save chunk to temporary file
            chunk_path = temp_path / f"chunk_{i:03d}.wav"
            sf.write(chunk_path, chunk_audio, sr)
            
            # Process the chunk
            chunk_file = {"uri": f"{file_uri}_chunk_{i}", "audio": str(chunk_path)}
            inference = pipeline(chunk_file)
            
            # Adjust timestamps by adding the chunk start time
            chunk_annotation = inference["annotation"]
            adjusted_segments = []
            for segment in chunk_annotation.get_timeline():  # Timeline is directly iterable
                # Adjust start and end times
                adjusted_start = segment.start + start_time
                adjusted_end = segment.end + start_time
                adjusted_segments.append((adjusted_start, adjusted_end))
            
            # Store adjusted speech segments
            all_speech_segments.extend(adjusted_segments)
            
            # Process SNR and C50 data
            if not skip_snr:
                chunk_snr = inference["snr"]
                all_snr_values.append((start_time, chunk_snr))
                
            if not skip_c50:
                chunk_c50 = inference["c50"]
                all_c50_values.append((start_time, chunk_c50))
    
    # Consolidate speech segments
    # Sort by start time and merge overlapping segments
    all_speech_segments.sort(key=lambda x: x[0])
    merged_segments = []
    
    if all_speech_segments:
        current_start, current_end = all_speech_segments[0]
        
        for start, end in all_speech_segments[1:]:
            if start <= current_end + 0.1:  # Allow small gaps (100ms)
                # Extend the current segment
                current_end = max(current_end, end)
            else:
                # Save the current segment and start a new one
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
                
        # Don't forget the last segment
        merged_segments.append((current_start, current_end))
    
    # Create RTTM content
    rttm_content = ""
    for start, end in merged_segments:
        duration = end - start
        rttm_content += f"SPEAKER {file_uri} 1 {start:.3f} {duration:.3f} <NA> <NA> A <NA> <NA>\n"
    
    # Write RTTM file
    with open(rttm_folder / f"{file_uri}.rttm", "w") as rttm_file:
        rttm_file.write(rttm_content)
    
    # Consolidate SNR values
    if not skip_snr:
        # Create a timeline of SNR values based on chunk processing
        # Each chunk has SNR values at regular intervals
        all_snr_timeline = []
        snr_values = []
        
        for start_time, chunk_snr in all_snr_values:
            chunk_len = len(chunk_snr)
            # Calculate time points for this chunk
            chunk_duration = chunk_duration  # This is already calculated above
            time_step = chunk_duration / chunk_len
            
            for i, snr_val in enumerate(chunk_snr):
                time_point = start_time + (i * time_step)
                all_snr_timeline.append((time_point, snr_val))
        
        # Sort by time
        all_snr_timeline.sort(key=lambda x: x[0])
        
        # Extract values in order
        snr_values = [val for _, val in all_snr_timeline]
        
        # Save SNR data
        np.save(snr_folder / f"{file_uri}.npy", np.array(snr_values))
    
    # Consolidate C50 values - similar to SNR
    if not skip_c50:
        all_c50_timeline = []
        c50_values = []
        
        for start_time, chunk_c50 in all_c50_values:
            chunk_len = len(chunk_c50)
            time_step = chunk_duration / chunk_len
            
            for i, c50_val in enumerate(chunk_c50):
                time_point = start_time + (i * time_step)
                all_c50_timeline.append((time_point, c50_val))
        
        # Sort by time
        all_c50_timeline.sort(key=lambda x: x[0])
        
        # Extract values in order
        c50_values = [val for _, val in all_c50_timeline]
        
        # Save C50 data
        np.save(c50_folder / f"{file_uri}.npy", np.array(c50_values))
    
    # Calculate means for summary files
    mean_snr = np.mean(snr_values) if not skip_snr and snr_values else 0.0
    mean_c50 = np.mean(c50_values) if not skip_c50 and c50_values else 0.0
    
    # Append to summary files
    with open(out_dir / "reverb_labels.txt", "a") as label_file:
        label_file.write(f"{file_uri} {mean_c50}\n")
    
    with open(out_dir / "mean_snr_labels.txt", "a") as snr_file:
        snr_file.write(f"{file_uri} {mean_snr}\n")
    
    # Return results
    return {
        "speech_segments": merged_segments,
        "mean_snr": float(mean_snr),
        "c50": float(mean_c50)
    }

def process_audio(
    audio_paths: Union[str, List[str]],
    out_dir: str,
    model_path: str = "models/best/checkpoints/best.ckpt",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    step: float = 0.02,
    duration: float = 5.0,
    skip_snr: bool = False,
    skip_c50: bool = False,
    verbose: bool = True,
    num_chunks: int = 10  # Number of chunks to split each audio file into
) -> Dict[str, Dict[str, Any]]:
    """
    Process audio files with Brouhaha VAD model to detect speech segments and estimate SNR and C50.
    
    Args:
        audio_paths: Path to an audio file or directory, or a list of audio file paths
        out_dir: Directory where to save predictions
        model_path: Path to the model checkpoint
        batch_size: Batch size for inference
        device: Device to run inference on ('cuda' or 'cpu')
        step: Step size in seconds for sliding window
        duration: Duration in seconds of each window
        skip_snr: Whether to skip SNR estimation
        skip_c50: Whether to skip C50 estimation
        verbose: Whether to print progress information
        num_chunks: Number of chunks to split each audio file into
        
    Returns:
        Dictionary with results for each processed file
    """
    # Create output directory if it doesn't exist
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Create subdirectories for outputs
    rttm_folder = out_dir / "rttm_files"
    snr_folder = out_dir / "detailed_snr_labels"
    c50_folder = out_dir / "c50"
    
    for folder in [rttm_folder, snr_folder, c50_folder]:
        os.makedirs(folder, exist_ok=True)
    
    # Convert single path to list
    if isinstance(audio_paths, str):
        if os.path.isdir(audio_paths):
            # Get all audio files in directory
            audio_files = []
            for ext in ['wav', 'mp3', 'flac', 'ogg']:
                audio_files.extend(list(Path(audio_paths).glob(f"**/*.{ext}")))
            audio_paths = [str(p) for p in audio_files]
        else:
            # Single file
            audio_paths = [audio_paths]
    
    # Load the model
    if verbose:
        print(f"Loading model from {model_path}...")
    
    model = Model.from_pretrained(
        Path(model_path),
        strict=False,
    )
    
    # Create the pipeline
    pipeline = RegressiveActivityDetectionPipeline(
        segmentation=model,
        step=step,
        batch_size=batch_size,
    )
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Processing {len(audio_paths)} file(s) in {num_chunks} chunks each...")
    
    # Process each file
    results = {}
    for audio_path in audio_paths:
        if verbose:
            print(f"\nProcessing {audio_path} in {num_chunks} chunks...")
        
        file_results = process_audio_in_chunks(
            audio_path=audio_path,
            out_dir=out_dir,
            model=model,
            pipeline=pipeline,
            num_chunks=num_chunks,
            skip_snr=skip_snr,
            skip_c50=skip_c50,
            verbose=verbose
        )
        
        results[audio_path] = file_results
    
    return results