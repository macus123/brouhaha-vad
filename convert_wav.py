from pathlib import Path
from pydub import AudioSegment
import os
import numpy as np
from typing import Dict, Any, List, Tuple
import datetime
from split_seg import read_ground_truth, get_non_speech_segments, save_audio_safely

def recompile_balanced_audio(
    input_wav: str,
    ground_truth: str = None,
    target_hours: float = 1.0,
    speech_padding_ms: int = 200,
    output_dir: str = "Recompiled_Output"
) -> Dict[str, Any]:
    """
    Recompile audio to achieve a balanced 1:1 speech/non-speech ratio at target duration.
    
    Args:
        input_wav: Path to input .wav file
        ground_truth: Path to ground truth file (if None, looks in standard location)
        target_hours: Target duration in hours for the balanced output file
        speech_padding_ms: Padding to add around speech segments in milliseconds
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with paths to output files and statistics
    """
    # Convert target hours to milliseconds
    target_ms = int(target_hours * 3600 * 1000)
    target_per_type_ms = target_ms // 2  # Equal parts speech and non-speech
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Resolve ground truth path if not provided
    if ground_truth is None:
        input_path = Path(input_wav)
        input_stem = input_path.stem
        potential_gt = Path("VAD_Input/Ground") / f"{input_stem}.txt"
        
        if potential_gt.exists():
            ground_truth = str(potential_gt)
        else:
            raise FileNotFoundError(f"Ground truth file not found for {input_wav}")
    
    print(f"Processing file: {input_wav}")
    print(f"Target: {target_hours:.2f} hours ({target_ms/1000:.1f} seconds) with 1:1 speech/non-speech ratio")
    
    # Load audio
    audio = AudioSegment.from_file(input_wav)
    total_duration_ms = len(audio)
    
    # Parse ground truth
    speech_segments = read_ground_truth(ground_truth)
    non_speech_segments = get_non_speech_segments(speech_segments, total_duration_ms/1000.0)
    
    # apply padding to speech segments, padding each segment by speech_padding_ms - extending the time boundaries of each speech segment before and after each speech segment
    # padding is taken from neighboring non-speech segments
    padded_speech_segments = []
    for start, end in speech_segments:
        # max(0, ...) ensures padding doesn't go before start of audio
        # min(total_duration_ms, ...) ensures padding doesn't go beyond end of audio
        padded_start = max(0, start * 1000 - speech_padding_ms) / 1000
        padded_end = min(total_duration_ms, end * 1000 + speech_padding_ms) / 1000
        padded_speech_segments.append((padded_start, padded_end))
    
    # Merge overlapping padded segments
    merged_speech_segments = []
    if padded_speech_segments:
        padded_speech_segments.sort(key=lambda x: x[0])
        current_start, current_end = padded_speech_segments[0]
        
        # this loop prevents chopping up closely spaced speech segments that now overlap due to padding
        for start, end in padded_speech_segments[1:]:
            if start <= current_end:
                # Merge with current segment
                current_end = max(current_end, end)
            else:
                # Save current segment and start a new one
                merged_speech_segments.append((current_start, current_end))
                current_start, current_end = start, end
                
        # Add the last segment
        merged_speech_segments.append((current_start, current_end))
    
    # Recalculate non-speech segments based on merged padded speech segments
    merged_non_speech_segments = get_non_speech_segments(merged_speech_segments, total_duration_ms/1000.0)
    
    # Calculate total speech and non-speech durations
    total_speech_ms = sum((end - start) * 1000 for start, end in merged_speech_segments)
    total_non_speech_ms = sum((end - start) * 1000 for start, end in merged_non_speech_segments)
    
    print(f"Original audio: {format_duration(total_duration_ms)}")
    print(f"Speech content: {format_duration(total_speech_ms)} ({total_speech_ms/total_duration_ms*100:.1f}%)")
    print(f"Non-speech content: {format_duration(total_non_speech_ms)} ({total_non_speech_ms/total_duration_ms*100:.1f}%)")
    
    # Check if we have enough audio to reach target
    if total_speech_ms < target_per_type_ms or total_non_speech_ms < target_per_type_ms:
        print(f"Warning: Insufficient audio to reach target duration with 1:1 ratio")
        print(f"Required for each type: {format_duration(target_per_type_ms)}")
        
        # Adjust target to what we can achieve
        target_per_type_ms = min(total_speech_ms, total_non_speech_ms)
        target_ms = target_per_type_ms * 2
        
        print(f"Adjusted target: {format_duration(target_ms)} ({target_ms/1000/3600:.2f} hours)")
    
    """
    segment selection logic
    """
    # Create timeline of all segments
    timeline = []
    
    # Add speech segments to timeline
    for start, end in merged_speech_segments:
        timeline.append({
            "start": start,
            "end": end,
            "type": "speech",
            "duration": (end - start) * 1000
        })
    
    # Add non-speech segments to timeline
    for start, end in merged_non_speech_segments:
        timeline.append({
            "start": start,
            "end": end,
            "type": "non-speech",
            "duration": (end - start) * 1000
        })
    
    # Sort by start time to maintain temporal order
    timeline.sort(key=lambda x: x["start"])
    
    # Process the timeline to collect segments for balanced output
    balanced_segments = []
    excess_speech_segments = []
    excess_non_speech_segments = []
    
    balanced_speech_ms = 0
    balanced_non_speech_ms = 0
    
    for segment in timeline:
        is_speech = segment["type"] == "speech"
        
        if is_speech:
            if balanced_speech_ms < target_per_type_ms:
                # Calculate how much of this segment we can use
                remaining_needed = target_per_type_ms - balanced_speech_ms
                
                if segment["duration"] <= remaining_needed:
                    # Use the whole segment
                    balanced_segments.append(segment)
                    balanced_speech_ms += segment["duration"]
                else:
                    # Split this segment
                    split_point = segment["start"] + (remaining_needed / 1000)
                    
                    # Add partial segment to balanced output
                    balanced_part = segment.copy()
                    balanced_part["end"] = split_point
                    balanced_part["duration"] = remaining_needed
                    balanced_segments.append(balanced_part)
                    balanced_speech_ms += remaining_needed
                    
                    # Add rest to excess speech
                    excess_part = segment.copy()
                    excess_part["start"] = split_point
                    excess_part["duration"] = segment["duration"] - remaining_needed
                    excess_speech_segments.append(excess_part)
            else:
                # Speech quota reached, add to excess
                excess_speech_segments.append(segment)
        else:
            # Non-speech segment
            if balanced_non_speech_ms < target_per_type_ms:
                # Calculate how much of this segment we can use
                remaining_needed = target_per_type_ms - balanced_non_speech_ms
                
                if segment["duration"] <= remaining_needed:
                    # Use the whole segment
                    balanced_segments.append(segment)
                    balanced_non_speech_ms += segment["duration"]
                else:
                    # Split this segment
                    split_point = segment["start"] + (remaining_needed / 1000)
                    
                    # Add partial segment to balanced output
                    balanced_part = segment.copy()
                    balanced_part["end"] = split_point
                    balanced_part["duration"] = remaining_needed
                    balanced_segments.append(balanced_part)
                    balanced_non_speech_ms += remaining_needed
                    
                    # Add rest to excess non-speech
                    excess_part = segment.copy()
                    excess_part["start"] = split_point
                    excess_part["duration"] = segment["duration"] - remaining_needed
                    excess_non_speech_segments.append(excess_part)
            else:
                # Non-speech quota reached, add to excess
                excess_non_speech_segments.append(segment)
    
    # Sort all segment collections by start time
    balanced_segments.sort(key=lambda x: x["start"])
    excess_speech_segments.sort(key=lambda x: x["start"])
    excess_non_speech_segments.sort(key=lambda x: x["start"])
    
    # Compile the audio files
    balanced_audio = AudioSegment.empty()
    excess_speech_audio = AudioSegment.empty()
    excess_non_speech_audio = AudioSegment.empty()
    
    for segment in balanced_segments:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        segment_audio = audio[start_ms:end_ms]
        balanced_audio += segment_audio
    
    for segment in excess_speech_segments:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        segment_audio = audio[start_ms:end_ms]
        excess_speech_audio += segment_audio
    
    for segment in excess_non_speech_segments:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        segment_audio = audio[start_ms:end_ms]
        excess_non_speech_audio += segment_audio
    
    # Save output files
    file_stem = Path(input_wav).stem
    balanced_output_path = output_path / f"{file_stem}_balanced_{target_hours:.1f}h.wav"
    excess_speech_path = output_path / f"{file_stem}_excess_speech.wav"
    excess_non_speech_path = output_path / f"{file_stem}_excess_non_speech.wav"
    
    # Use safe audio saving
    sample_rate = audio.frame_rate
    save_audio_safely(balanced_audio, balanced_output_path, sample_rate)
    save_audio_safely(excess_speech_audio, excess_speech_path, sample_rate)
    save_audio_safely(excess_non_speech_audio, excess_non_speech_path, sample_rate)
    
    # Generate a metadata file
    metadata_path = output_path / f"{file_stem}_recompile_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"RECOMPILATION METADATA\n")
        f.write(f"====================\n\n")
        f.write(f"Original file: {input_wav}\n")
        f.write(f"Ground truth: {ground_truth}\n")
        f.write(f"Target duration: {target_hours:.2f} hours\n")
        f.write(f"Speech padding: {speech_padding_ms} ms\n\n")
        
        f.write(f"Original duration: {format_duration(total_duration_ms)}\n")
        f.write(f"Original speech: {format_duration(total_speech_ms)} ({total_speech_ms/total_duration_ms*100:.1f}%)\n")
        f.write(f"Original non-speech: {format_duration(total_non_speech_ms)} ({total_non_speech_ms/total_duration_ms*100:.1f}%)\n\n")
        
        f.write(f"Balanced output file: {balanced_output_path.name}\n")
        f.write(f"  Duration: {format_duration(len(balanced_audio))}\n")
        f.write(f"  Speech: {format_duration(balanced_speech_ms)} ({balanced_speech_ms/(balanced_speech_ms+balanced_non_speech_ms)*100:.1f}%)\n")
        f.write(f"  Non-speech: {format_duration(balanced_non_speech_ms)} ({balanced_non_speech_ms/(balanced_speech_ms+balanced_non_speech_ms)*100:.1f}%)\n\n")
        
        f.write(f"Excess speech file: {excess_speech_path.name}\n")
        f.write(f"  Duration: {format_duration(len(excess_speech_audio))}\n\n")
        
        f.write(f"Excess non-speech file: {excess_non_speech_path.name}\n")
        f.write(f"  Duration: {format_duration(len(excess_non_speech_audio))}\n")
    
    # Return statistics
    return {
        "balanced_output": str(balanced_output_path),
        "excess_speech": str(excess_speech_path),
        "excess_non_speech": str(excess_non_speech_path),
        "metadata": str(metadata_path),
        "balanced_duration_hours": len(balanced_audio)/1000/3600,
        "balanced_speech_hours": balanced_speech_ms/1000/3600,
        "balanced_non_speech_hours": balanced_non_speech_ms/1000/3600,
        "excess_speech_hours": len(excess_speech_audio)/1000/3600,
        "excess_non_speech_hours": len(excess_non_speech_audio)/1000/3600,
    }

def format_duration(ms):
    """Format duration in milliseconds to a readable string."""
    seconds = ms / 1000
    return str(datetime.timedelta(seconds=seconds))

if __name__ == "__main__":
    result = recompile_balanced_audio(
        input_wav="recording.wav",
        ground_truth="my_gt_data/ground_truth/recording.txt",
        target_hours=0.1350,
        speech_padding_ms=300,
        output_dir="my_gt_data/Recompiled_Output"
    )

    # Access detailed results
    print(f"Balanced output: {result['balanced_output']}")
    print(f"Speech content: {result['balanced_speech_hours']:.2f} hours")
    print(f"Non-speech content: {result['balanced_non_speech_hours']:.2f} hours")
    print(f"Excess speech: {result['excess_speech_hours']:.2f} hours")
    print(f"Excess non-speech: {result['excess_non_speech_hours']:.2f} hours")
    print(f"Metadata saved to: {result['metadata']}")

