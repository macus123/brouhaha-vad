from pathlib import Path
from typing import Dict, Any, List, Tuple
from pydub import AudioSegment
import os
import numpy as np
import datetime
from split_seg import read_ground_truth, get_non_speech_segments, save_audio_safely

def format_duration(ms):
    """Format duration in milliseconds to a readable string."""
    seconds = ms / 1000
    return str(datetime.timedelta(seconds=seconds))

def format_time_mmss(seconds):
    """Format seconds as MM:SS.mmm"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"

def scan_input_directory(input_dir="input_data"):
    """
    Scan input_data directory structure to find audio files and match with ground truth.
    
    Args:
        input_dir: Path to input_data directory
        
    Returns:
        List of dictionaries with matched audio/ground_truth files
    """
    base_dir = Path(input_dir)
    audio_dir = base_dir / "audio"
    ground_truth_dir = base_dir / "ground_truth"
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found at {audio_dir}")
        return []
        
    if not ground_truth_dir.exists():
        print(f"Error: Ground truth directory not found at {ground_truth_dir}")
        return []
    
    # Get all audio files
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    
    result = []
    
    # Find matching ground truth for each audio file
    for audio_file in audio_files:
        # Extract base name without extension
        file_stem = audio_file.stem
        
        # Look for matching ground truth file
        gt_file = ground_truth_dir / f"{file_stem}.txt"
        if gt_file.exists():
            result.append({
                "audio_path": str(audio_file),
                "ground_truth_path": str(gt_file),
                "set_type": "TEST"  # Default to TEST for primary output
            })
        else:
            print(f"Warning: No ground truth found for {audio_file}")
    
    print(f"Matched {len(result)} files with ground truth")
    return result

def recompile_balanced_audio(
    input_wav: str,
    ground_truth: str = None,
    target_hours: float = 1.0,
    speech_padding_ms: int = 200,
    output_dir: str = "Recompiled_Output",
    create_splits: bool = True,
    dev_ratio: float = 0.2,
    silence_reserve_ratio: float = 0.4  # Reserve this portion of silence for interspersing
) -> Dict[str, Any]:
    """
    Recompile audio to achieve a balanced 1:1 speech/non-speech ratio at target duration.
    Uses improved strategy to distribute silence throughout the audio.
    
    Args:
        input_wav: Path to input audio file
        ground_truth: Path to ground truth file (default: auto-detect)
        target_hours: Target duration in hours for balanced output
        speech_padding_ms: Padding to add around speech segments (ms)
        output_dir: Directory to save output files
        create_splits: Whether to create DEV/TRAIN splits
        dev_ratio: Ratio of excess content to allocate to DEV set
        silence_reserve_ratio: Portion of silence to reserve for interspersing
        
    Returns:
        Dict with paths to output files and statistics
    """
    # Get file stem early to avoid reference before assignment
    file_stem = Path(input_wav).stem
    
    # Convert target hours to milliseconds
    target_ms = int(target_hours * 3600 * 1000)
    target_per_type_ms = target_ms // 2  # Equal parts speech and non-speech
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create timestamp directory
    timestamp_dir = output_path / "timestamp_maps"
    timestamp_dir.mkdir(exist_ok=True, parents=True)
    
    # Auto-detect ground truth if not provided
    if ground_truth is None:
        input_path = Path(input_wav)
        potential_gt = Path(input_path.parent.parent, "ground_truth", f"{input_path.stem}.txt")
        if potential_gt.exists():
            ground_truth = str(potential_gt)
        else:
            raise ValueError(f"Ground truth file not provided and could not be auto-detected for {input_wav}")
    
    # Load audio and get duration
    print(f"Processing file: {input_wav}")
    print(f"Target: {target_hours} hours ({target_ms/1000:.1f} seconds) with 1:1 speech/non-speech ratio")
    
    global audio  # Make accessible to helper functions
    audio = AudioSegment.from_file(input_wav)
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000
    
    print(f"Original audio: {format_duration(total_duration_ms)}")
    
    # Read ground truth and get speech segments
    speech_segments = read_ground_truth(ground_truth)
    non_speech_segments = get_non_speech_segments(speech_segments, total_duration_sec)
    
    # Add padding to speech segments and merge overlapping segments
    padded_speech_segments = []
    for start, end in speech_segments:
        padded_start = max(0, start - speech_padding_ms / 1000)
        padded_end = min(total_duration_sec, end + speech_padding_ms / 1000)
        padded_speech_segments.append((padded_start, padded_end))
    
    # Merge overlapping padded segments
    merged_speech_segments = []
    if padded_speech_segments:
        padded_speech_segments.sort()  # Sort by start time
        current_start, current_end = padded_speech_segments[0]
        
        for start, end in padded_speech_segments[1:]:
            if start <= current_end:  # Overlapping segments
                current_end = max(current_end, end)
            else:  # Non-overlapping
                merged_speech_segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_speech_segments.append((current_start, current_end))
    
    # Recalculate non-speech segments based on merged speech segments
    merged_non_speech_segments = get_non_speech_segments(merged_speech_segments, total_duration_sec)
    
    # Calculate total speech and non-speech durations
    total_speech_ms = sum((end - start) * 1000 for start, end in merged_speech_segments)
    total_non_speech_ms = sum((end - start) * 1000 for start, end in merged_non_speech_segments)
    
    # Print statistics
    print(f"Speech content: {format_duration(total_speech_ms)} ({total_speech_ms/total_duration_ms*100:.1f}%)")
    print(f"Non-speech content: {format_duration(total_non_speech_ms)} ({total_non_speech_ms/total_duration_ms*100:.1f}%)")
    
    # Check if we have enough audio to reach target
    if total_speech_ms < target_per_type_ms or total_non_speech_ms < target_per_type_ms:
        print(f"Warning: Insufficient audio to reach target duration with 1:1 ratio")
        print(f"Available speech: {format_duration(total_speech_ms)}")
        print(f"Available non-speech: {format_duration(total_non_speech_ms)}")
        target_per_type_ms = min(total_speech_ms, total_non_speech_ms)
        print(f"Adjusting target to: {format_duration(target_per_type_ms * 2)}")
    
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
    
    # IMPROVED APPROACH: Reserve some silence for interspersing
    # First pass: Categorize segments by type and duration
    speech_segments = []
    silence_segments = []
    
    for segment in timeline:
        if segment["type"] == "speech":
            speech_segments.append(segment)
        else:
            silence_segments.append(segment)
    
    # Sort speech segments chronologically
    speech_segments.sort(key=lambda x: x["start"])
    
    # Calculate silence quotas
    reserved_silence_ms = target_per_type_ms * silence_reserve_ratio
    primary_silence_ms = target_per_type_ms - reserved_silence_ms
    
    # Sort silence segments by duration (shortest first for interspersing)
    short_silence_segments = sorted(silence_segments, key=lambda x: x["duration"])
    
    # Select short silence segments to reserve for interspersing
    reserved_silence = []
    reserved_silence_duration = 0
    
    for segment in short_silence_segments:
        if reserved_silence_duration < reserved_silence_ms:
            # Check if adding this segment would exceed our reserve quota
            if reserved_silence_duration + segment["duration"] <= reserved_silence_ms * 1.1:  # Allow 10% overage
                reserved_silence.append(segment)
                reserved_silence_duration += segment["duration"]
            elif reserved_silence_duration < reserved_silence_ms * 0.9:  # If we're under 90% of target
                # Try to split the segment to get exactly what we need
                remaining_needed = reserved_silence_ms - reserved_silence_duration
                
                # Only split if the segment is at least twice what we need
                if segment["duration"] > remaining_needed * 2:
                    # Calculate split point
                    split_point = segment["start"] + (remaining_needed / 1000)
                    
                    # Create first part for reserved silence
                    first_part = segment.copy()
                    first_part["end"] = split_point
                    first_part["duration"] = remaining_needed
                    reserved_silence.append(first_part)
                    reserved_silence_duration += remaining_needed
                    
                    # Second part stays in pool for primary selection
                else:
                    # Segment is too small to split effectively, just add it all
                    reserved_silence.append(segment)
                    reserved_silence_duration += segment["duration"]
    
    # Remove reserved segments from consideration for primary selection
    reserved_ids = set(id(seg) for seg in reserved_silence)
    primary_candidates = [seg for seg in silence_segments if id(seg) not in reserved_ids]
    
    # Sort remaining silence segments chronologically for temporal coherence
    primary_silence = sorted(primary_candidates, key=lambda x: x["start"])
    
    # Select primary speech and silence segments to build the base timeline
    balanced_segments = []
    speech_quota_remaining = target_per_type_ms
    silence_quota_remaining = primary_silence_ms
    
    # Try to start with some silence for a natural beginning
    if primary_silence and silence_quota_remaining > 0:
        first_segment = primary_silence[0]
        # If first silence is very long, trim it
        if first_segment["duration"] > silence_quota_remaining * 0.25:  # Limit to 25% of quota
            # Split the segment
            desired_duration = min(silence_quota_remaining * 0.2, first_segment["duration"])  # Take at most 20%
            split_point = first_segment["start"] + (desired_duration / 1000)
            
            # Create shortened version
            shortened = first_segment.copy()
            shortened["end"] = split_point
            shortened["duration"] = desired_duration
            balanced_segments.append(shortened)
            silence_quota_remaining -= desired_duration
        else:
            # Use whole segment if it's reasonable length
            balanced_segments.append(first_segment)
            silence_quota_remaining -= first_segment["duration"]
        primary_silence.pop(0)  # Remove first segment as we've used it
    
    # Now alternate between speech and silence in rough chronological order
    speech_index = 0
    silence_index = 0
    
    # This helps us track which type we last added
    last_type_added = "non-speech" if balanced_segments else None
    
    # Alternate between speech and silence until quotas are met
    while (speech_quota_remaining > 0 or silence_quota_remaining > 0) and \
          (speech_index < len(speech_segments) or silence_index < len(primary_silence)):
        
        # Determine which type to add next (alternate when possible)
        add_speech = False
        
        if last_type_added == "speech" and silence_quota_remaining > 0 and silence_index < len(primary_silence):
            # Prefer silence after speech
            add_speech = False
        elif last_type_added == "non-speech" and speech_quota_remaining > 0 and speech_index < len(speech_segments):
            # Prefer speech after silence
            add_speech = True
        else:
            # Add whatever has quota remaining
            if speech_quota_remaining > 0 and speech_index < len(speech_segments):
                add_speech = True
            elif silence_quota_remaining > 0 and silence_index < len(primary_silence):
                add_speech = False
            else:
                break  # No more quota or segments
        
        if add_speech:
            segment = speech_segments[speech_index]
            speech_index += 1
            
            if segment["duration"] <= speech_quota_remaining:
                # Use whole segment
                balanced_segments.append(segment)
                speech_quota_remaining -= segment["duration"]
            else:
                # Need to split segment
                split_point = segment["start"] + (speech_quota_remaining / 1000)
                
                # Create partial segment
                partial = segment.copy()
                partial["end"] = split_point
                partial["duration"] = speech_quota_remaining
                balanced_segments.append(partial)
                speech_quota_remaining = 0
            
            last_type_added = "speech"
        else:
            segment = primary_silence[silence_index]
            silence_index += 1
            
            if segment["duration"] <= silence_quota_remaining:
                # Use whole segment
                balanced_segments.append(segment)
                silence_quota_remaining -= segment["duration"]
            else:
                # Need to split segment
                split_point = segment["start"] + (silence_quota_remaining / 1000)
                
                # Create partial segment
                partial = segment.copy()
                partial["end"] = split_point
                partial["duration"] = silence_quota_remaining
                balanced_segments.append(partial)
                silence_quota_remaining = 0
            
            last_type_added = "non-speech"
    
    # Sort the segments collected so far by start time
    balanced_segments.sort(key=lambda x: x["start"])
    
    # Find where to intersperse the reserved silence segments
    # Look for long runs of speech without any silence
    speech_runs = []
    current_run_start = None
    current_run_end = None
    
    for i, segment in enumerate(balanced_segments):
        if segment["type"] == "speech":
            if current_run_start is None:
                # Start a new speech run
                current_run_start = i
                current_run_end = i
            else:
                # Extend current run
                current_run_end = i
        else:
            # End of speech run
            if current_run_start is not None:
                run_length = current_run_end - current_run_start + 1
                if run_length >= 3:  # Only consider runs of 3+ segments
                    speech_runs.append((current_run_start, current_run_end))
                current_run_start = None
    
    # Add the last run if we ended on speech
    if current_run_start is not None:
        run_length = current_run_end - current_run_start + 1
        if run_length >= 3:
            speech_runs.append((current_run_start, current_run_end))
    
    # Sort runs by length (longest first)
    speech_runs.sort(key=lambda x: x[1]-x[0], reverse=True)
    
    # Intersperse silence in long speech runs
    reserved_index = 0
    new_segments = balanced_segments.copy()
    inserted = 0  # Track how many segments we've inserted
    
    # Loop through speech runs (longest first) and intersperse silence
    for run_start, run_end in speech_runs:
        run_length = run_end - run_start + 1
        
        # Determine how many silence segments to insert in this run
        # For a run of length N, insert N/3 silence segments (rounded down)
        num_to_insert = run_length // 3
        
        if num_to_insert > 0 and reserved_index < len(reserved_silence):
            # Calculate insertion positions (evenly spaced)
            positions = []
            for i in range(1, num_to_insert + 1):
                pos = run_start + (i * run_length) // (num_to_insert + 1)
                positions.append(pos + inserted)  # Adjust for previously inserted segments
            
            # Insert silence segments at these positions
            for pos in positions:
                if reserved_index < len(reserved_silence):
                    new_segments.insert(pos, reserved_silence[reserved_index])
                    reserved_index += 1
                    inserted += 1
    
    # If we have leftover reserved silence, add it at the end
    if reserved_index < len(reserved_silence):
        new_segments.extend(reserved_silence[reserved_index:])
    
    # Final sort to ensure everything is in time order
    balanced_segments = sorted(new_segments, key=lambda x: x["start"])
    
    # Calculate total duration of each type
    balanced_speech_ms = sum(segment["duration"] for segment in balanced_segments if segment["type"] == "speech")
    balanced_non_speech_ms = sum(segment["duration"] for segment in balanced_segments if segment["type"] == "non-speech")
    
    # Collect remaining segments for dev/train
    used_segments = set(id(segment) for segment in balanced_segments)
    remaining_segments = [seg for seg in timeline if id(seg) not in used_segments]
    remaining_segments.sort(key=lambda x: x["start"])
    
    # Split remaining segments into dev and train sets
    dev_segments = []
    train_segments = []
    
    if create_splits and remaining_segments:
        # Calculate total duration of remaining segments
        remaining_speech_ms = sum(s["duration"] for s in remaining_segments if s["type"] == "speech")
        remaining_non_speech_ms = sum(s["duration"] for s in remaining_segments if s["type"] == "non-speech")
        
        # Calculate dev/train split
        dev_speech_ms = remaining_speech_ms * dev_ratio
        dev_non_speech_ms = remaining_non_speech_ms * dev_ratio
        
        # Initialize counters
        current_dev_speech_ms = 0
        current_dev_non_speech_ms = 0
        
        # Allocate segments to dev/train sets
        for segment in remaining_segments:
            is_speech = segment["type"] == "speech"
            
            # Add to dev if we still need content of this type
            if (is_speech and current_dev_speech_ms < dev_speech_ms) or \
               (not is_speech and current_dev_non_speech_ms < dev_non_speech_ms):
                dev_segments.append(segment)
                if is_speech:
                    current_dev_speech_ms += segment["duration"]
                else:
                    current_dev_non_speech_ms += segment["duration"]
            else:
                # Otherwise add to train
                train_segments.append(segment)
        
        # Sort dev and train segments by start time
        dev_segments.sort(key=lambda x: x["start"])
        train_segments.sort(key=lambda x: x["start"])
    
    # Compile audio from segments with timestamp tracking
    def compile_audio_from_segments_with_tracking(segments, set_name):
        compiled_audio = AudioSegment.empty()
        timestamp_map = []
        output_time_ms = 0
        
        for i, segment in enumerate(segments):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            segment_audio = audio[start_ms:end_ms]
            segment_duration_ms = len(segment_audio)
            
            # Fix the continuity tracking: first segment is ALWAYS continuous
            is_continuous = i == 0 or abs(segment["start"] - segments[i-1]["end"]) < 0.001
            
            # Track the mapping between output time and original time
            timestamp_map.append({
                "segment_index": i,
                "original_start_sec": segment["start"],
                "original_end_sec": segment["end"],
                "output_start_sec": output_time_ms / 1000,
                "output_end_sec": (output_time_ms + segment_duration_ms) / 1000,
                "duration_sec": segment_duration_ms / 1000,
                "type": segment["type"],
                "is_continuous": is_continuous
            })
            
            compiled_audio += segment_audio
            output_time_ms += segment_duration_ms
        
        return compiled_audio, timestamp_map
    
    # Compile the three sets: balanced, dev, and train
    balanced_audio, balanced_timestamps = compile_audio_from_segments_with_tracking(balanced_segments, "balanced")
    
    dev_audio = None
    dev_timestamps = []
    train_audio = None
    train_timestamps = []
    
    if create_splits:
        dev_audio, dev_timestamps = compile_audio_from_segments_with_tracking(dev_segments, "dev")
        train_audio, train_timestamps = compile_audio_from_segments_with_tracking(train_segments, "train")
    
    # Create output paths
    balanced_output_path = output_path / f"{file_stem}_balanced_{target_hours:.1f}h.wav"
    dev_output_path = output_path / f"{file_stem}_dev.wav" if create_splits else None
    train_output_path = output_path / f"{file_stem}_train.wav" if create_splits else None
    
    # Create timestamp mapping files
    balanced_timestamp_file = timestamp_dir / f"{file_stem}_balanced_timestamps.txt"
    dev_timestamp_file = timestamp_dir / f"{file_stem}_dev_timestamps.txt" if create_splits else None
    train_timestamp_file = timestamp_dir / f"{file_stem}_train_timestamps.txt" if create_splits else None
    
    # Define function to create timestamp files
    def create_timestamp_file(timestamps, file_path, set_name):
        """Create a detailed timestamp mapping file with improved continuity analysis."""
        # Mark first segment as continuous by definition
        if timestamps:
            timestamps[0]["is_continuous"] = True
            
        continuity_info = analyze_continuity(timestamps)
        sequences = continuity_info["sequences"]
        
        with open(file_path, 'w') as f:
            f.write(f"TIMESTAMP MAPPING FOR {set_name.upper()} SET\n")
            f.write(f"{'='*50}\n\n")
            
            # Continuity summary at the top for quick reference
            f.write(f"CONTINUITY SUMMARY:\n")
            f.write(f"  Total segments: {len(timestamps)}\n")
            if timestamps:
                f.write(f"  Continuous segments: {continuity_info['continuous_segments']} ({continuity_info['continuous_segments']/len(timestamps)*100:.1f}%)\n")
            else:
                f.write("  Continuous segments: 0 (0.0%)\n")
            f.write(f"  Continuous sequences: {continuity_info['num_sequences']}\n")
            f.write(f"  Longest continuous sequence: {continuity_info['longest_sequence']} segments\n")
            f.write(f"  Average sequence length: {continuity_info['avg_sequence_length']:.1f} segments\n\n")
            
            # Sequence mapping for a clearer view of audio structure
            f.write(f"CONTINUOUS SEQUENCES:\n")
            for i, seq in enumerate(sequences):
                duration = seq["end_time"] - seq["start_time"]
                f.write(f"  Sequence {i+1}: {seq['length']} segments, ")
                f.write(f"{format_time_mmss(seq['start_time'])} - {format_time_mmss(seq['end_time'])} ")
                f.write(f"({format_time_mmss(duration)})\n")
            f.write("\n")
            
            # Individual segment mapping
            f.write(f"FORMAT: [Output Time] <- [Original Time] (Type) [Continuity]\n")
            f.write(f"Times in format: MM:SS.mmm\n\n")
            
            for i, ts in enumerate(timestamps):
                output_start = format_time_mmss(ts["output_start_sec"])
                output_end = format_time_mmss(ts["output_end_sec"])
                original_start = format_time_mmss(ts["original_start_sec"])
                original_end = format_time_mmss(ts["original_end_sec"])
                
                continuity = "CONTINUOUS" if i == 0 or ts["is_continuous"] else "JUMP"
                type_indicator = "SPEECH" if ts["type"] == "speech" else "SILENCE"
                
                f.write(f"Segment {i+1:2d}: [{output_start} - {output_end}] <- ")
                f.write(f"[{original_start} - {original_end}] ({type_indicator:7s}) {continuity}\n")
    
    # Save the audio files
    save_audio_safely(balanced_audio, balanced_output_path)
    create_timestamp_file(balanced_timestamps, balanced_timestamp_file, "balanced")
    
    if create_splits:
        if dev_audio and len(dev_audio) > 0:
            save_audio_safely(dev_audio, dev_output_path)
            create_timestamp_file(dev_timestamps, dev_timestamp_file, "dev")
        
        if train_audio and len(train_audio) > 0:
            save_audio_safely(train_audio, train_output_path)
            create_timestamp_file(train_timestamps, train_timestamp_file, "train")
    
    # Calculate statistics
    balanced_duration_ms = len(balanced_audio)
    continuous_segments = sum(1 for ts in balanced_timestamps if ts["is_continuous"])
    temporal_jumps = len(balanced_timestamps) - continuous_segments
    
    # Create metadata file
    metadata_path = output_path / f"{file_stem}_recompile_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("RECOMPILATION METADATA\n")
        f.write("====================\n\n")
        
        f.write(f"Original file: {input_wav}\n")
        f.write(f"Ground truth: {ground_truth}\n")
        f.write(f"Target duration: {target_hours} hours\n")
        f.write(f"Speech padding: {speech_padding_ms} ms\n")
        f.write(f"Silence reserve ratio: {silence_reserve_ratio}\n\n")
        
        f.write(f"Original duration: {format_duration(total_duration_ms)}\n")
        f.write(f"Original speech: {format_duration(total_speech_ms)} ({total_speech_ms/total_duration_ms*100:.1f}%)\n")
        f.write(f"Original non-speech: {format_duration(total_non_speech_ms)} ({total_non_speech_ms/total_duration_ms*100:.1f}%)\n\n")
        
        f.write("BALANCED OUTPUT:\n")
        f.write(f"Duration: {format_duration(balanced_duration_ms)} ({balanced_duration_ms/total_duration_ms*100:.1f}% of original)\n")
        f.write(f"Speech content: {format_duration(balanced_speech_ms)} ({balanced_speech_ms/balanced_duration_ms*100:.1f}%)\n")
        f.write(f"Non-speech content: {format_duration(balanced_non_speech_ms)} ({balanced_non_speech_ms/balanced_duration_ms*100:.1f}%)\n")
        f.write(f"Continuous segments: {continuous_segments}/{len(balanced_timestamps)}\n")
        f.write(f"Temporal jumps: {temporal_jumps}\n\n")
        
        if create_splits:
            dev_duration_ms = len(dev_audio) if dev_audio else 0
            train_duration_ms = len(train_audio) if train_audio else 0
            
            dev_speech_ms = sum(ts["duration_sec"] * 1000 for ts in dev_timestamps if ts["type"] == "speech")
            dev_non_speech_ms = sum(ts["duration_sec"] * 1000 for ts in dev_timestamps if ts["type"] == "non-speech")
            
            train_speech_ms = sum(ts["duration_sec"] * 1000 for ts in train_timestamps if ts["type"] == "speech")
            train_non_speech_ms = sum(ts["duration_sec"] * 1000 for ts in train_timestamps if ts["type"] == "non-speech")
            
            dev_continuous = sum(1 for ts in dev_timestamps if ts["is_continuous"])
            train_continuous = sum(1 for ts in train_timestamps if ts["is_continuous"])
            
            f.write("DEV SET:\n")
            if dev_duration_ms > 0:
                f.write(f"Duration: {format_duration(dev_duration_ms)}\n")
                f.write(f"Speech content: {format_duration(dev_speech_ms)} ({dev_speech_ms/dev_duration_ms*100:.1f}%)\n")
                f.write(f"Non-speech content: {format_duration(dev_non_speech_ms)} ({dev_non_speech_ms/dev_duration_ms*100:.1f}%)\n")
                f.write(f"Continuous segments: {dev_continuous}/{len(dev_timestamps)}\n")
                f.write(f"Temporal jumps: {len(dev_timestamps) - dev_continuous}\n\n")
            else:
                f.write("No DEV set created (insufficient excess audio)\n\n")
            
            f.write("TRAIN SET:\n")
            if train_duration_ms > 0:
                f.write(f"Duration: {format_duration(train_duration_ms)}\n")
                f.write(f"Speech content: {format_duration(train_speech_ms)} ({train_speech_ms/train_duration_ms*100:.1f}%)\n")
                f.write(f"Non-speech content: {format_duration(train_non_speech_ms)} ({train_non_speech_ms/train_duration_ms*100:.1f}%)\n")
                f.write(f"Continuous segments: {train_continuous}/{len(train_timestamps)}\n")
                f.write(f"Temporal jumps: {len(train_timestamps) - train_continuous}\n")
            else:
                f.write("No TRAIN set created (insufficient excess audio)\n")
    
    # Return statistics
    result = {
        "balanced_output": str(balanced_output_path),
        "metadata": str(metadata_path),
        "original_duration_hours": total_duration_ms/1000/3600,
        "balanced_duration_hours": balanced_duration_ms/1000/3600,
        "balanced_speech_hours": balanced_speech_ms/1000/3600,
        "balanced_non_speech_hours": balanced_non_speech_ms/1000/3600,
    }
    
    if create_splits:
        result["dev_output"] = str(dev_output_path) if dev_output_path else None
        result["train_output"] = str(train_output_path) if train_output_path else None
    
    return result

def extract_continuity_stats(timestamp_file):
    """Extract continuity statistics from a timestamp file by directly parsing sequences."""
    try:
        with open(timestamp_file, 'r') as f:
            content = f.read()
            
        # Extract stats from CONTINUITY SUMMARY section if available
        if "CONTINUITY SUMMARY:" in content:
            summary_section = content.split("CONTINUITY SUMMARY:")[1].split("\n\n")[0]
            
            stats = {
                "total_segments": 0,
                "continuous_segments": 0,
                "num_sequences": 0,
                "longest_sequence": 0,
                "avg_sequence_length": 0
            }
            
            for line in summary_section.strip().split('\n'):
                line = line.strip()
                if "Total segments:" in line:
                    stats["total_segments"] = int(line.split(":")[1].strip())
                elif "Continuous segments:" in line:
                    parts = line.split(":")[1].strip().split()
                    # Handle both "15 (21.1%)" and "15/71" formats
                    if "(" in parts[0]:
                        stats["continuous_segments"] = int(parts[0].split("(")[0].strip())
                    elif "/" in parts[0]:
                        stats["continuous_segments"] = int(parts[0].split("/")[0].strip())
                    else:
                        stats["continuous_segments"] = int(parts[0])
                elif "Continuous sequences:" in line:
                    stats["num_sequences"] = int(line.split(":")[1].strip())
                elif "Longest continuous sequence:" in line:
                    parts = line.split(":")[1].strip().split()
                    stats["longest_sequence"] = int(parts[0])
                elif "Average sequence length:" in line:
                    parts = line.split(":")[1].strip().split()
                    stats["avg_sequence_length"] = float(parts[0])
                    
            return stats
                
        # Fallback: Parse individual segment entries to calculate stats
        segments = []
        continuous_count = 0
        
        # Find segment lines using regex
        import re
        segment_pattern = re.compile(r'Segment\s+(\d+):.+\((?:SPEECH|SILENCE)\s*\)\s+(CONTINUOUS|JUMP)')
        for match in segment_pattern.finditer(content):
            is_continuous = match.group(2) == "CONTINUOUS"
            if is_continuous:
                continuous_count += 1
            segments.append({"continuous": is_continuous})
        
        # Calculate sequences
        sequences = []
        current_seq = []
        for i, seg in enumerate(segments):
            if i == 0 or seg["continuous"]:
                current_seq.append(i)
            else:
                if current_seq:
                    sequences.append(current_seq)
                current_seq = [i]
                
        if current_seq:
            sequences.append(current_seq)
            
        return {
            "total_segments": len(segments),
            "continuous_segments": continuous_count,
            "num_sequences": len(sequences),
            "longest_sequence": max([len(seq) for seq in sequences]) if sequences else 0,
            "avg_sequence_length": sum([len(seq) for seq in sequences]) / len(sequences) if sequences else 0
        }
        
    except Exception as e:
        print(f"Error extracting continuity stats from {timestamp_file}: {e}")
        return {
            "total_segments": 0,
            "continuous_segments": 0,
            "num_sequences": 0,
            "longest_sequence": 0,
            "avg_sequence_length": 0
        }

def batch_recompile_audio(
    input_files: List[dict],
    target_hours: float = 1.0,
    speech_padding_ms: int = 200,
    output_dir: str = "Recompiled_Output",
    create_splits: bool = True,
    dev_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Process multiple audio files in batch and provide summary statistics.
    Organizes outputs into TEST/DEV/TRAIN directories.
    """
    # Create output directories for TEST, DEV, TRAIN
    output_path = Path(output_dir)
    test_dir = output_path / "TEST"
    dev_dir = output_path / "DEV"
    train_dir = output_path / "TRAIN"
    
    # Create directories
    test_dir.mkdir(exist_ok=True, parents=True)
    if create_splits:
        dev_dir.mkdir(exist_ok=True, parents=True)
        train_dir.mkdir(exist_ok=True, parents=True)
        
    # Initialize batch statistics
    batch_stats = {
        "total_files": len(input_files),
        "total_original_duration": 0,
        "total_balanced_duration": 0,
        "total_speech_duration": 0,
        "total_non_speech_duration": 0,
        "speech_ratio_accuracy": [],  # How close to 1:1 ratio we achieved
        "continuity_stats": {
            "total_segments": 0,
            "continuous_segments": 0,
            "total_sequences": 0,
            "longest_sequence": 0,
            "avg_sequence_length": 0
        },
        "file_details": []
    }
    
    # Process each file
    for i, file_info in enumerate(input_files):
        print(f"\n[{i+1}/{len(input_files)}] Processing {file_info['audio_path']}...")
        file_stem = Path(file_info["audio_path"]).stem
        
        try:
            # Process this file with primary output going to TEST directory
            result = recompile_balanced_audio(
                input_wav=file_info["audio_path"],
                ground_truth=file_info["ground_truth_path"],
                target_hours=target_hours,
                speech_padding_ms=speech_padding_ms,
                output_dir=str(test_dir),  # Primary balanced output to TEST directory
                create_splits=create_splits,
                dev_ratio=dev_ratio
            )
            
            # If create_splits is enabled, move the DEV/TRAIN files to their respective directories
            if create_splits and "dev_output" in result and "train_output" in result:
                import shutil  # Import here to avoid polluting global namespace
                
                # Move DEV file
                dev_src = Path(result["dev_output"])
                dev_dest = dev_dir / dev_src.name
                if dev_src.exists():
                    shutil.move(str(dev_src), str(dev_dest))
                    
                # Move TRAIN file
                train_src = Path(result["train_output"])
                train_dest = train_dir / train_src.name
                if train_src.exists():
                    shutil.move(str(train_src), str(train_dest))
                    
                # Also move timestamp files if they exist
                timestamp_dir = test_dir / "timestamp_maps"
                if timestamp_dir.exists():
                    # Create destination directories
                    (dev_dir / "timestamp_maps").mkdir(exist_ok=True, parents=True)
                    (train_dir / "timestamp_maps").mkdir(exist_ok=True, parents=True)
                    
                    # Move dev timestamp file
                    dev_ts_src = timestamp_dir / f"{file_stem}_dev_timestamps.txt"
                    dev_ts_dest = dev_dir / "timestamp_maps" / f"{file_stem}_dev_timestamps.txt"
                    if dev_ts_src.exists():
                        shutil.move(str(dev_ts_src), str(dev_ts_dest))
                    
                    # Move train timestamp file
                    train_ts_src = timestamp_dir / f"{file_stem}_train_timestamps.txt"
                    train_ts_dest = train_dir / "timestamp_maps" / f"{file_stem}_train_timestamps.txt"
                    if train_ts_src.exists():
                        shutil.move(str(train_ts_src), str(train_ts_dest))
            
            # Get continuity stats from all sets
            test_stats_file = test_dir / "timestamp_maps" / f"{file_stem}_balanced_timestamps.txt"
            test_stats = extract_continuity_stats(str(test_stats_file)) if test_stats_file.exists() else {
                "total_segments": 0, "continuous_segments": 0, "num_sequences": 0, "longest_sequence": 0
            }
            
            dev_stats = {"total_segments": 0, "continuous_segments": 0, "num_sequences": 0, "longest_sequence": 0}
            train_stats = {"total_segments": 0, "continuous_segments": 0, "num_sequences": 0, "longest_sequence": 0}
            
            if create_splits:
                dev_stats_file = dev_dir / "timestamp_maps" / f"{file_stem}_dev_timestamps.txt"
                if dev_stats_file.exists():
                    dev_stats = extract_continuity_stats(str(dev_stats_file))
                    
                train_stats_file = train_dir / "timestamp_maps" / f"{file_stem}_train_timestamps.txt"
                if train_stats_file.exists():
                    train_stats = extract_continuity_stats(str(train_stats_file))
            
            # Add to batch statistics
            batch_stats["total_original_duration"] += result["original_duration_hours"]
            batch_stats["total_balanced_duration"] += result["balanced_duration_hours"]
            batch_stats["total_speech_duration"] += result["balanced_speech_hours"]
            batch_stats["total_non_speech_duration"] += result["balanced_non_speech_hours"]
            
            # Calculate ratio accuracy (how close to 1:1)
            speech_ratio = result["balanced_speech_hours"] / (result["balanced_speech_hours"] + result["balanced_non_speech_hours"])
            ratio_accuracy = min(speech_ratio, 1-speech_ratio) * 2  # 1.0 = perfect 50:50
            batch_stats["speech_ratio_accuracy"].append(ratio_accuracy)
            
            # Add continuity stats - combine all sets
            batch_stats["continuity_stats"]["total_segments"] += test_stats["total_segments"]
            batch_stats["continuity_stats"]["continuous_segments"] += test_stats["continuous_segments"]
            batch_stats["continuity_stats"]["total_sequences"] += test_stats["num_sequences"]
            
            if test_stats["longest_sequence"] > batch_stats["continuity_stats"]["longest_sequence"]:
                batch_stats["continuity_stats"]["longest_sequence"] = test_stats["longest_sequence"]
                
            # Include DEV and TRAIN stats if available
            if create_splits:
                batch_stats["continuity_stats"]["total_segments"] += dev_stats["total_segments"] + train_stats["total_segments"]
                batch_stats["continuity_stats"]["continuous_segments"] += dev_stats["continuous_segments"] + train_stats["continuous_segments"]
                batch_stats["continuity_stats"]["total_sequences"] += dev_stats["num_sequences"] + train_stats["num_sequences"]
                
                if dev_stats["longest_sequence"] > batch_stats["continuity_stats"]["longest_sequence"]:
                    batch_stats["continuity_stats"]["longest_sequence"] = dev_stats["longest_sequence"]
                    
                if train_stats["longest_sequence"] > batch_stats["continuity_stats"]["longest_sequence"]:
                    batch_stats["continuity_stats"]["longest_sequence"] = train_stats["longest_sequence"]
            
            # Add file details
            file_stats = {
                "filename": Path(file_info["audio_path"]).name,
                "original_hours": result["original_duration_hours"],
                "balanced_hours": result["balanced_duration_hours"],
                "speech_non_speech_ratio": f"{speech_ratio:.2f}:{1-speech_ratio:.2f}",
                "continuous_segments": f"{test_stats['continuous_segments']}/{test_stats['total_segments']}",
                "continuous_sequences": test_stats["num_sequences"],
                "longest_sequence": test_stats["longest_sequence"]
            }
            batch_stats["file_details"].append(file_stats)
            
            print(f"  ✓ Processed successfully - {result['balanced_duration_hours']*60:.1f}min balanced output")
            if create_splits:
                print(f"    - TEST output: {Path(result['balanced_output']).name}")
                print(f"    - DEV output: {dev_dest.name if dev_dest.exists() else 'N/A'}")
                print(f"    - TRAIN output: {train_dest.name if train_dest.exists() else 'N/A'}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print the full error stack trace for debugging
            print(f"  ✗ Error processing file: {e}")
            batch_stats["file_details"].append({
                "filename": Path(file_info["audio_path"]).name,
                "error": str(e)
            })
    
    # Calculate aggregate statistics
    if batch_stats["continuity_stats"]["total_sequences"] > 0:
        batch_stats["continuity_stats"]["avg_sequence_length"] = (
            batch_stats["continuity_stats"]["continuous_segments"] / 
            batch_stats["continuity_stats"]["total_sequences"]
        )
    
    # Create batch summary file in the main output directory
    create_batch_summary(batch_stats, output_dir)
    
    return batch_stats

def analyze_continuity(timestamps):
    """
    Analyze continuity in a more sophisticated way by identifying sequences.
    A sequence is a group of continuous segments without jumps.
    """
    if not timestamps:
        return {
            "sequences": [],
            "num_sequences": 0,
            "longest_sequence": 0,
            "avg_sequence_length": 0,
            "total_segments": 0,
            "continuous_segments": 0
        }
        
    sequences = []
    current_sequence = []
    continuous_count = 0
    
    for i, ts in enumerate(timestamps):
        # First segment is always continuous by definition (nothing comes before it)
        is_continuous = ts["is_continuous"] if i > 0 else True
        
        if is_continuous:
            # Part of current sequence
            if not current_sequence:
                current_sequence = [i]
            else:
                current_sequence.append(i)
            continuous_count += 1
        else:
            # Jump detected - end current sequence and start a new one
            if current_sequence:
                seq_start_idx = current_sequence[0]
                seq_end_idx = current_sequence[-1]
                sequences.append({
                    "segments": current_sequence,
                    "length": len(current_sequence),
                    "start_time": timestamps[seq_start_idx]["output_start_sec"],
                    "end_time": timestamps[seq_end_idx]["output_end_sec"],
                })
            # Start a new sequence with current segment
            current_sequence = [i]
    
    # Add the last sequence if there is one
    if current_sequence:
        seq_start_idx = current_sequence[0]
        seq_end_idx = current_sequence[-1]
        sequences.append({
            "segments": current_sequence,
            "length": len(current_sequence),
            "start_time": timestamps[seq_start_idx]["output_start_sec"],
            "end_time": timestamps[seq_end_idx]["output_end_sec"],
        })
    
    # Calculate stats
    longest_seq = max([s["length"] for s in sequences]) if sequences else 0
    avg_seq_len = sum([s["length"] for s in sequences]) / len(sequences) if sequences else 0
    
    return {
        "sequences": sequences,
        "num_sequences": len(sequences),
        "longest_sequence": longest_seq,
        "avg_sequence_length": avg_seq_len,
        "total_segments": len(timestamps),
        "continuous_segments": continuous_count
    }

def create_timestamp_file(timestamps, file_path, set_name):
    """Create a detailed timestamp mapping file with improved continuity analysis."""
    # Mark first segment as continuous by definition
    if timestamps:
        timestamps[0]["is_continuous"] = True
        
    continuity_info = analyze_continuity(timestamps)
    sequences = continuity_info["sequences"]
    
    with open(file_path, 'w') as f:
        f.write(f"TIMESTAMP MAPPING FOR {set_name.upper()} SET\n")
        f.write(f"{'='*50}\n\n")
        
        # Continuity summary at the top for quick reference
        f.write(f"CONTINUITY SUMMARY:\n")
        f.write(f"  Total segments: {len(timestamps)}\n")
        f.write(f"  Continuous segments: {continuity_info['continuous_segments']} ({continuity_info['continuous_segments']/len(timestamps)*100:.1f}%)\n")
        f.write(f"  Continuous sequences: {continuity_info['num_sequences']}\n")
        f.write(f"  Longest continuous sequence: {continuity_info['longest_sequence']} segments\n")
        f.write(f"  Average sequence length: {continuity_info['avg_sequence_length']:.1f} segments\n\n")
        
        # Sequence mapping for a clearer view of audio structure
        f.write(f"CONTINUOUS SEQUENCES:\n")
        for i, seq in enumerate(sequences):
            duration = seq["end_time"] - seq["start_time"]
            f.write(f"  Sequence {i+1}: {seq['length']} segments, ")
            f.write(f"{format_time_mmss(seq['start_time'])} - {format_time_mmss(seq['end_time'])} ")
            f.write(f"({format_time_mmss(duration)})\n")
        f.write("\n")
        
        # Individual segment mapping
        f.write(f"FORMAT: [Output Time] <- [Original Time] (Type) [Continuity]\n")
        f.write(f"Times in format: MM:SS.mmm\n\n")
        
        for i, ts in enumerate(timestamps):
            output_start = format_time_mmss(ts["output_start_sec"])
            output_end = format_time_mmss(ts["output_end_sec"])
            original_start = format_time_mmss(ts["original_start_sec"])
            original_end = format_time_mmss(ts["original_end_sec"])
            
            continuity = "CONTINUOUS" if i == 0 or ts["is_continuous"] else "JUMP"
            type_indicator = "SPEECH" if ts["type"] == "speech" else "SILENCE"
            
            f.write(f"Segment {i+1:2d}: [{output_start} - {output_end}] <- ")
            f.write(f"[{original_start} - {original_end}] ({type_indicator:7s}) {continuity}\n")

def create_batch_summary(batch_stats, output_dir):
    """Create a summary file for the entire batch processing job."""
    output_path = Path(output_dir)
    summary_file = output_path / "batch_processing_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("AUDIO RECOMPILATION - BATCH PROCESSING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Files processed: {batch_stats['total_files']}\n")
        f.write(f"  Total original duration: {batch_stats['total_original_duration']:.2f} hours\n")
        f.write(f"  Total balanced output: {batch_stats['total_balanced_duration']:.2f} hours\n")
        f.write(f"  Speech content: {batch_stats['total_speech_duration']:.2f} hours ")
        speech_pct = batch_stats['total_speech_duration'] / batch_stats['total_balanced_duration'] * 100
        f.write(f"({speech_pct:.1f}%)\n")
        f.write(f"  Non-speech content: {batch_stats['total_non_speech_duration']:.2f} hours ")
        nonspeech_pct = batch_stats['total_non_speech_duration'] / batch_stats['total_balanced_duration'] * 100
        f.write(f"({nonspeech_pct:.1f}%)\n")
        
        # Target ratio achievement
        avg_ratio_accuracy = sum(batch_stats['speech_ratio_accuracy']) / len(batch_stats['speech_ratio_accuracy'])
        f.write(f"  Average 1:1 ratio accuracy: {avg_ratio_accuracy*100:.1f}% (100% = perfect)\n\n")
        
        # Continuity statistics
        cont_stats = batch_stats['continuity_stats']
        f.write("CONTINUITY STATISTICS:\n")
        f.write(f"  Total segments: {cont_stats['total_segments']}\n")
        f.write(f"  Continuous segments: {cont_stats['continuous_segments']} ")
        cont_pct = cont_stats['continuous_segments'] / cont_stats['total_segments'] * 100 if cont_stats['total_segments'] > 0 else 0
        f.write(f"({cont_pct:.1f}%)\n")
        f.write(f"  Continuous sequences: {cont_stats['total_sequences']}\n")
        f.write(f"  Longest continuous sequence: {cont_stats['longest_sequence']} segments\n")
        f.write(f"  Average sequence length: {cont_stats['avg_sequence_length']:.1f} segments\n\n")
        
        # Per-file summary
        f.write("PER-FILE SUMMARY:\n")
        f.write(f"{'Filename':30} | {'Duration':9} | {'Ratio':8} | {'Continuity':15} | {'Sequences':8}\n")
        f.write("-"*80 + "\n")
        
        for file_stat in batch_stats['file_details']:
            if 'error' in file_stat:
                f.write(f"{file_stat['filename']:30} | ERROR: {file_stat['error']}\n")
                continue
                
            f.write(f"{file_stat['filename'][:30]:30} | ")
            f.write(f"{file_stat['balanced_hours']*60:5.1f}min | ")
            f.write(f"{file_stat['speech_non_speech_ratio']:8} | ")
            f.write(f"{file_stat['continuous_segments']:15} | ")
            f.write(f"{file_stat['continuous_sequences']:3} ({file_stat['longest_sequence']} max)\n")
        
        print(f"\nBatch processing summary saved to {summary_file}")

def process_audio_directory(
    input_dir="input_data", 
    output_dir="Recompiled_Output", 
    target_hours=0.135, 
    speech_padding_ms=200,
    create_splits=True
):
    """
    Process all audio files in a directory to create balanced TEST/DEV/TRAIN sets.
    This is a wrapper function for easy integration into pipelines.
    
    Args:
        input_dir: Input directory containing 'audio' and 'ground_truth' subdirectories
        output_dir: Output directory for processed files
        target_hours: Target hours for balanced output
        speech_padding_ms: Padding to add around speech segments (ms)
        create_splits: Whether to create DEV/TRAIN splits
        
    Returns:
        Dictionary with batch processing statistics
    """
    # Scan input directory
    input_files = scan_input_directory(input_dir)
    
    if not input_files:
        print(f"No valid files found in {input_dir}. Check directory structure.")
        return None
    
    # Process files
    batch_stats = batch_recompile_audio(
        input_files=input_files,
        target_hours=target_hours,
        speech_padding_ms=speech_padding_ms,
        output_dir=output_dir,
        create_splits=create_splits
    )
    
    return batch_stats

if __name__ == "__main__":
    stats = process_audio_directory(
        input_dir="input_data",              # Directory with audio/ and ground_truth/ folders
        output_dir="Recompiled_Output",      # Where to save the TEST/TRAIN/DEV outputs
        target_hours=0.135,                  # Create x-hour balanced files 
        speech_padding_ms=200,               # Add 200ms padding to speech segments
        create_splits=True                   # Create TRAIN/DEV splits from unused audio
    )
    
    if stats:
        print(f"Processed {stats['total_files']} files")
        print(f"Total balanced output: {stats['total_balanced_duration']:.2f} hours")
