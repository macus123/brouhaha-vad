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
    output_dir: str = "Recompiled_Output",
    create_splits: bool = True,
    dev_ratio: float = 0.2
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

    file_stem = Path(input_wav).stem

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
    Improved segment selection logic that preserves speech context
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
    
    # segment selection logic
    balanced_segments = []
    remaining_segments = []
    
    balanced_speech_ms = 0
    balanced_non_speech_ms = 0
    
    for segment in timeline:
        is_speech = segment["type"] == "speech"
        
        if is_speech:
            # For speech segments, try to include them whole to preserve context
            if balanced_speech_ms < target_per_type_ms:
                if balanced_speech_ms + segment["duration"] <= target_per_type_ms:
                    # Speech segment fits completely within quota
                    balanced_segments.append(segment)
                    balanced_speech_ms += segment["duration"]
                else:
                    # Speech segment would exceed quota
                    # Check if we can accommodate it by being flexible with the balance
                    # Allow up to 10% overage to preserve speech context
                    max_overage = target_per_type_ms * 0.1
                    if balanced_speech_ms + segment["duration"] <= target_per_type_ms + max_overage:
                        balanced_segments.append(segment)
                        balanced_speech_ms += segment["duration"]
                        print(f"Including full speech segment with small overage to preserve context")
                    else:
                        # Segment is too large, add to remaining
                        remaining_segments.append(segment)
            else:
                # Speech quota reached, add to remaining
                remaining_segments.append(segment)
        else:
            # For non-speech segments, we can be more flexible with splitting
            if balanced_non_speech_ms < target_per_type_ms:
                remaining_needed = target_per_type_ms - balanced_non_speech_ms
                
                if segment["duration"] <= remaining_needed:
                    # Use the whole segment
                    balanced_segments.append(segment)
                    balanced_non_speech_ms += segment["duration"]
                else:
                    # Split non-speech segment (this is acceptable for non-speech)
                    split_point = segment["start"] + (remaining_needed / 1000)
                    
                    # Add partial segment to balanced output
                    balanced_part = segment.copy()
                    balanced_part["end"] = split_point
                    balanced_part["duration"] = remaining_needed
                    balanced_segments.append(balanced_part)
                    balanced_non_speech_ms += remaining_needed
                    
                    # Add rest to remaining
                    remaining_part = segment.copy()
                    remaining_part["start"] = split_point
                    remaining_part["duration"] = segment["duration"] - remaining_needed
                    remaining_segments.append(remaining_part)
            else:
                # Non-speech quota reached, add to remaining
                remaining_segments.append(segment)    
    # Now process remaining segments for DEV/TRAIN splits if requested
    if create_splits and remaining_segments:
        # Separate remaining segments by type
        remaining_speech = [s for s in remaining_segments if s["type"] == "speech"]
        remaining_non_speech = [s for s in remaining_segments if s["type"] == "non-speech"]
        
        # Calculate total remaining durations
        total_remaining_speech_ms = sum(s["duration"] for s in remaining_speech)
        total_remaining_non_speech_ms = sum(s["duration"] for s in remaining_non_speech)
        
        # Create DEV set (20% of remaining, balanced)
        dev_speech_target = total_remaining_speech_ms * dev_ratio
        dev_non_speech_target = total_remaining_non_speech_ms * dev_ratio
        
        dev_segments = []
        train_segments = []
        
        dev_speech_ms = 0
        dev_non_speech_ms = 0
        
        # Process remaining segments maintaining temporal order
        all_remaining = remaining_speech + remaining_non_speech
        all_remaining.sort(key=lambda x: x["start"])
        
        for segment in all_remaining:
            is_speech = segment["type"] == "speech"
            
            # Determine if this should go to DEV or TRAIN
            if is_speech:
                if dev_speech_ms < dev_speech_target:
                    if dev_speech_ms + segment["duration"] <= dev_speech_target * 1.1:  # 10% flexibility
                        dev_segments.append(segment)
                        dev_speech_ms += segment["duration"]
                    else:
                        train_segments.append(segment)
                else:
                    train_segments.append(segment)
            else:
                if dev_non_speech_ms < dev_non_speech_target:
                    if dev_non_speech_ms + segment["duration"] <= dev_non_speech_target * 1.1:  # 10% flexibility
                        dev_segments.append(segment)
                        dev_non_speech_ms += segment["duration"]
                    else:
                        train_segments.append(segment)
                else:
                    train_segments.append(segment)
        
        # Sort all segment collections by start time
        dev_segments.sort(key=lambda x: x["start"])
        train_segments.sort(key=lambda x: x["start"])
    else:
        dev_segments = []
        train_segments = remaining_segments
    
    # Sort all segment collections by start time
    balanced_segments.sort(key=lambda x: x["start"])
      # Compile the audio files and track timestamps
    def compile_audio_from_segments_with_tracking(segments, set_name):
        compiled_audio = AudioSegment.empty()
        timestamp_map = []
        output_time_ms = 0
        
        for i, segment in enumerate(segments):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            segment_audio = audio[start_ms:end_ms]
            segment_duration_ms = len(segment_audio)
            
            # Track the mapping between output time and original time
            timestamp_map.append({
                "segment_index": i,
                "original_start_sec": segment["start"],
                "original_end_sec": segment["end"],
                "output_start_sec": output_time_ms / 1000,
                "output_end_sec": (output_time_ms + segment_duration_ms) / 1000,
                "duration_sec": segment_duration_ms / 1000,
                "type": segment["type"],
                "is_continuous": i == 0 or (segment["start"] == segments[i-1]["end"])
            })
            
            compiled_audio += segment_audio
            output_time_ms += segment_duration_ms
        
        return compiled_audio, timestamp_map
    
    balanced_audio, balanced_timestamps = compile_audio_from_segments_with_tracking(balanced_segments, "balanced")
    
    if create_splits:
        dev_audio, dev_timestamps = compile_audio_from_segments_with_tracking(dev_segments, "dev")
        train_audio, train_timestamps = compile_audio_from_segments_with_tracking(train_segments, "train")
    else:
        dev_audio = AudioSegment.empty()
        train_audio = AudioSegment.empty()
        dev_timestamps = []
        train_timestamps = []
    # Generate detailed timestamp metadata files
    def create_timestamp_file(timestamps, file_path, set_name):
        """Create a detailed timestamp mapping file."""
        with open(file_path, 'w') as f:
            f.write(f"TIMESTAMP MAPPING FOR {set_name.upper()} SET\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Format: [Output Time] <- [Original Time] (Type) [Continuity]\n")
            f.write(f"Times in format: MM:SS.mmm\n\n")
            
            total_continuous_duration = 0
            total_jumps = 0
            jump_details = []
            
            for i, ts in enumerate(timestamps):
                output_start = format_time_mmss(ts["output_start_sec"])
                output_end = format_time_mmss(ts["output_end_sec"])
                original_start = format_time_mmss(ts["original_start_sec"])
                original_end = format_time_mmss(ts["original_end_sec"])
                
                continuity = "CONTINUOUS" if ts["is_continuous"] else "JUMP"
                type_indicator = "SPEECH" if ts["type"] == "speech" else "SILENCE"
                
                f.write(f"Segment {i+1:2d}: [{output_start} - {output_end}] <- ")
                f.write(f"[{original_start} - {original_end}] ({type_indicator:7s}) {continuity}\n")
                
                if ts["is_continuous"]:
                    total_continuous_duration += ts["duration_sec"]
                else:
                    total_jumps += 1
                    if i > 0:
                        prev_end = timestamps[i-1]["original_end_sec"]
                        jump_size = ts["original_start_sec"] - prev_end
                        jump_details.append({
                            "segment": i+1,
                            "jump_size_sec": jump_size,
                            "from_time": prev_end,
                            "to_time": ts["original_start_sec"]
                        })
            
            # Summary statistics
            f.write(f"\n{'='*50}\n")
            f.write(f"CONTINUITY ANALYSIS\n")
            f.write(f"{'='*50}\n")
            f.write(f"Total segments: {len(timestamps)}\n")
            f.write(f"Continuous segments: {len(timestamps) - total_jumps}\n")
            f.write(f"Temporal jumps: {total_jumps}\n")
            
            if total_jumps > 0:
                f.write(f"\nJUMP DETAILS:\n")
                for jump in jump_details:
                    jump_time = format_time_mmss(jump["jump_size_sec"])
                    from_time = format_time_mmss(jump["from_time"])
                    to_time = format_time_mmss(jump["to_time"])
                    f.write(f"  Segment {jump['segment']}: Jump of {jump_time} ")
                    f.write(f"(from {from_time} to {to_time})\n")
                
                avg_jump = sum(j["jump_size_sec"] for j in jump_details) / len(jump_details)
                f.write(f"  Average jump size: {format_time_mmss(avg_jump)}\n")
                max_jump = max(j["jump_size_sec"] for j in jump_details)
                f.write(f"  Largest jump: {format_time_mmss(max_jump)}\n")
            
            f.write(f"\nTEMPORAL COVERAGE:\n")
            if timestamps:
                original_span = timestamps[-1]["original_end_sec"] - timestamps[0]["original_start_sec"]
                coverage_ratio = sum(ts["duration_sec"] for ts in timestamps) / original_span * 100
                f.write(f"  Original timespan covered: {format_time_mmss(timestamps[0]['original_start_sec'])} to {format_time_mmss(timestamps[-1]['original_end_sec'])}\n")
                f.write(f"  Total original span: {format_time_mmss(original_span)}\n")
                f.write(f"  Actual usage ratio: {coverage_ratio:.1f}%\n")
    
    def format_time_mmss(seconds):
        """Format seconds as MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    # Create timestamp files for each output
    timestamp_dir = output_path / "timestamp_maps"
    timestamp_dir.mkdir(exist_ok=True)
    
    balanced_timestamp_file = timestamp_dir / f"{file_stem}_balanced_timestamps.txt"
    create_timestamp_file(balanced_timestamps, balanced_timestamp_file, "balanced")
    
    if create_splits:
        dev_timestamp_file = timestamp_dir / f"{file_stem}_dev_timestamps.txt"
        train_timestamp_file = timestamp_dir / f"{file_stem}_train_timestamps.txt"
        create_timestamp_file(dev_timestamps, dev_timestamp_file, "dev")
        create_timestamp_file(train_timestamps, train_timestamp_file, "train")
    
    # Save output files
    file_stem = Path(input_wav).stem
    balanced_output_path = output_path / f"{file_stem}_balanced_{target_hours:.1f}h.wav"
    dev_output_path = output_path / f"{file_stem}_dev.wav"
    train_output_path = output_path / f"{file_stem}_train.wav"
    
    # Use safe audio saving
    sample_rate = audio.frame_rate
    save_audio_safely(balanced_audio, balanced_output_path, sample_rate)
    
    if create_splits:
        save_audio_safely(dev_audio, dev_output_path, sample_rate)
        save_audio_safely(train_audio, train_output_path, sample_rate)
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
        
        f.write(f"BALANCED OUTPUT:\n")
        f.write(f"================\n")
        f.write(f"File: {balanced_output_path.name}\n")
        f.write(f"Duration: {format_duration(len(balanced_audio))}\n")
        f.write(f"Speech: {format_duration(balanced_speech_ms)} ({balanced_speech_ms/(balanced_speech_ms+balanced_non_speech_ms)*100:.1f}%)\n")
        f.write(f"Non-speech: {format_duration(balanced_non_speech_ms)} ({balanced_non_speech_ms/(balanced_speech_ms+balanced_non_speech_ms)*100:.1f}%)\n")
        f.write(f"Segments used: {len(balanced_segments)}\n")
        
        # Calculate continuity stats for balanced set
        continuous_segments = sum(1 for ts in balanced_timestamps if ts["is_continuous"])
        f.write(f"Continuous segments: {continuous_segments}/{len(balanced_timestamps)}\n")
        f.write(f"Temporal jumps: {len(balanced_timestamps) - continuous_segments}\n")
        f.write(f"Detailed timestamps: timestamp_maps/{file_stem}_balanced_timestamps.txt\n\n")
        
        if create_splits:
            f.write(f"DEV SET:\n")
            f.write(f"========\n")
            f.write(f"File: {dev_output_path.name}\n")
            f.write(f"Duration: {format_duration(len(dev_audio))}\n")
            if len(dev_audio) > 0:
                dev_speech_duration = sum(s["duration"] for s in dev_segments if s["type"] == "speech")
                dev_non_speech_duration = sum(s["duration"] for s in dev_segments if s["type"] == "non-speech")
                f.write(f"Speech: {format_duration(dev_speech_duration)} ({dev_speech_duration/(dev_speech_duration+dev_non_speech_duration)*100:.1f}%)\n")
                f.write(f"Non-speech: {format_duration(dev_non_speech_duration)} ({dev_non_speech_duration/(dev_speech_duration+dev_non_speech_duration)*100:.1f}%)\n")
                f.write(f"Segments used: {len(dev_segments)}\n")
                dev_continuous = sum(1 for ts in dev_timestamps if ts["is_continuous"])
                f.write(f"Continuous segments: {dev_continuous}/{len(dev_timestamps)}\n")
                f.write(f"Temporal jumps: {len(dev_timestamps) - dev_continuous}\n")
                f.write(f"Detailed timestamps: timestamp_maps/{file_stem}_dev_timestamps.txt\n\n")
            
            f.write(f"TRAIN SET:\n")
            f.write(f"==========\n")
            f.write(f"File: {train_output_path.name}\n")
            f.write(f"Duration: {format_duration(len(train_audio))}\n")
            if len(train_audio) > 0:
                train_speech_duration = sum(s["duration"] for s in train_segments if s["type"] == "speech")
                train_non_speech_duration = sum(s["duration"] for s in train_segments if s["type"] == "non-speech")
                f.write(f"Speech: {format_duration(train_speech_duration)} ({train_speech_duration/(train_speech_duration+train_non_speech_duration)*100:.1f}%)\n")
                f.write(f"Non-speech: {format_duration(train_non_speech_duration)} ({train_non_speech_duration/(train_non_speech_duration+train_non_speech_duration)*100:.1f}%)\n")
                f.write(f"Segments used: {len(train_segments)}\n")
                train_continuous = sum(1 for ts in train_timestamps if ts["is_continuous"])
                f.write(f"Continuous segments: {train_continuous}/{len(train_timestamps)}\n")
                f.write(f"Temporal jumps: {len(train_timestamps) - train_continuous}\n")
                f.write(f"Detailed timestamps: timestamp_maps/{file_stem}_train_timestamps.txt\n")
    
    # Return statistics
    result = {
        "balanced_output": str(balanced_output_path),
        "metadata": str(metadata_path),
        "balanced_duration_hours": len(balanced_audio)/1000/3600,
        "balanced_speech_hours": balanced_speech_ms/1000/3600,
        "balanced_non_speech_hours": balanced_non_speech_ms/1000/3600,
    }
    
    if create_splits:
        result.update({
            "dev_output": str(dev_output_path),
            "train_output": str(train_output_path),
            "dev_duration_hours": len(dev_audio)/1000/3600,
            "train_duration_hours": len(train_audio)/1000/3600,
        })
    
    return result

def format_duration(ms):
    """Format duration in milliseconds to a readable string."""
    seconds = ms / 1000
    return str(datetime.timedelta(seconds=seconds))

if __name__ == "__main__":
    result = recompile_balanced_audio(
        input_wav="my_gt_data/audio/recording.wav",
        ground_truth="my_gt_data/ground_truth/recording.txt",
        target_hours=0.1350,
        speech_padding_ms=300,
        output_dir="my_gt_data/Recompiled_Output",
        create_splits=True,
        dev_ratio=0.2
    )

    print(f"Balanced output: {result['balanced_output']}")
    print(f"Speech content: {result['balanced_speech_hours']:.2f} hours")
    print(f"Non-speech content: {result['balanced_non_speech_hours']:.2f} hours")
    
    if 'dev_output' in result:
        print(f"DEV set: {result['dev_output']}")
        print(f"DEV duration: {result['dev_duration_hours']:.2f} hours")
        print(f"TRAIN set: {result['train_output']}")
        print(f"TRAIN duration: {result['train_duration_hours']:.2f} hours")
    
    print(f"Metadata saved to: {result['metadata']}")