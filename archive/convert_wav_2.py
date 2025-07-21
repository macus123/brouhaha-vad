import csv
import datetime
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List

from pydub import AudioSegment

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
    print(f"Target: {target_hours} hours ({target_ms / 1000:.1f} seconds) with 1:1 speech/non-speech ratio")

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
    print(f"Speech content: {format_duration(total_speech_ms)} ({total_speech_ms / total_duration_ms * 100:.1f}%)")
    print(
        f"Non-speech content: {format_duration(total_non_speech_ms)} ({total_non_speech_ms / total_duration_ms * 100:.1f}%)")

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
    speech_runs.sort(key=lambda x: x[1] - x[0], reverse=True)

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

    # Calculate total duration of each type in the final balanced set
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
    def compile_audio_from_segments_with_tracking(segments):
        compiled_audio = AudioSegment.empty()
        timestamp_map = []
        output_time_ms = 0

        for i, segment in enumerate(segments):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            segment_audio = audio[start_ms:end_ms]
            segment_duration_ms = len(segment_audio)

            # Fix the continuity tracking: first segment is ALWAYS continuous
            is_continuous = i == 0 or abs(segment["start"] - segments[i - 1]["end"]) < 0.001

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
    balanced_audio, balanced_timestamps = compile_audio_from_segments_with_tracking(balanced_segments)

    dev_audio = None
    dev_timestamps = []
    train_audio = None
    train_timestamps = []

    if create_splits:
        dev_audio, dev_timestamps = compile_audio_from_segments_with_tracking(dev_segments)
        train_audio, train_timestamps = compile_audio_from_segments_with_tracking(train_segments)

    # Create output paths
    balanced_output_path = output_path / f"{file_stem}_balanced_{target_hours:.1f}h.wav"
    dev_output_path = output_path / f"{file_stem}_dev.wav" if create_splits else None
    train_output_path = output_path / f"{file_stem}_train.wav" if create_splits else None

    # Save the audio files
    save_audio_safely(balanced_audio, balanced_output_path)

    if create_splits:
        if dev_audio and len(dev_audio) > 0:
            save_audio_safely(dev_audio, dev_output_path)

        if train_audio and len(train_audio) > 0:
            save_audio_safely(train_audio, train_output_path)

    # Return statistics and generated data
    result = {
        "balanced_output": str(balanced_output_path),
        "original_duration_hours": total_duration_ms / 1000 / 3600,
        "balanced_audio": balanced_audio,
        "balanced_timestamps": balanced_timestamps,
        "balanced_speech_ms": balanced_speech_ms,
        "balanced_non_speech_ms": balanced_non_speech_ms,
    }

    if create_splits:
        result["dev_output"] = str(dev_output_path) if dev_output_path else None
        result["train_output"] = str(train_output_path) if train_output_path else None
        result["dev_audio"] = dev_audio
        result["dev_timestamps"] = dev_timestamps
        result["train_audio"] = train_audio
        result["train_timestamps"] = train_timestamps

    return result


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
        print(f"\n[{i + 1}/{len(input_files)}] Processing {file_info['audio_path']}...")
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

            balanced_audio = result["balanced_audio"]
            balanced_duration_hours = len(balanced_audio) / 3600000
            balanced_speech_hours = result["balanced_speech_ms"] / 3600000
            balanced_non_speech_hours = result["balanced_non_speech_ms"] / 3600000

            # If create_splits is enabled, move the DEV/TRAIN files to their respective directories
            if create_splits and "dev_output" in result and "train_output" in result:
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

            # Get continuity stats directly from the returned timestamp lists
            test_stats = analyze_continuity(result["balanced_timestamps"])

            dev_stats = {}
            train_stats = {}

            if create_splits:
                dev_timestamps = result.get("dev_timestamps", [])
                dev_audio = result.get("dev_audio")
                if dev_timestamps and dev_audio and len(dev_audio) > 0:
                    dev_stats = analyze_continuity(dev_timestamps)
                    dev_duration_ms = len(dev_audio)
                    dev_speech_ms = sum(ts["duration_sec"] * 1000 for ts in dev_timestamps if ts["type"] == "speech")
                    dev_stats["duration_hours"] = dev_duration_ms / 3600000
                    dev_stats["speech_hours"] = dev_speech_ms / 3600000
                else:
                    dev_stats = {"total_segments": 0, "continuous_segments": 0, "num_sequences": 0,
                                 "longest_sequence": 0, "duration_hours": 0, "speech_hours": 0}

                train_timestamps = result.get("train_timestamps", [])
                train_audio = result.get("train_audio")
                if train_timestamps and train_audio and len(train_audio) > 0:
                    train_stats = analyze_continuity(train_timestamps)
                    train_duration_ms = len(train_audio)
                    train_speech_ms = sum(
                        ts["duration_sec"] * 1000 for ts in train_timestamps if ts["type"] == "speech")
                    train_stats["duration_hours"] = train_duration_ms / 3600000
                    train_stats["speech_hours"] = train_speech_ms / 3600000
                else:
                    train_stats = {"total_segments": 0, "continuous_segments": 0, "num_sequences": 0,
                                   "longest_sequence": 0, "duration_hours": 0, "speech_hours": 0}

            # Add to batch statistics
            batch_stats["total_original_duration"] += result["original_duration_hours"]
            batch_stats["total_balanced_duration"] += balanced_duration_hours
            batch_stats["total_speech_duration"] += balanced_speech_hours
            batch_stats["total_non_speech_duration"] += balanced_non_speech_hours

            # Calculate ratio accuracy (how close to 1:1)
            if balanced_duration_hours > 0:
                speech_ratio = balanced_speech_hours / balanced_duration_hours
                ratio_accuracy = min(speech_ratio, 1 - speech_ratio) * 2  # 1.0 = perfect 50:50
                batch_stats["speech_ratio_accuracy"].append(ratio_accuracy)

            # Add continuity stats - combine all sets
            batch_stats["continuity_stats"]["total_segments"] += test_stats.get("total_segments", 0)
            batch_stats["continuity_stats"]["continuous_segments"] += test_stats.get("continuous_segments", 0)
            batch_stats["continuity_stats"]["total_sequences"] += test_stats.get("num_sequences", 0)
            if test_stats.get("longest_sequence", 0) > batch_stats["continuity_stats"]["longest_sequence"]:
                batch_stats["continuity_stats"]["longest_sequence"] = test_stats.get("longest_sequence", 0)

            if create_splits:
                batch_stats["continuity_stats"]["total_segments"] += dev_stats.get("total_segments", 0) + train_stats.get("total_segments", 0)
                batch_stats["continuity_stats"]["continuous_segments"] += dev_stats.get("continuous_segments", 0) + train_stats.get("continuous_segments", 0)
                batch_stats["continuity_stats"]["total_sequences"] += dev_stats.get("num_sequences", 0) + train_stats.get("num_sequences", 0)
                if dev_stats.get("longest_sequence", 0) > batch_stats["continuity_stats"]["longest_sequence"]:
                    batch_stats["continuity_stats"]["longest_sequence"] = dev_stats.get("longest_sequence", 0)
                if train_stats.get("longest_sequence", 0) > batch_stats["continuity_stats"]["longest_sequence"]:
                    batch_stats["continuity_stats"]["longest_sequence"] = train_stats.get("longest_sequence", 0)

            # Add file details for CSV reporting
            speech_ratio_val = balanced_speech_hours / balanced_duration_hours if balanced_duration_hours > 0 else 0
            file_stats = {
                "filename": Path(file_info["audio_path"]).name,
                "original_hours": result["original_duration_hours"],
                "balanced_hours": balanced_duration_hours,
                "speech_hours": balanced_speech_hours,
                "speech_non_speech_ratio": f"{speech_ratio_val:.2f}:{1 - speech_ratio_val:.2f}",
                "continuous_segments": f"{test_stats.get('continuous_segments', 0)}/{test_stats.get('total_segments', 0)}",
                "continuous_sequences": test_stats.get('num_sequences', 0),
                "longest_sequence": test_stats.get('longest_sequence', 0),
                "dev_stats": dev_stats,
                "train_stats": train_stats,
            }
            batch_stats["file_details"].append(file_stats)

            print(f"  ✓ Processed successfully - {balanced_duration_hours * 60:.1f}min balanced output")
            if create_splits:
                print(f"    - TEST output: {Path(result['balanced_output']).name}")
                if 'dev_dest' in locals() and dev_dest.exists():
                     print(f"    - DEV output: {dev_dest.name}")
                if 'train_dest' in locals() and train_dest.exists():
                     print(f"    - TRAIN output: {train_dest.name}")


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
    create_batch_summary_csv(batch_stats, output_dir)

    return batch_stats


def analyze_continuity(timestamps):
    """
    Analyze continuity in a more sophisticated way by identifying sequences.
    A sequence is a group of continuous segments without jumps.
    """
    if not timestamps:
        return {
            "sequences": [], "num_sequences": 0, "longest_sequence": 0, "avg_sequence_length": 0,
            "total_segments": 0, "continuous_segments": 0
        }

    sequences = []
    current_sequence = []
    continuous_count = 0

    for i, ts in enumerate(timestamps):
        # First segment is always continuous by definition (nothing comes before it)
        is_continuous = ts["is_continuous"] if i > 0 else True

        if is_continuous:
            if not current_sequence:
                current_sequence = [i]
            else:
                current_sequence.append(i)
            continuous_count += 1
        else:
            if current_sequence:
                seq_start_idx = current_sequence[0]
                seq_end_idx = current_sequence[-1]
                sequences.append({
                    "segments": current_sequence,
                    "length": len(current_sequence),
                    "start_time": timestamps[seq_start_idx]["output_start_sec"],
                    "end_time": timestamps[seq_end_idx]["output_end_sec"],
                })
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

    longest_seq = max([s["length"] for s in sequences]) if sequences else 0
    avg_seq_len = sum([s["length"] for s in sequences]) / len(sequences) if sequences else 0

    return {
        "sequences": sequences, "num_sequences": len(sequences), "longest_sequence": longest_seq,
        "avg_sequence_length": avg_seq_len, "total_segments": len(timestamps), "continuous_segments": continuous_count
    }


def create_batch_summary_csv(batch_stats, output_dir):
    """Create a CSV summary file for the entire batch processing job."""
    output_path = Path(output_dir)
    summary_file = output_path / "batch_processing_summary.csv"

    headers = [
        "Filename", "SplitType", "OriginalDuration(h)", "OutputDuration(h)",
        "SpeechDuration(h)", "SpeechPercentage", "SilenceDuration(h)", "SilencePercentage",
        "TotalSegments", "ContinuousSegments", "Sequences", "LongestSequence"
    ]

    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for file_detail in batch_stats['file_details']:
            if 'error' in file_detail:
                writer.writerow({"Filename": file_detail['filename'], "SplitType": "ERROR"})
                continue

            # Add TEST (balanced) entry
            speech_hours = file_detail.get('speech_hours', 0)
            total_hours = file_detail.get('balanced_hours', 0)
            speech_percentage = (speech_hours / total_hours) * 100 if total_hours > 0 else 0
            silence_hours = total_hours - speech_hours
            segments_info = file_detail.get('continuous_segments', '0/0').split('/')
            total_segments = int(segments_info[1]) if len(segments_info) > 1 else 0
            continuous_segments = int(segments_info[0]) if len(segments_info) > 0 else 0

            writer.writerow({
                "Filename": file_detail['filename'], "SplitType": "TEST",
                "OriginalDuration(h)": f"{file_detail.get('original_hours', 0):.4f}",
                "OutputDuration(h)": f"{total_hours:.4f}",
                "SpeechDuration(h)": f"{speech_hours:.4f}",
                "SpeechPercentage": f"{speech_percentage:.2f}",
                "SilenceDuration(h)": f"{silence_hours:.4f}",
                "SilencePercentage": f"{100 - speech_percentage:.2f}",
                "TotalSegments": total_segments, "ContinuousSegments": continuous_segments,
                "Sequences": file_detail.get('continuous_sequences', 0),
                "LongestSequence": file_detail.get('longest_sequence', 0)
            })

            # Add DEV entry if it exists and is not empty
            if 'dev_stats' in file_detail and file_detail['dev_stats'].get('duration_hours', 0) > 0:
                dev = file_detail['dev_stats']
                dev_speech_pct = (dev.get('speech_hours', 0) / dev['duration_hours']) * 100
                writer.writerow({
                    "Filename": file_detail['filename'], "SplitType": "DEV",
                    "OriginalDuration(h)": f"{file_detail.get('original_hours', 0):.4f}",
                    "OutputDuration(h)": f"{dev['duration_hours']:.4f}",
                    "SpeechDuration(h)": f"{dev.get('speech_hours', 0):.4f}",
                    "SpeechPercentage": f"{dev_speech_pct:.2f}",
                    "SilenceDuration(h)": f"{dev['duration_hours'] - dev.get('speech_hours', 0):.4f}",
                    "SilencePercentage": f"{100 - dev_speech_pct:.2f}",
                    "TotalSegments": dev.get('total_segments', 0),
                    "ContinuousSegments": dev.get('continuous_segments', 0),
                    "Sequences": dev.get('num_sequences', 0),
                    "LongestSequence": dev.get('longest_sequence', 0)
                })

            # Add TRAIN entry if it exists and is not empty
            if 'train_stats' in file_detail and file_detail['train_stats'].get('duration_hours', 0) > 0:
                train = file_detail['train_stats']
                train_speech_pct = (train.get('speech_hours', 0) / train['duration_hours']) * 100
                writer.writerow({
                    "Filename": file_detail['filename'], "SplitType": "TRAIN",
                    "OriginalDuration(h)": f"{file_detail.get('original_hours', 0):.4f}",
                    "OutputDuration(h)": f"{train['duration_hours']:.4f}",
                    "SpeechDuration(h)": f"{train.get('speech_hours', 0):.4f}",
                    "SpeechPercentage": f"{train_speech_pct:.2f}",
                    "SilenceDuration(h)": f"{train['duration_hours'] - train.get('speech_hours', 0):.4f}",
                    "SilencePercentage": f"{100 - train_speech_pct:.2f}",
                    "TotalSegments": train.get('total_segments', 0),
                    "ContinuousSegments": train.get('continuous_segments', 0),
                    "Sequences": train.get('num_sequences', 0),
                    "LongestSequence": train.get('longest_sequence', 0)
                })

        # Add a final summary row
        total_output = batch_stats.get('total_balanced_duration', 0) + \
                       sum(f['dev_stats'].get('duration_hours', 0) for f in batch_stats['file_details'] if 'dev_stats' in f) + \
                       sum(f['train_stats'].get('duration_hours', 0) for f in batch_stats['file_details'] if 'train_stats' in f)
        total_speech = batch_stats.get('total_speech_duration', 0) + \
                       sum(f['dev_stats'].get('speech_hours', 0) for f in batch_stats['file_details'] if 'dev_stats' in f) + \
                       sum(f['train_stats'].get('speech_hours', 0) for f in batch_stats['file_details'] if 'train_stats' in f)
        total_silence = total_output - total_speech
        total_speech_pct = (total_speech / total_output) * 100 if total_output > 0 else 0

        writer.writerow({}) # Blank row for separation
        writer.writerow({
            "Filename": "AGGREGATE_SUMMARY", "SplitType": "ALL",
            "OriginalDuration(h)": f"{batch_stats['total_original_duration']:.4f}",
            "OutputDuration(h)": f"{total_output:.4f}",
            "SpeechDuration(h)": f"{total_speech:.4f}",
            "SpeechPercentage": f"{total_speech_pct:.2f}",
            "SilenceDuration(h)": f"{total_silence:.4f}",
            "SilencePercentage": f"{100 - total_speech_pct:.2f}",
            "TotalSegments": batch_stats['continuity_stats']['total_segments'],
            "ContinuousSegments": batch_stats['continuity_stats']['continuous_segments'],
            "Sequences": batch_stats['continuity_stats']['total_sequences'],
            "LongestSequence": batch_stats['continuity_stats']['longest_sequence']
        })

    print(f"\nBatch processing summary saved to {summary_file}")
    return summary_file


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
        input_dir="input_data",  # Directory with audio/ and ground_truth/ folders
        output_dir="Recompiled_Output",  # Where to save the TEST/TRAIN/DEV outputs
        target_hours=0.135,  # Create x-hour balanced files
        speech_padding_ms=200,  # Add 200ms padding to speech segments
        create_splits=True  # Create TRAIN/DEV splits from unused audio
    )

    if stats:
        print(f"\nProcessed {stats['total_files']} files")
        total_output_duration = stats['total_balanced_duration']
        # Add dev/train durations to total if they exist
        for f in stats['file_details']:
            if 'dev_stats' in f:
                total_output_duration += f['dev_stats'].get('duration_hours', 0)
            if 'train_stats' in f:
                total_output_duration += f['train_stats'].get('duration_hours', 0)
        print(f"Total output generated: {total_output_duration:.2f} hours")