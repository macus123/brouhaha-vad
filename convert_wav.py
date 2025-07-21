import csv
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydub import AudioSegment
from split_seg import read_ground_truth, get_non_speech_segments
from audio_dataclasses import (
    AudioSegmentInfo, TimestampMapping, ProcessingConfig,
    FileProcessingInfo, ProcessingResult, SetStats, BatchStats
)

class AudioProcessor:

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.audio: Optional[AudioSegment] = None
        self.audio_cache: Dict[str, AudioSegment] = {}  # Cache for multi-file processing
        self.sample_rate_cache: Dict[str, int] = {}
    
    def format_duration(self, ms: float) -> str:
        seconds = ms / 1000
        return str(datetime.timedelta(seconds=int(seconds)))
    
    def _add_padding_to_speech(self, speech_segments: List[Tuple[float, float]], total_duration_sec: float) -> List[Tuple[float, float]]:
        padded_segments = []
        padding_sec = self.config.speech_padding_ms / 1000
        
        for start, end in speech_segments:
            padded_start = max(0, start - padding_sec)
            padded_end = min(total_duration_sec, end + padding_sec)
            padded_segments.append((padded_start, padded_end))
        
        return padded_segments
    
    def _merge_overlapping_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping segments."""
        if not segments:
            return []
        
        segments.sort()
        merged = []
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start <= current_end:  # Overlapping
                current_end = max(current_end, end)
            else:  # Non-overlapping
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        return merged
    
    def load_and_analyze_multiple_files(self, file_infos: List[FileProcessingInfo]) -> Tuple[List[AudioSegmentInfo], Dict[str, float]]:
        """Load and analyze multiple audio files for multi-file stitching."""
        all_segments = []
        file_durations = {}
        
        for i, file_info in enumerate(file_infos):
            file_id = f"file_{i}"
            print(f"Loading file {i+1}/{len(file_infos)}: {file_info.audio_path}")
            audio = AudioSegment.from_file(file_info.audio_path)
            self.audio_cache[file_id] = audio
            
            total_duration_ms = len(audio)
            total_duration_sec = total_duration_ms / 1000
            file_durations[file_id] = total_duration_sec

            speech_segments = read_ground_truth(file_info.ground_truth_path)
            padded_speech_segments = self._add_padding_to_speech(speech_segments, total_duration_sec)
            merged_speech_segments = self._merge_overlapping_segments(padded_speech_segments)
            
            for start, end in merged_speech_segments:
                all_segments.append(AudioSegmentInfo(
                    start=start, 
                    end=end, 
                    type="speech", 
                    source_file=file_info.audio_path,
                    file_id=file_id
                ))
            
            # Recalculate non-speech segments based on merged speech segments
            merged_non_speech_segments = get_non_speech_segments(merged_speech_segments, total_duration_sec)
            
            for start, end in merged_non_speech_segments:
                all_segments.append(AudioSegmentInfo(
                    start=start, 
                    end=end, 
                    type="non-speech", 
                    source_file=file_info.audio_path,
                    file_id=file_id
                ))
        
        # Sort segments by file_id and then by start time to maintain temporal order within files
        all_segments.sort(key=lambda x: (x.file_id, x.start))
        
        return all_segments, file_durations
    
    def create_ground_truth_file(self, timestamps: List[TimestampMapping], output_path: str) -> None:
        """Create ground truth file for recompiled audio - only include speech segments."""
        with open(output_path, 'w') as f:
            # Filter for speech segments only
            speech_timestamps = [ts for ts in timestamps if ts.type == "speech"]
            
            if speech_timestamps:
                # Write only speech segments
                for ts in speech_timestamps:
                    f.write(f"{ts.output_start_sec:.3f}\t{ts.output_end_sec:.3f}\tspeech\n")
            else:
                # No speech segments - write entire file duration with filename
                if timestamps:
                    # Get total duration from the last timestamp
                    total_duration = timestamps[-1].output_end_sec
                    wav_filename = Path(output_path).stem + ".wav"
                    f.write(f"0.000\t{total_duration:.3f}\t{wav_filename}\n")
                else:
                    # Fallback: empty file case
                    wav_filename = Path(output_path).stem + ".wav"
                    f.write(f"0.000\t0.000\t{wav_filename}\n")
    
    def compile_multi_source_audio(self, segments: List[AudioSegmentInfo]) -> Tuple[AudioSegment, List[TimestampMapping]]:
        compiled_audio = AudioSegment.empty()
        timestamp_map = []
        output_time_ms = 0
        
        for i, segment in enumerate(segments):
            # Get source audio from cache
            source_audio = self.audio_cache[segment.file_id]
            
            # Extract segment
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            
            # Handle edge case where end might be slightly past file length
            if end_ms > len(source_audio):
                end_ms = len(source_audio)
            
            # Extract the actual audio
            segment_audio = source_audio[start_ms:end_ms]
            segment_duration_ms = len(segment_audio)
            
            # Create timestamp mapping using ACTUAL durations
            timestamp_map.append(TimestampMapping(
                segment_index=i,
                original_start_sec=segment.start,
                original_end_sec=segment.end,
                output_start_sec=output_time_ms / 1000,
                output_end_sec=(output_time_ms + segment_duration_ms) / 1000,
                duration_sec=segment_duration_ms / 1000,
                type=segment.type,
                source_file=segment.source_file,
                file_id=segment.file_id
            ))
            
            # Append to compiled audio
            compiled_audio += segment_audio
            output_time_ms += segment_duration_ms
        
        final_audio_duration_sec = len(compiled_audio) / 1000
        if timestamp_map and abs(timestamp_map[-1].output_end_sec - final_audio_duration_sec) > 0.001:
            # Adjust the last timestamp to match exact audio length
            print(f"  Adjusting final timestamp: {timestamp_map[-1].output_end_sec:.3f}s → {final_audio_duration_sec:.3f}s")
            timestamp_map[-1].output_end_sec = final_audio_duration_sec
            timestamp_map[-1].duration_sec = timestamp_map[-1].output_end_sec - timestamp_map[-1].output_start_sec
        
        return compiled_audio, timestamp_map
    
    def group_segments_into_temporal_sequences(self, segments: List[AudioSegmentInfo]) -> Dict[str, List[List[AudioSegmentInfo]]]:
        # Group segments by file
        file_segments = {}
        for segment in segments:
            if segment.file_id not in file_segments:
                file_segments[segment.file_id] = []
            file_segments[segment.file_id].append(segment)
        
        file_sequences = {}
        
        # Process each file
        for file_id, file_segs in file_segments.items():
            # Sort by start time
            file_segs.sort(key=lambda x: x.start)
            sequences = []
            current_sequence = []
            current_duration = 0
            has_speech = False
            has_silence = False
            
            min_sequence_length = 3.0  # minimum target length: seconds
            
            max_gap = 0.5  # Maximum gap to consider segments as part of same sequence
            
            for segment in file_segs:
                if not current_sequence:
                    # Start new sequence
                    current_sequence.append(segment)
                    current_duration = segment.duration / 1000  # Convert ms to seconds
                    has_speech = segment.type == "speech"
                    has_silence = segment.type == "non-speech"
                else:
                    # Check if this segment is temporally continuous with the previous one
                    prev_segment = current_sequence[-1]
                    gap = segment.start - prev_segment.end
                    
                    # Always accept if gap is small enough
                    if gap <= max_gap:
                        current_sequence.append(segment)
                        current_duration += segment.duration / 1000
                        has_speech = has_speech or segment.type == "speech"
                        has_silence = has_silence or segment.type == "non-speech"
                    else:
                        # If sequence is long enough and has both speech and silence, finalize it
                        if current_duration >= min_sequence_length and has_speech and has_silence:
                            sequences.append(current_sequence)
                            current_sequence = [segment]
                            current_duration = segment.duration / 1000
                            has_speech = segment.type == "speech"
                            has_silence = segment.type == "non-speech"
                        # If sequence doesn't meet criteria, try to extend it despite the gap
                        elif current_duration < min_sequence_length or not (has_speech and has_silence):
                            # If adding this segment would create a balanced sequence, include it
                            if (has_speech and segment.type == "non-speech") or (has_silence and segment.type == "speech"):
                                current_sequence.append(segment)
                                current_duration += segment.duration / 1000
                                has_speech = has_speech or segment.type == "speech"
                                has_silence = has_silence or segment.type == "non-speech"
                            # If gap is too large (> 2 seconds) or sequence is getting too long, finalize it anyway
                            elif gap > 2.0 or current_duration > min_sequence_length * 2:
                                sequences.append(current_sequence)
                                current_sequence = [segment]
                                current_duration = segment.duration / 1000
                                has_speech = segment.type == "speech"
                                has_silence = segment.type == "non-speech"
            
            # Add the last sequence if it exists
            if current_sequence:
                sequences.append(current_sequence)
            
            # Perform a final pass to merge very short sequences with neighbors
            if len(sequences) > 1:
                merged_sequences = []
                current_merged = sequences[0]
                
                for i in range(1, len(sequences)):
                    current_seq = current_merged
                    next_seq = sequences[i]
                    
                    # If current sequence is too short and adding next wouldn't make it too long
                    current_length = sum(s.duration / 1000 for s in current_seq)
                    next_length = sum(s.duration / 1000 for s in next_seq)
                    
                    if current_length < min_sequence_length and current_length + next_length < min_sequence_length * 3:
                        # Merge with next sequence
                        current_merged = current_seq + next_seq
                    else:
                        merged_sequences.append(current_merged)
                        current_merged = next_seq
                
                # Add the last merged sequence
                merged_sequences.append(current_merged)
                sequences = merged_sequences
            
            file_sequences[file_id] = sequences
        
        return file_sequences 
    
    def save_temporal_sequences(self, segments: List[AudioSegmentInfo], split_name: str, output_dir: str) -> List[str]:
        """Save temporal sequences and return paths to created files with accurate statistics tracking."""
        sequences_by_file = self.group_segments_into_temporal_sequences(segments)
        output_files = []
        
        for file_id, sequences in sequences_by_file.items():
            for seq_idx, sequence in enumerate(sequences):
                if not sequence:
                    continue
                    
                # Get source file from first segment
                source_file = sequence[0].source_file
                original_filename = Path(source_file).stem

                output_subdir = Path(output_dir)
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                seq_num = f"{seq_idx+1:03d}"
                audio_filename = f"{split_name}_{original_filename}_{seq_num}.wav"
                gt_filename = f"{split_name}_{original_filename}_{seq_num}.txt"
                
                output_path = output_subdir / audio_filename
                gt_path = output_subdir / gt_filename
                
                # Create sequence audio file
                compiled_audio, sequence_timestamps = self.compile_multi_source_audio(sequence)
                
                total_ms = len(compiled_audio)
                speech_ms = sum(ts.duration_sec * 1000 for ts in sequence_timestamps if ts.type == "speech")
                silence_ms = total_ms - speech_ms
                speech_percent = (speech_ms / total_ms) * 100 if total_ms > 0 else 0
                
                # Save audio and ground truth
                self.save_audio_with_cached_sample_rate(compiled_audio, output_path, source_file)
                self.create_ground_truth_file(sequence_timestamps, str(gt_path))
                
                # Store comprehensive sequence stats
                sequence_stat = {
                    "filename": str(output_path),
                    "display_filename": audio_filename,
                    "split": split_name,
                    "source_file": source_file,
                    "source_filename": original_filename,
                    "sequence_number": seq_num,
                    "total_duration_sec": total_ms / 1000,
                    "total_duration_ms": total_ms,
                    "speech_duration_sec": speech_ms / 1000,
                    "speech_duration_ms": speech_ms,
                    "silence_duration_sec": silence_ms / 1000,
                    "silence_duration_ms": silence_ms,
                    "speech_percent": speech_percent,
                    "file_id": file_id
                }
                
                # Store sequence statistics for later CSV generation
                if not hasattr(self, 'sequence_stats'):
                    self.sequence_stats = []
                self.sequence_stats.append(sequence_stat)
                
                print(f"  Created {audio_filename}: {self.format_duration(total_ms)} "
                    f"(speech: {self.format_duration(speech_ms)}, "
                    f"non-speech: {self.format_duration(silence_ms)}, "
                    f"{speech_percent:.1f}% speech)")
                    
                output_files.append(str(output_path))
        
        return output_files

    def _truncate_sequence_to_quota(self, sequence: List[AudioSegmentInfo], remaining_quota_ms: float, protect_speech: bool = False) -> List[AudioSegmentInfo]:
        truncated_sequence = []
        used_ms = 0
        
        for segment in sequence:
            if used_ms + segment.duration <= remaining_quota_ms:
                # Whole segment fits
                truncated_sequence.append(segment)
                used_ms += segment.duration
            else:
                # Segment doesn't fit completely
                if protect_speech and segment.type == "speech":
                    # NEVER truncate speech when protection is enabled
                    break
                else:
                    # Non-speech can be truncated (or speech if protection disabled)
                    remaining_ms = remaining_quota_ms - used_ms
                    if remaining_ms > 100:  # Only include if at least 100ms remains
                        # Create truncated segment
                        truncated_segment = AudioSegmentInfo(
                            start=segment.start,
                            end=segment.start + (remaining_ms / 1000),
                            type=segment.type,
                            source_file=segment.source_file,
                            file_id=segment.file_id
                        )
                        truncated_sequence.append(truncated_segment)
                break
        
        return truncated_sequence

    def create_balanced_timeline_multi_file(self, segments: List[AudioSegmentInfo]) -> List[AudioSegmentInfo]:
        speech_target_ms = int(self.config.target_hours_speech * 3600 * 1000)
        silence_target_ms = int(self.config.target_hours_silence * 3600 * 1000)
        
        # Separate speech and silence segments
        speech_segments, silence_segments = self._separate_segments_by_type(segments)
        
        # Calculate totals
        total_speech_ms, total_silence_ms = self._calculate_totals(segments)
        
        print(f"Multi-file speech content: {self.format_duration(total_speech_ms)}")
        print(f"Multi-file non-speech content: {self.format_duration(total_silence_ms)}")
        
        # Check if we have enough content and adjust if needed
        if total_speech_ms < speech_target_ms:
            print(f"Warning: Insufficient speech audio to reach target")
            print(f"  Available: {self.format_duration(total_speech_ms)}")
            print(f"  Target: {self.format_duration(speech_target_ms)}")
            speech_target_ms = total_speech_ms
            
        if total_silence_ms < silence_target_ms:
            print(f"Warning: Insufficient silence audio to reach target")
            print(f"  Available: {self.format_duration(total_silence_ms)}")
            print(f"  Target: {self.format_duration(silence_target_ms)}")
            silence_target_ms = total_silence_ms
        
        # Use direct targets for distribution algorithm
        return self._distribute_silence_intelligently_multi_file(speech_segments, silence_segments, speech_target_ms, silence_target_ms)


    def _create_natural_timeline(self, speech_segments: List[AudioSegmentInfo],
                            silence_segments: List[AudioSegmentInfo],
                            speech_quota: float, silence_quota: float,
                            reserved_silence: List[AudioSegmentInfo]) -> List[AudioSegmentInfo]:
        # Group segments by file_id to maintain file coherence
        segments_by_file = {}
        for segment in speech_segments + silence_segments:
            if segment.file_id not in segments_by_file:
                segments_by_file[segment.file_id] = []
            segments_by_file[segment.file_id].append(segment)
        
        # Sort all segments by start time within each file
        for file_id in segments_by_file:
            segments_by_file[file_id].sort(key=lambda x: x.start)
        
        # Calculate quota per file based on available content
        file_speech_content = {}
        file_silence_content = {}
        for file_id, segments in segments_by_file.items():
            file_speech_content[file_id] = sum(s.duration for s in segments if s.type == "speech")
            file_silence_content[file_id] = sum(s.duration for s in segments if s.type == "non-speech")
        
        total_speech = sum(file_speech_content.values())
        total_silence = sum(file_silence_content.values())
        
        # Assign quota proportionally to each file
        file_speech_quota = {}
        file_silence_quota = {}
        for file_id in segments_by_file:
            if total_speech > 0:
                file_speech_quota[file_id] = speech_quota * (file_speech_content[file_id] / total_speech)
            else:
                file_speech_quota[file_id] = 0
                
            if total_silence > 0:
                file_silence_quota[file_id] = silence_quota * (file_silence_content[file_id] / total_silence)
            else:
                file_silence_quota[file_id] = 0
        
        # Select segments from each file respecting temporal order
        timeline = []
        for file_id, segments in segments_by_file.items():
            file_timeline = []
            speech_used = 0
            silence_used = 0
            
            for segment in segments:
                if segment.type == "speech" and speech_used < file_speech_quota[file_id]:
                    if speech_used + segment.duration <= file_speech_quota[file_id]:
                        file_timeline.append(segment)
                        speech_used += segment.duration
                    # Don't truncate speech segments - prefer to skip
                    
                elif segment.type == "non-speech" and silence_used < file_silence_quota[file_id]:
                    if silence_used + segment.duration <= file_silence_quota[file_id]:
                        file_timeline.append(segment)
                        silence_used += segment.duration
                    else:
                        # Truncate silence segments if needed
                        remaining = file_silence_quota[file_id] - silence_used
                        if remaining > 100:  # Only add if at least 100ms
                            partial = AudioSegmentInfo(
                                start=segment.start,
                                end=segment.start + (remaining / 1000),
                                type=segment.type,
                                source_file=segment.source_file,
                                file_id=segment.file_id
                            )
                            file_timeline.append(partial)
                            silence_used = file_silence_quota[file_id]
            
            timeline.extend(file_timeline)
        
        # Intersperse reserved silence in long speech runs
        final_segments = self._intersperse_reserved_silence(timeline, reserved_silence)
        
        return final_segments
    
    def _distribute_silence_intelligently_multi_file(self, speech_segments: List[AudioSegmentInfo], 
                                                silence_segments: List[AudioSegmentInfo], 
                                                speech_target_ms: float, silence_target_ms: float) -> List[AudioSegmentInfo]:
        """Multi-file version of sophisticated silence distribution with improved temporal representation."""
        # Reserve silence for interspersing (same algorithm)
        reserved_silence_ms = silence_target_ms * self.config.silence_reserve_ratio
        primary_silence_ms = silence_target_ms - reserved_silence_ms
        
        # Sort segments by their original temporal position
        speech_segments.sort(key=lambda x: (x.file_id, x.start))
        
        # CHANGE: Sort silence segments by file and position instead of duration
        silence_segments.sort(key=lambda x: (x.file_id, x.start))
        
        # Create a copy for reservation (we'll still need some silence for interspersing)
        silence_for_reservation = sorted(silence_segments.copy(), key=lambda x: x.duration)
        
        # Reserve short silence segments for interspersing
        reserved_silence = []
        reserved_silence_duration = 0
        
        # Take some percentage of shortest silence segments for reservation
        reservation_count = max(1, int(len(silence_for_reservation) * 0.3))
        for i in range(min(reservation_count, len(silence_for_reservation))):
            segment = silence_for_reservation[i]
            if reserved_silence_duration < reserved_silence_ms:
                reserved_silence.append(segment)
                reserved_silence_duration += segment.duration
        
        # Build primary timeline with remaining silence
        reserved_ids = set(id(seg) for seg in reserved_silence)
        primary_candidates = [seg for seg in silence_segments if id(seg) not in reserved_ids]
        
        # Maintain temporal ordering when selecting primary silence
        primary_silence = sorted(primary_candidates, key=lambda x: (x.file_id, x.start))
        
        # Create a more natural timeline that maintains temporal proximity
        return self._create_natural_timeline(speech_segments, primary_silence, 
                                            speech_target_ms, primary_silence_ms,
                                            reserved_silence)    
    
    def _intersperse_reserved_silence(self, balanced_segments: List[AudioSegmentInfo], 
                                    reserved_silence: List[AudioSegmentInfo]) -> List[AudioSegmentInfo]:
        """Intersperse reserved silence in long speech runs."""
        # Find long speech runs
        speech_runs = self._find_speech_runs(balanced_segments)
        
        # Sort runs by length (longest first)
        speech_runs.sort(key=lambda x: x[1] - x[0], reverse=True)
        
        new_segments = balanced_segments.copy()
        reserved_index = 0
        inserted = 0
        
        for run_start, run_end in speech_runs:
            run_length = run_end - run_start + 1
            num_to_insert = run_length // 3
            
            if num_to_insert > 0 and reserved_index < len(reserved_silence):
                # Calculate insertion positions
                positions = []
                for i in range(1, num_to_insert + 1):
                    pos = run_start + (i * run_length) // (num_to_insert + 1)
                    positions.append(pos + inserted)
                
                # Insert silence segments
                for pos in positions:
                    if reserved_index < len(reserved_silence):
                        new_segments.insert(pos, reserved_silence[reserved_index])
                        reserved_index += 1
                        inserted += 1
        
        # Add any leftover reserved silence at the end
        if reserved_index < len(reserved_silence):
            new_segments.extend(reserved_silence[reserved_index:])
        
        return new_segments
    
    def _find_speech_runs(self, segments: List[AudioSegmentInfo]) -> List[Tuple[int, int]]:
        """Find consecutive speech segments (runs)."""
        speech_runs = []
        current_run_start = None
        current_run_end = None
        
        for i, segment in enumerate(segments):
            if segment.type == "speech":
                if current_run_start is None:
                    current_run_start = i
                    current_run_end = i
                else:
                    current_run_end = i
            else:
                if current_run_start is not None:
                    run_length = current_run_end - current_run_start + 1
                    if run_length >= 3:
                        speech_runs.append((current_run_start, current_run_end))
                    current_run_start = None
        
        # Add the last run if it exists
        if current_run_start is not None:
            run_length = current_run_end - current_run_start + 1
            if run_length >= 3:
                speech_runs.append((current_run_start, current_run_end))
        
        return speech_runs

    def process_unified_output(self, input_files: List[FileProcessingInfo], 
                             test_dir: str, dev_dir: str, train_dir: str) -> ProcessingResult:

        all_segments, file_durations = self.load_and_analyze_multiple_files(input_files)
        total_speech_ms, total_silence_ms = self._calculate_totals(all_segments)
        total_duration_sec = sum(file_durations.values())
        
        print(f"Total content available: {self.format_duration((total_speech_ms + total_silence_ms))}")
        print(f"  Speech: {self.format_duration(total_speech_ms)}")
        print(f"  Silence: {self.format_duration(total_silence_ms)}")
        
        # Create TEST set using target duration and ratio
        test_segments = self.create_balanced_timeline_multi_file(all_segments)
        
        # Calculate TEST statistics for reporting
        test_speech_ms, test_non_speech_ms = self._calculate_totals(test_segments)
        
        # Save TEST set as temporal sequences
        print(f"Creating TEST temporal sequences...")
        test_output_files = self.save_temporal_sequences(test_segments, "TEST", test_dir)
        
        print(f"TEST set created: {self.format_duration(test_speech_ms + test_non_speech_ms)} "
              f"across {len(test_output_files)//2} sequences")

        result = ProcessingResult(
            test_output=test_dir,  # Directory instead of single file
            test_ground_truth=test_dir,  # Directory instead of single file
            original_duration_hours=total_duration_sec / 3600,
            test_audio=None,  # No single audio file
            test_timestamps=None,  # No single timestamp file
            test_speech_ms=test_speech_ms,
            test_non_speech_ms=test_non_speech_ms,
            files_used=[file_info.audio_path for file_info in input_files]
        )
        
        if self.config.create_splits:
            self._create_temporal_dev_train_splits(all_segments, test_segments, 
                                                 dev_dir, train_dir, result)
        
        return result
    
    def _create_temporal_dev_train_splits(self, all_segments: List[AudioSegmentInfo], 
                                         test_segments: List[AudioSegmentInfo],
                                         dev_dir: str, train_dir: str, 
                                         result: ProcessingResult) -> None:
        used_segments = set(id(segment) for segment in test_segments)
        remaining_segments = [seg for seg in all_segments if id(seg) not in used_segments]
        
        # Sort by file_id and then by start time to maintain temporal order within files
        remaining_segments.sort(key=lambda x: (x.file_id, x.start))
        
        if not remaining_segments:
            print("No remaining content for DEV/TRAIN splits")
            return
        
        # Split remaining segments into DEV and TRAIN
        dev_segments, train_segments = self._split_remaining_segments(remaining_segments)
        
        # Create DEV set
        if dev_segments:
            # Apply the same balanced timeline algorithm to DEV set
            dev_balanced_segments = self._create_balanced_subset(dev_segments, "DEV")
            
            print(f"Creating DEV temporal sequences...")
            dev_output_files = self.save_temporal_sequences(dev_balanced_segments, "DEV", dev_dir)
            
            dev_speech_ms = sum(s.duration for s in dev_balanced_segments if s.type == "speech")
            dev_non_speech_ms = sum(s.duration for s in dev_balanced_segments if s.type == "non-speech")
            
            result.dev_output = dev_dir
            result.dev_audio = None
            result.dev_timestamps = None
            result.dev_ground_truth = dev_dir
            
            print(f"DEV set created: {self.format_duration(dev_speech_ms + dev_non_speech_ms)} "
                  f"across {len(dev_output_files)//2} sequences")
        
        # Create TRAIN set
        if train_segments:
            # Apply the same balanced timeline algorithm to TRAIN set
            train_balanced_segments = self._create_balanced_subset(train_segments, "TRAIN")
            
            print(f"Creating TRAIN temporal sequences...")
            train_output_files = self.save_temporal_sequences(train_balanced_segments, "TRAIN", train_dir)
            
            train_speech_ms = sum(s.duration for s in train_balanced_segments if s.type == "speech")
            train_non_speech_ms = sum(s.duration for s in train_balanced_segments if s.type == "non-speech")
            
            result.train_output = train_dir
            result.train_audio = None
            result.train_timestamps = None
            result.train_ground_truth = train_dir
            
            print(f"TRAIN set created: {self.format_duration(train_speech_ms + train_non_speech_ms)} "
                  f"across {len(train_output_files)//2} sequences")
    
    def _create_balanced_subset(self, segments: List[AudioSegmentInfo], set_name: str) -> List[AudioSegmentInfo]:
        speech_segments, silence_segments = self._separate_segments_by_type(segments)
        
        total_speech_ms, total_silence_ms = self._calculate_totals(segments)
        
        # Check if we have any content at all
        if total_speech_ms == 0 or total_silence_ms == 0:
            print(f"  Warning: {set_name} set has insufficient content (speech: {self.format_duration(total_speech_ms)}, silence: {self.format_duration(total_silence_ms)})")
            return segments
        
        # For DEV/TRAIN, maintain similar ratio as original targets but use available content
        total_target_ms = (self.config.target_hours_speech + self.config.target_hours_silence) * 3600 * 1000
        if total_target_ms > 0:
            target_ratio = self.config.target_hours_speech / (self.config.target_hours_speech + self.config.target_hours_silence)
        else:
            target_ratio = 0.5  # Default to 50/50 if no targets specified
        
        # Calculate available content and proportional targets
        total_available_ms = total_speech_ms + total_silence_ms
        target_speech_ms = total_available_ms * target_ratio
        target_silence_ms = total_available_ms * (1 - target_ratio)
        
        if total_speech_ms < target_speech_ms:
            target_speech_ms = total_speech_ms
            target_silence_ms = min(total_silence_ms, total_available_ms - target_speech_ms)
        
        if total_silence_ms < target_silence_ms:
            target_silence_ms = total_silence_ms
            target_speech_ms = min(total_speech_ms, total_available_ms - target_silence_ms)
        
        print(f"  {set_name} targets - Speech: {self.format_duration(target_speech_ms)}, Silence: {self.format_duration(target_silence_ms)}")
        return self._distribute_silence_intelligently_multi_file(
            speech_segments, silence_segments, target_speech_ms, target_silence_ms
        )
        
    # def _split_remaining_segments(self, remaining_segments: List[AudioSegmentInfo]) -> Tuple[List[AudioSegmentInfo], List[AudioSegmentInfo]]:
    #     """Split remaining segments into DEV and TRAIN sets."""
    #     if not remaining_segments:
    #         return [], []
        
    #     # Calculate split point based on dev_ratio
    #     total_segments = len(remaining_segments)
    #     dev_count = int(total_segments * self.config.dev_ratio)
        
    #     # Ensure we have at least some segments for each set if possible
    #     if dev_count == 0 and total_segments > 1:
    #         dev_count = 1
    #     elif dev_count == total_segments and total_segments > 1:
    #         dev_count = total_segments - 1
        
    #     # Split the segments
    #     dev_segments = remaining_segments[:dev_count]
    #     train_segments = remaining_segments[dev_count:]
        
    #     print(f"    DEV: {len(dev_segments)} segments")
    #     print(f"    TRAIN: {len(train_segments)} segments")
        
    #     return dev_segments, train_segments
    
    def _split_remaining_segments(self, remaining_segments: List[AudioSegmentInfo]) -> Tuple[List[AudioSegmentInfo], List[AudioSegmentInfo]]:
        speech_segments = [seg for seg in remaining_segments if seg.type == "speech"]
        
        if speech_segments:
            total_speech_ms = sum(seg.duration for seg in speech_segments)
            
            for i, seg in enumerate(speech_segments[:5]):  # Show first 5 for brevity
                source_file = Path(seg.source_file).name
                print(f"  - Speech segment #{i+1}: {source_file} {seg.start:.2f}s-{seg.end:.2f}s ({seg.duration/1000:.2f}s)")
            if len(speech_segments) > 5:
                print(f"  - ...and {len(speech_segments)-5} more speech segments")
        
        # Continue with normal split (or optionally move speech segments back to TEST)
        non_speech_segments = [seg for seg in remaining_segments if seg.type == "non-speech"]
        
        # Calculate split point based on dev_ratio (using only non-speech segments)
        total_segments = len(non_speech_segments)
        dev_count = int(total_segments * self.config.dev_ratio)
        
        if dev_count == 0 and total_segments > 1:
            dev_count = 1
        elif dev_count == total_segments and total_segments > 1:
            dev_count = total_segments - 1
        
        dev_segments = non_speech_segments[:dev_count]
        train_segments = non_speech_segments[dev_count:]
        
        print(f"    DEV: {len(dev_segments)} segments (non-speech only)")
        print(f"    TRAIN: {len(train_segments)} segments (non-speech only)")
        print(f"    Unallocated speech: {len(speech_segments)} segments")
        
        return dev_segments, train_segments

    def save_audio_with_cached_sample_rate(self, audio: AudioSegment, output_path: Path, source_file_path: str) -> None:
        # Use cached sample rate or get it once
        if source_file_path not in self.sample_rate_cache:
            self.sample_rate_cache[source_file_path] = self.get_sample_rate_from_file(source_file_path)
        
        sample_rate = self.sample_rate_cache[source_file_path]
        
        # Direct export without temporary file when possible
        audio.export(str(output_path), format="wav", parameters=["-ar", str(sample_rate)])

    def get_sample_rate_from_file(self, file_path: str) -> int:
        """Get sample rate directly from an audio file."""
        try:
            audio = AudioSegment.from_file(file_path)
            return audio.frame_rate
        except Exception as e:
            print(f"Warning: Could not read sample rate from {file_path}: {e}")
            return self.config.default_sample_rate
    
    def _calculate_totals(self, data: List) -> Tuple[float, float]:
        """Calculate total speech and non-speech durations from segments or timestamps."""
        if not data:
            return 0.0, 0.0
        
        # Handle AudioSegmentInfo objects
        if hasattr(data[0], 'duration'):
            speech_ms = sum(item.duration for item in data if item.type == "speech")
            non_speech_ms = sum(item.duration for item in data if item.type == "non-speech")
        # Handle TimestampMapping objects
        elif hasattr(data[0], 'duration_sec'):
            speech_ms = sum(item.duration_sec * 1000 for item in data if item.type == "speech")
            non_speech_ms = sum(item.duration_sec * 1000 for item in data if item.type == "non-speech")
        else:
            raise ValueError("Unsupported data type for calculation")
        
        return speech_ms, non_speech_ms

    def _separate_segments_by_type(self, segments: List[AudioSegmentInfo]) -> Tuple[List[AudioSegmentInfo], List[AudioSegmentInfo]]:
        """Separate segments into speech and non-speech lists."""
        speech_segments = [s for s in segments if s.type == "speech"]
        silence_segments = [s for s in segments if s.type == "non-speech"]
        return speech_segments, silence_segments

class DirectoryScanner:
    """Handles scanning of input directories for audio files."""
    
    @staticmethod
    def scan_input_directory(input_dir: str = "input_data") -> List[FileProcessingInfo]:
        """Scan input_data directory structure to find audio files and match with ground truth."""
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
            file_stem = audio_file.stem
            gt_file = ground_truth_dir / f"{file_stem}.txt"
            
            if gt_file.exists():
                result.append(FileProcessingInfo(
                    audio_path=str(audio_file),
                    ground_truth_path=str(gt_file),
                    set_type="TEST"
                ))
            else:
                print(f"Warning: No ground truth found for {audio_file}")
        
        print(f"Matched {len(result)} files with ground truth")
        return result


class BatchProcessor:
    """Handles batch processing of multiple audio files with comprehensive statistics."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processor = AudioProcessor(config)
        self.batch_stats = None
    
    def process_batch(self, input_files: List[FileProcessingInfo], output_dir: str) -> BatchStats:
        """Process multiple audio files with progress display."""
        # Create output directories
        base_output_dir = Path(output_dir)
        test_dir = str(base_output_dir / "TEST")
        dev_dir = str(base_output_dir / "DEV")
        train_dir = str(base_output_dir / "TRAIN")
        
        for dir_path in [test_dir, dev_dir, train_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {len(input_files)} files into temporal sequence TEST/DEV/TRAIN sets...")

        self.batch_stats = BatchStats()
        
        try:
            self.processor.batch_processor = self
            
            result = self.processor.process_unified_output(
                input_files, str(test_dir), str(dev_dir), str(train_dir)
            )
            
            self._update_temporal_batch_stats(self.batch_stats, result, input_files)
            
            print(f"  ✓ Processing Complete")
            
        except Exception as e:
            print(f"  ✗ Error in processing: {e}")
            error_entry = {
                'error': str(e),
                'files_processed': len(input_files)
            }
            self.batch_stats.test_stats.file_details.append(error_entry)
        
        # Create CSV summary with accurate statistics
        self._create_summary_csv(self.batch_stats, output_dir)
        
        return self.batch_stats
    
    def _update_temporal_batch_stats(self, batch_stats: BatchStats, 
                                    result: ProcessingResult, 
                                    input_files: List[FileProcessingInfo]) -> None:
        """Update batch statistics for temporal sequence processing."""
        # Update original duration (sum of all input files)
        batch_stats.total_original_duration = result.original_duration_hours
        
        # Process TEST set
        self._process_temporal_set_stats(
            batch_stats.test_stats,
            "TEST",
            result.test_output,  # Directory path
            result.test_speech_ms,
            result.test_non_speech_ms,
            input_files,
            result.original_duration_hours
        )
        
        # Process DEV set if available
        if result.dev_output:
            # Calculate DEV statistics from directory
            dev_speech_ms, dev_non_speech_ms = self._calculate_set_durations_from_dir(result.dev_output)
            
            self._process_temporal_set_stats(
                batch_stats.dev_stats,
                "DEV",
                result.dev_output,
                dev_speech_ms,
                dev_non_speech_ms,
                input_files,
                result.original_duration_hours
            )
        
        # Process TRAIN set if available
        if result.train_output:
            # Calculate TRAIN statistics from directory
            train_speech_ms, train_non_speech_ms = self._calculate_set_durations_from_dir(result.train_output)
            
            self._process_temporal_set_stats(
                batch_stats.train_stats,
                "TRAIN",
                result.train_output,
                train_speech_ms,
                train_non_speech_ms,
                input_files,
                result.original_duration_hours
            )
    
    def _calculate_set_durations_from_dir(self, output_dir: str) -> tuple[float, float]:
        """Calculate speech and non-speech durations from ground truth files in directory."""
        speech_ms = 0.0
        non_speech_ms = 0.0
        
        try:
            # Find all .txt files in the directory
            output_path = Path(output_dir)
            for gt_file in output_path.glob("*.txt"):
                with open(gt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and '\t' in line:
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                start_time = float(parts[0])
                                end_time = float(parts[1])
                                label = parts[2]
                                duration_ms = (end_time - start_time) * 1000
                                
                                if label == "speech":
                                    speech_ms += duration_ms
                                elif label == "non-speech":
                                    non_speech_ms += duration_ms
        except Exception as e:
            print(f"Warning: Could not calculate durations from {output_dir}: {e}")
        
        return speech_ms, non_speech_ms
    
    def _process_temporal_set_stats(self, set_stats: SetStats, 
                                set_type: str,
                                output_dir: str,
                                speech_ms: float,
                                non_speech_ms: float,
                                input_files: List[FileProcessingInfo],
                                original_duration_hours: float) -> None:
        """Process statistics for temporal sequence processing."""
        total_ms = speech_ms + non_speech_ms
        duration_hours = total_ms / 3600000
        speech_hours = speech_ms / 3600000
        non_speech_hours = non_speech_ms / 3600000
        
        # Update set totals
        set_stats.total_duration += duration_hours
        set_stats.speech_duration += speech_hours
        set_stats.non_speech_duration += non_speech_hours
        
        # Calculate ratio accuracy (only for TEST set)
        if set_type == "TEST" and duration_hours > 0:
            speech_ratio = speech_hours / duration_hours
            
            # Calculate target ratio from direct targets
            total_target_hours = self.config.target_hours_speech + self.config.target_hours_silence
            if total_target_hours > 0:
                target_ratio = self.config.target_hours_speech / total_target_hours
            else:
                target_ratio = 0.5  # Default fallback
            
            ratio_accuracy = 1 - abs(speech_ratio - target_ratio)
            self.batch_stats.speech_ratio_accuracy.append(ratio_accuracy)
        
        # Collect individual file details from the output directory
        try:
            self._collect_individual_file_details(set_stats, set_type, output_dir, input_files)
        except Exception as e:
            print(f"Warning: Could not collect individual file details from {output_dir}: {e}")
            # Fallback to summary entry
            sequence_filename = f"temporal_sequences_{set_type.lower()}_from_{len(input_files)}_files"
            speech_ratio_val = speech_hours / duration_hours if duration_hours > 0 else 0
            
            file_detail = {
                "filename": sequence_filename,
                "set_type": set_type,
                "output_duration_hours": duration_hours,
                "speech_duration_hours": speech_hours,
                "non_speech_duration_hours": non_speech_hours,
                "speech_ratio": speech_ratio_val,
                "original_duration_hours": original_duration_hours,
                "reduction_factor": original_duration_hours / duration_hours if duration_hours > 0 else 0,
                "num_sequences": 1,
                "output_directory": output_dir,
                "original_file": "multiple_files"
            }
            set_stats.file_details.append(file_detail)
    
    def _collect_individual_file_details(self, set_stats: SetStats, set_type: str, 
                                        output_dir: str, input_files: List[FileProcessingInfo]) -> None:
        """Collect details for each individual output file."""
        output_path = Path(output_dir)
        
        # Create mapping from original filenames to their full paths
        original_file_map = {}
        for file_info in input_files:
            original_name = Path(file_info.audio_path).stem
            original_file_map[original_name] = file_info.audio_path
        
        # Process each audio file in the output directory
        for audio_file in output_path.glob("*.wav"):
            try:
                # Parse filename: {SPLIT}_{original_filename}_{sequence_number}.wav
                filename_parts = audio_file.stem.split('_')
                if len(filename_parts) >= 3:
                    split_name = filename_parts[0]
                    sequence_num = filename_parts[-1]
                    original_filename = '_'.join(filename_parts[1:-1])  # Handle filenames with underscores
                else:
                    continue  # Skip malformed filenames
                
                # Find corresponding ground truth file
                gt_file = output_path / f"{audio_file.stem}.txt"
                if not gt_file.exists():
                    continue
                
                # Calculate durations from ground truth file
                speech_duration_sec = 0.0
                non_speech_duration_sec = 0.0
                
                with open(gt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            start_sec = float(parts[0])
                            end_sec = float(parts[1])
                            segment_type = parts[2]
                            duration = end_sec - start_sec
                            
                            if segment_type == "speech":
                                speech_duration_sec += duration
                            else:
                                non_speech_duration_sec += duration
                
                total_duration_sec = speech_duration_sec + non_speech_duration_sec
                speech_hours = speech_duration_sec / 3600
                non_speech_hours = non_speech_duration_sec / 3600
                total_hours = total_duration_sec / 3600
                
                speech_percentage = (speech_hours / total_hours) * 100 if total_hours > 0 else 0
                
                # Get original file path
                original_file_path = original_file_map.get(original_filename, f"Unknown ({original_filename})")
                
                file_detail = {
                    "filename": audio_file.name,
                    "set_type": set_type,
                    "output_duration_hours": total_hours,
                    "speech_duration_hours": speech_hours,
                    "non_speech_duration_hours": non_speech_hours,
                    "speech_ratio": speech_percentage / 100 if total_hours > 0 else 0,
                    "original_duration_hours": 0,  # Will be filled later if needed
                    "reduction_factor": 0,  # Will be calculated later if needed
                    "num_sequences": 1,  # Each entry represents one sequence
                    "output_directory": output_dir,
                    "original_file": original_file_path,
                    "sequence_number": sequence_num
                }
                set_stats.file_details.append(file_detail)
                
            except Exception as e:
                print(f"Warning: Could not process file {audio_file}: {e}")
                continue
    
    def _create_summary_csv(self, batch_stats: BatchStats, output_dir: str) -> None:
        """Create a comprehensive CSV summary using sequence statistics."""
        import csv
        from datetime import datetime
        
        output_path = Path(output_dir)
        summary_file = output_path / "batch_processing_summary.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        headers = [
            "Filename", 
            "Split", 
            "SourceFile", 
            "SequenceNum",
            "DurationSec", 
            "SpeechSec", 
            "SilenceSec", 
            "SpeechPercent"
        ]
        
        total_files = 0
        total_duration_sec = 0
        total_speech_sec = 0
        
        processor = getattr(self, 'processor', None)
        sequence_stats = getattr(processor, 'sequence_stats', []) if processor else []
        
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for stat in sequence_stats:
                writer.writerow({
                    "Filename": stat["display_filename"],
                    "Split": stat["split"],
                    "SourceFile": stat["source_filename"],
                    "SequenceNum": stat["sequence_number"],
                    "DurationSec": f"{stat['total_duration_sec']:.2f}",
                    "SpeechSec": f"{stat['speech_duration_sec']:.2f}",
                    "SilenceSec": f"{stat['silence_duration_sec']:.2f}",
                    "SpeechPercent": f"{stat['speech_percent']:.2f}"
                })
                
                total_files += 1
                total_duration_sec += stat['total_duration_sec']
                total_speech_sec += stat['speech_duration_sec']
        split_summary = {}
        for stat in sequence_stats:
            split = stat["split"]
            if split not in split_summary:
                split_summary[split] = {"files": 0, "duration": 0, "speech": 0}
            
            split_summary[split]["files"] += 1
            split_summary[split]["duration"] += stat["total_duration_sec"]
            split_summary[split]["speech"] += stat["speech_duration_sec"]
            
        with open(summary_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["SUMMARY", "Generated:", timestamp])
            writer.writerow(["Split", "Files", "Duration(sec)", "Speech(sec)", "Speech(%)"])
            for split, data in split_summary.items():
                speech_percent = (data["speech"] / data["duration"] * 100) if data["duration"] > 0 else 0
                writer.writerow([
                    split, 
                    data["files"], 
                    f"{data['duration']:.2f}", 
                    f"{data['speech']:.2f}", 
                    f"{speech_percent:.2f}%"
                ])
            
            overall_speech_percent = (total_speech_sec / total_duration_sec * 100) if total_duration_sec > 0 else 0
            writer.writerow([
                "TOTAL", 
                total_files, 
                f"{total_duration_sec:.2f}", 
                f"{total_speech_sec:.2f}", 
                f"{overall_speech_percent:.2f}%"
            ])
        
        print(f"Batch processing summary saved to {summary_file}")
        print(f"  Total: {total_files} sequence files recorded in CSV")

class AudioProcessingPipeline:
    """Main pipeline class that orchestrates the entire audio processing workflow."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.scanner = DirectoryScanner()
        self.batch_processor = BatchProcessor(self.config)
    
    def process_directory(self, input_dir: str = "input_data", 
                         output_dir: str = "Recompiled_Output") -> Optional[BatchStats]:
        # Scan input directory
        input_files = self.scanner.scan_input_directory(input_dir)
        
        if not input_files:
            print(f"No valid files found in {input_dir}. Check directory structure.")
            return None
        
        # Process files
        batch_stats = self.batch_processor.process_batch(input_files, output_dir)
        
        return batch_stats

def main():
    # Configure processing parameters with direct targets
    config = ProcessingConfig(
        target_hours_speech=1,    # Target hours for speech
        target_hours_silence=1,   # Target hours for silence
        speech_padding_ms=200,         # 200ms padding around speech
        create_splits=True,            # Create DEV/TRAIN splits
        dev_ratio=0.2,                 # 20% for DEV set
        silence_reserve_ratio=0.4,     # 40% of silence reserved for interspersing
    )
    
    # Load config into pipeline
    pipeline = AudioProcessingPipeline(config)
    
    # Process directory with temporal sequence output approach
    stats = pipeline.process_directory(
        input_dir="input_data",
        output_dir="Recompiled_Output"
    )
    
if __name__ == "__main__":
    main()
