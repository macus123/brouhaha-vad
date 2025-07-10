import csv
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from pydub import AudioSegment

from split_seg import read_ground_truth, get_non_speech_segments, save_audio_safely
from audio_dataclasses import (
    AudioSegmentInfo, TimestampMapping, ProcessingConfig,
    FileProcessingInfo, ProcessingResult, SetStats, BatchStats
)


class AudioProcessor:
    """Main audio processing class with sophisticated algorithms."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.audio: Optional[AudioSegment] = None
        self.audio_cache: Dict[str, AudioSegment] = {}  # Cache for multi-file processing
    
    def format_duration(self, ms: float) -> str:
        """Format duration in milliseconds to a readable string."""
        seconds = ms / 1000
        return str(datetime.timedelta(seconds=int(seconds)))
    
    def format_time_mmss(self, seconds: float) -> str:
        """Format seconds as MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    def load_and_analyze_audio(self, input_wav: str, ground_truth: str) -> Tuple[List[AudioSegmentInfo], float]:
        """Load audio file and analyze speech/non-speech segments."""
        file_stem = Path(input_wav).stem
        
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
        print(f"Target: {self.config.target_hours} hours with 1:1 speech/non-speech ratio")
        
        self.audio = AudioSegment.from_file(input_wav)
        total_duration_ms = len(self.audio)
        total_duration_sec = total_duration_ms / 1000
        
        print(f"Original audio: {self.format_duration(total_duration_ms)}")
        
        # Read ground truth and get speech segments
        speech_segments = read_ground_truth(ground_truth)
        non_speech_segments = get_non_speech_segments(speech_segments, total_duration_sec)
        
        # Create AudioSegmentInfo objects
        all_segments = []
        
        # Add speech segments with padding
        padded_speech_segments = self._add_padding_to_speech(speech_segments, total_duration_sec)
        merged_speech_segments = self._merge_overlapping_segments(padded_speech_segments)
        
        for start, end in merged_speech_segments:
            all_segments.append(AudioSegmentInfo(start=start, end=end, type="speech"))
        
        # Recalculate non-speech segments based on merged speech segments
        merged_non_speech_segments = get_non_speech_segments(merged_speech_segments, total_duration_sec)
        
        for start, end in merged_non_speech_segments:
            all_segments.append(AudioSegmentInfo(start=start, end=end, type="non-speech"))
        
        # Sort by start time
        all_segments.sort(key=lambda x: x.start)
        
        return all_segments, total_duration_sec
    
    def _add_padding_to_speech(self, speech_segments: List[Tuple[float, float]], total_duration_sec: float) -> List[Tuple[float, float]]:
        """Add padding to speech segments."""
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
            
            # Load audio
            audio = AudioSegment.from_file(file_info.audio_path)
            self.audio_cache[file_id] = audio
            
            total_duration_ms = len(audio)
            total_duration_sec = total_duration_ms / 1000
            file_durations[file_id] = total_duration_sec
            
            # Read ground truth and get speech segments
            speech_segments = read_ground_truth(file_info.ground_truth_path)
            non_speech_segments = get_non_speech_segments(speech_segments, total_duration_sec)
            
            # Add speech segments with padding
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
        """Create ground truth file for recompiled audio."""
        speech_segments = [ts for ts in timestamps if ts.type == "speech"]
        
        with open(output_path, 'w') as f:
            for ts in speech_segments:
                # Write in the expected format: start_time \t end_time \t text
                f.write(f"{ts.output_start_sec:.3f}\t{ts.output_end_sec:.3f}\tspeech\n")
    
    def compile_multi_source_audio(self, segments: List[AudioSegmentInfo]) -> Tuple[AudioSegment, List[TimestampMapping]]:
        """Compile audio from multiple source files."""
        compiled_audio = AudioSegment.empty()
        timestamp_map = []
        output_time_ms = 0
        
        for i, segment in enumerate(segments):
            # Get source audio from cache
            source_audio = self.audio_cache[segment.file_id]
            
            # Extract segment
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            segment_audio = source_audio[start_ms:end_ms]
            segment_duration_ms = len(segment_audio)
            
            # Create timestamp mapping
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
        
        return compiled_audio, timestamp_map
    
    def create_balanced_timeline_multi_file(self, segments: List[AudioSegmentInfo]) -> List[AudioSegmentInfo]:
        """Create balanced timeline from multiple files using the same core algorithms."""
        target_ms = int(self.config.target_hours * 3600 * 1000)
        speech_target_ms = target_ms * self.config.speech_ratio
        silence_target_ms = target_ms * (1 - self.config.speech_ratio)
        
        # Separate speech and silence segments
        speech_segments = [s for s in segments if s.type == "speech"]
        silence_segments = [s for s in segments if s.type == "non-speech"]
        
        # Calculate totals
        total_speech_ms = sum(s.duration for s in speech_segments)
        total_silence_ms = sum(s.duration for s in silence_segments)
        
        print(f"Multi-file speech content: {self.format_duration(total_speech_ms)}")
        print(f"Multi-file non-speech content: {self.format_duration(total_silence_ms)}")
        
        # Check if we have enough content
        if total_speech_ms < speech_target_ms or total_silence_ms < silence_target_ms:
            print(f"Warning: Insufficient audio to reach target duration")
            # Adjust target proportionally
            available_total = min(total_speech_ms / self.config.speech_ratio, 
                                total_silence_ms / (1 - self.config.speech_ratio))
            speech_target_ms = available_total * self.config.speech_ratio
            silence_target_ms = available_total * (1 - self.config.speech_ratio)
            print(f"Adjusting target - Speech: {self.format_duration(speech_target_ms)}, Silence: {self.format_duration(silence_target_ms)}")
        
        # Apply sophisticated silence distribution algorithm (preserved from original)
        return self._distribute_silence_intelligently_multi_file(speech_segments, silence_segments, 
                                                               speech_target_ms, silence_target_ms)
    
    def _distribute_silence_intelligently_multi_file(self, speech_segments: List[AudioSegmentInfo], 
                                                   silence_segments: List[AudioSegmentInfo], 
                                                   speech_target_ms: float, silence_target_ms: float) -> List[AudioSegmentInfo]:
        """Multi-file version of sophisticated silence distribution."""
        # Reserve silence for interspersing (same algorithm)
        reserved_silence_ms = silence_target_ms * self.config.silence_reserve_ratio
        primary_silence_ms = silence_target_ms - reserved_silence_ms
        
        # Sort segments - preserve temporal order within files
        speech_segments.sort(key=lambda x: (x.file_id, x.start))
        silence_segments.sort(key=lambda x: x.duration)  # Sort by duration for reservation
        
        # Reserve short silence segments for interspersing
        reserved_silence = []
        reserved_silence_duration = 0
        
        for segment in silence_segments:
            if reserved_silence_duration < reserved_silence_ms:
                if reserved_silence_duration + segment.duration <= reserved_silence_ms * 1.1:
                    reserved_silence.append(segment)
                    reserved_silence_duration += segment.duration
                elif reserved_silence_duration < reserved_silence_ms * 0.9:
                    # Split segment if needed
                    remaining_needed = reserved_silence_ms - reserved_silence_duration
                    if segment.duration > remaining_needed * 2:
                        split_segment = AudioSegmentInfo(
                            start=segment.start,
                            end=segment.start + (remaining_needed / 1000),
                            type=segment.type,
                            source_file=segment.source_file,
                            file_id=segment.file_id
                        )
                        reserved_silence.append(split_segment)
                        reserved_silence_duration += remaining_needed
                    else:
                        reserved_silence.append(segment)
                        reserved_silence_duration += segment.duration
        
        # Build primary timeline with remaining silence
        reserved_ids = set(id(seg) for seg in reserved_silence)
        primary_candidates = [seg for seg in silence_segments if id(seg) not in reserved_ids]
        # Sort by file_id and then by start time to maintain temporal order
        primary_silence = sorted(primary_candidates, key=lambda x: (x.file_id, x.start))
        
        # Create balanced segments using alternating approach
        balanced_segments = self._alternate_segments_multi_file(speech_segments, primary_silence, 
                                                              speech_target_ms, primary_silence_ms)
        
        # Intersperse reserved silence in long speech runs
        final_segments = self._intersperse_reserved_silence(balanced_segments, reserved_silence)
        
        return final_segments
    
    def _alternate_segments_multi_file(self, speech_segments: List[AudioSegmentInfo], 
                                     silence_segments: List[AudioSegmentInfo],
                                     speech_quota: float, silence_quota: float) -> List[AudioSegmentInfo]:
        """Multi-file version of alternating segments algorithm."""
        balanced_segments = []
        speech_quota_remaining = speech_quota
        silence_quota_remaining = silence_quota
        
        speech_index = 0
        silence_index = 0
        last_type_added = None
        
        # Start with some silence for natural beginning
        if silence_segments and silence_quota_remaining > 0:
            first_segment = silence_segments[0]
            if first_segment.duration > silence_quota_remaining * 0.25:
                desired_duration = min(silence_quota_remaining * 0.2, first_segment.duration)
                shortened = AudioSegmentInfo(
                    start=first_segment.start,
                    end=first_segment.start + (desired_duration / 1000),
                    type=first_segment.type,
                    source_file=first_segment.source_file,
                    file_id=first_segment.file_id
                )
                balanced_segments.append(shortened)
                silence_quota_remaining -= desired_duration
            else:
                balanced_segments.append(first_segment)
                silence_quota_remaining -= first_segment.duration
            silence_index = 1
            last_type_added = "non-speech"
        
        # Alternate between speech and silence
        while (speech_quota_remaining > 0 or silence_quota_remaining > 0) and \
              (speech_index < len(speech_segments) or silence_index < len(silence_segments)):
            
            add_speech = self._should_add_speech(last_type_added, speech_quota_remaining, 
                                               silence_quota_remaining, speech_index, 
                                               silence_index, len(speech_segments), 
                                               len(silence_segments))
            
            if add_speech:
                segment = speech_segments[speech_index]
                speech_index += 1
                
                if segment.duration <= speech_quota_remaining:
                    balanced_segments.append(segment)
                    speech_quota_remaining -= segment.duration
                else:
                    # Split segment
                    partial = AudioSegmentInfo(
                        start=segment.start,
                        end=segment.start + (speech_quota_remaining / 1000),
                        type=segment.type,
                        source_file=segment.source_file,
                        file_id=segment.file_id
                    )
                    balanced_segments.append(partial)
                    speech_quota_remaining = 0
                
                last_type_added = "speech"
            else:
                segment = silence_segments[silence_index]
                silence_index += 1
                
                if segment.duration <= silence_quota_remaining:
                    balanced_segments.append(segment)
                    silence_quota_remaining -= segment.duration
                else:
                    # Split segment
                    partial = AudioSegmentInfo(
                        start=segment.start,
                        end=segment.start + (silence_quota_remaining / 1000),
                        type=segment.type,
                        source_file=segment.source_file,
                        file_id=segment.file_id
                    )
                    balanced_segments.append(partial)
                    silence_quota_remaining = 0
                
                last_type_added = "non-speech"
        
        return balanced_segments
    
    def _should_add_speech(self, last_type: str, speech_quota: float, silence_quota: float,
                          speech_index: int, silence_index: int, total_speech: int, 
                          total_silence: int) -> bool:
        """Determine whether to add speech or silence segment next."""
        if last_type == "speech" and silence_quota > 0 and silence_index < total_silence:
            return False
        elif last_type == "non-speech" and speech_quota > 0 and speech_index < total_speech:
            return True
        else:
            # Add whatever has quota remaining
            if speech_quota > 0 and speech_index < total_speech:
                return True
            elif silence_quota > 0 and silence_index < total_silence:
                return False
            else:
                return False
    
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
    
    def compile_audio_from_segments(self, segments: List[AudioSegmentInfo]) -> Tuple[AudioSegment, List[TimestampMapping]]:
        """Compile audio from segments with timestamp tracking."""
        return self.compile_multi_source_audio(segments)
    
    def process_multiple_files(self, input_files: List[FileProcessingInfo], 
                             output_dir: str, output_name: str = "combined") -> ProcessingResult:
        """Process multiple audio files with sophisticated multi-file stitching."""
        file_stem = output_name
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Load and analyze all files
        all_segments, file_durations = self.load_and_analyze_multiple_files(input_files)
        
        # Create balanced timeline using multi-file algorithm
        balanced_segments = self.create_balanced_timeline_multi_file(all_segments)
        
        # Compile balanced audio from multiple sources
        balanced_audio, balanced_timestamps = self.compile_multi_source_audio(balanced_segments)
        
        # Calculate statistics
        balanced_speech_ms = sum(ts.duration_sec * 1000 for ts in balanced_timestamps if ts.type == "speech")
        balanced_non_speech_ms = sum(ts.duration_sec * 1000 for ts in balanced_timestamps if ts.type == "non-speech")
        
        # Create output paths
        balanced_output_path = output_path / f"{file_stem}_balanced_{self.config.target_hours:.1f}h.wav"
        gt_output_path = output_path / f"{file_stem}_balanced_{self.config.target_hours:.1f}h.txt"
        
        # Save balanced audio
        save_audio_safely(balanced_audio, balanced_output_path)
        
        # Create ground truth file
        self.create_ground_truth_file(balanced_timestamps, str(gt_output_path))
        
        # Calculate total original duration
        total_original_duration_sec = sum(file_durations.values())
        
        # Prepare result
        result = ProcessingResult(
            balanced_output=str(balanced_output_path),
            original_duration_hours=total_original_duration_sec / 3600,
            balanced_audio=balanced_audio,
            balanced_timestamps=balanced_timestamps,
            balanced_speech_ms=balanced_speech_ms,
            balanced_non_speech_ms=balanced_non_speech_ms,
            ground_truth_output=str(gt_output_path),
            files_used=[file_info.audio_path for file_info in input_files]
        )
        
        # Handle dev/train splits if enabled
        if self.config.create_splits:
            self._create_dev_train_splits_multi_file(all_segments, balanced_segments, 
                                                   input_files, file_stem, output_path, result)
        
        return result
    
    def _create_dev_train_splits_multi_file(self, all_segments: List[AudioSegmentInfo], 
                                          balanced_segments: List[AudioSegmentInfo],
                                          input_files: List[FileProcessingInfo],
                                          file_stem: str, output_path: Path, 
                                          result: ProcessingResult) -> None:
        """Create dev/train splits from remaining segments across multiple files."""
        # Find remaining segments
        used_segments = set(id(segment) for segment in balanced_segments)
        remaining_segments = [seg for seg in all_segments if id(seg) not in used_segments]
        
        # Sort by file_id and then by start time to maintain temporal order within files
        remaining_segments.sort(key=lambda x: (x.file_id, x.start))
        
        if not remaining_segments:
            return
        
        # Split remaining segments
        dev_segments, train_segments = self._split_remaining_segments(remaining_segments)
        
        if dev_segments:
            dev_audio, dev_timestamps = self.compile_multi_source_audio(dev_segments)
            dev_output_path = output_path / f"{file_stem}_dev.wav"
            dev_gt_path = output_path / f"{file_stem}_dev.txt"
            
            save_audio_safely(dev_audio, dev_output_path)
            self.create_ground_truth_file(dev_timestamps, str(dev_gt_path))
            
            result.dev_output = str(dev_output_path)
            result.dev_audio = dev_audio
            result.dev_timestamps = dev_timestamps
            result.dev_ground_truth = str(dev_gt_path)
        
        if train_segments:
            train_audio, train_timestamps = self.compile_multi_source_audio(train_segments)
            train_output_path = output_path / f"{file_stem}_train.wav"
            train_gt_path = output_path / f"{file_stem}_train.txt"
            
            save_audio_safely(train_audio, train_output_path)
            self.create_ground_truth_file(train_timestamps, str(train_gt_path))
            
            result.train_output = str(train_output_path)
            result.train_audio = train_audio
            result.train_timestamps = train_timestamps
            result.train_ground_truth = str(train_gt_path)

    def process_unified_output(self, input_files: List[FileProcessingInfo], 
                             test_dir: str, dev_dir: str, train_dir: str) -> ProcessingResult:
        """Process multiple files into unified TEST/DEV/TRAIN outputs."""
        print(f"Creating unified outputs from {len(input_files)} files...")
        
        # Load and analyze all files
        all_segments, file_durations = self.load_and_analyze_multiple_files(input_files)
        
        # Calculate total content available
        total_speech_ms = sum(s.duration for s in all_segments if s.type == "speech")
        total_silence_ms = sum(s.duration for s in all_segments if s.type == "non-speech")
        total_duration_sec = sum(file_durations.values())
        
        print(f"Total content available: {self.format_duration((total_speech_ms + total_silence_ms))}")
        print(f"  Speech: {self.format_duration(total_speech_ms)}")
        print(f"  Silence: {self.format_duration(total_silence_ms)}")
        
        # Create TEST set using target duration and ratio
        test_segments = self.create_balanced_timeline_multi_file(all_segments)
        test_audio, test_timestamps = self.compile_multi_source_audio(test_segments)
        
        # Calculate TEST statistics
        test_speech_ms = sum(ts.duration_sec * 1000 for ts in test_timestamps if ts.type == "speech")
        test_non_speech_ms = sum(ts.duration_sec * 1000 for ts in test_timestamps if ts.type == "non-speech")
        
        # Create TEST output files
        test_output_path = Path(test_dir) / f"unified_test_{self.config.target_hours:.1f}h.wav"
        test_gt_path = Path(test_dir) / f"unified_test_{self.config.target_hours:.1f}h.txt"
        
        save_audio_safely(test_audio, test_output_path)
        self.create_ground_truth_file(test_timestamps, str(test_gt_path))
        
        print(f"TEST set created: {self.format_duration(len(test_audio))}")
        
        # Prepare result structure
        result = ProcessingResult(
            test_output=str(test_output_path),
            test_ground_truth=str(test_gt_path),
            original_duration_hours=total_duration_sec / 3600,
            test_audio=test_audio,
            test_timestamps=test_timestamps,
            test_speech_ms=test_speech_ms,
            test_non_speech_ms=test_non_speech_ms,
            files_used=[file_info.audio_path for file_info in input_files]
        )
        
        # Create DEV and TRAIN sets from remaining content if enabled
        if self.config.create_splits:
            self._create_unified_dev_train_splits(all_segments, test_segments, 
                                                dev_dir, train_dir, result)
        
        return result
    
    def _create_unified_dev_train_splits(self, all_segments: List[AudioSegmentInfo], 
                                       test_segments: List[AudioSegmentInfo],
                                       dev_dir: str, train_dir: str, 
                                       result: ProcessingResult) -> None:
        """Create unified DEV and TRAIN sets from remaining segments."""
        # Find remaining segments after TEST set creation
        used_segments = set(id(segment) for segment in test_segments)
        remaining_segments = [seg for seg in all_segments if id(seg) not in used_segments]
        
        # Sort by file_id and then by start time to maintain temporal order within files
        remaining_segments.sort(key=lambda x: (x.file_id, x.start))
        
        if not remaining_segments:
            print("No remaining content for DEV/TRAIN splits")
            return
        
        print(f"Creating DEV/TRAIN splits from remaining {len(remaining_segments)} segments...")
        
        # Split remaining segments into DEV and TRAIN
        dev_segments, train_segments = self._split_remaining_segments(remaining_segments)
        
        # Create DEV set
        if dev_segments:
            # Apply the same balanced timeline algorithm to DEV set
            dev_balanced_segments = self._create_balanced_subset(dev_segments, "DEV")
            dev_audio, dev_timestamps = self.compile_multi_source_audio(dev_balanced_segments)
            
            dev_output_path = Path(dev_dir) / "unified_dev.wav"
            dev_gt_path = Path(dev_dir) / "unified_dev.txt"
            
            save_audio_safely(dev_audio, dev_output_path)
            self.create_ground_truth_file(dev_timestamps, str(dev_gt_path))
            
            result.dev_output = str(dev_output_path)
            result.dev_audio = dev_audio
            result.dev_timestamps = dev_timestamps
            result.dev_ground_truth = str(dev_gt_path)
            
            print(f"DEV set created: {self.format_duration(len(dev_audio))}")
        
        # Create TRAIN set
        if train_segments:
            # Apply the same balanced timeline algorithm to TRAIN set
            train_balanced_segments = self._create_balanced_subset(train_segments, "TRAIN")
            train_audio, train_timestamps = self.compile_multi_source_audio(train_balanced_segments)
            
            train_output_path = Path(train_dir) / "unified_train.wav"
            train_gt_path = Path(train_dir) / "unified_train.txt"
            
            save_audio_safely(train_audio, train_output_path)
            self.create_ground_truth_file(train_timestamps, str(train_gt_path))
            
            result.train_output = str(train_output_path)
            result.train_audio = train_audio
            result.train_timestamps = train_timestamps
            result.train_ground_truth = str(train_gt_path)
            
            print(f"TRAIN set created: {self.format_duration(len(train_audio))}")
    
    def _create_balanced_subset(self, segments: List[AudioSegmentInfo], set_name: str) -> List[AudioSegmentInfo]:
        """Create a balanced subset using the same algorithms as the main processing."""
        speech_segments = [s for s in segments if s.type == "speech"]
        silence_segments = [s for s in segments if s.type == "non-speech"]
        
        total_speech_ms = sum(s.duration for s in speech_segments)
        total_silence_ms = sum(s.duration for s in silence_segments)
        
        # Check if we have any content at all
        if total_speech_ms == 0 or total_silence_ms == 0:
            print(f"  Warning: {set_name} set has insufficient content (speech: {self.format_duration(total_speech_ms)}, silence: {self.format_duration(total_silence_ms)})")
            # Return all segments if we don't have both types
            return segments
        
        # Use all available content, but apply balanced ratio
        available_total_ms = total_speech_ms + total_silence_ms
        target_speech_ms = available_total_ms * self.config.speech_ratio
        target_silence_ms = available_total_ms * (1 - self.config.speech_ratio)
        
        # Adjust targets based on available content while preventing zero targets
        if total_speech_ms < target_speech_ms:
            speech_ratio = total_speech_ms / available_total_ms
            target_speech_ms = total_speech_ms
            # Make sure we don't exceed available silence
            target_silence_ms = min(available_total_ms - target_speech_ms, total_silence_ms)
        
        if total_silence_ms < target_silence_ms:
            silence_ratio = total_silence_ms / available_total_ms
            target_silence_ms = total_silence_ms
            # Make sure we don't exceed available speech
            target_speech_ms = min(available_total_ms - target_silence_ms, total_speech_ms)
        
        # Final safeguard against zero targets
        if target_speech_ms <= 0 or target_silence_ms <= 0:
            print(f"  Warning: {set_name} targets too small, using all available content")
            target_speech_ms = total_speech_ms
            target_silence_ms = total_silence_ms
        
        print(f"  {set_name} targets - Speech: {self.format_duration(target_speech_ms)}, Silence: {self.format_duration(target_silence_ms)}")
        
        # Apply the same sophisticated distribution algorithm
        return self._distribute_silence_intelligently_multi_file(
            speech_segments, silence_segments, target_speech_ms, target_silence_ms
        )
        
    def _split_remaining_segments(self, remaining_segments: List[AudioSegmentInfo]) -> Tuple[List[AudioSegmentInfo], List[AudioSegmentInfo]]:
        """Split remaining segments into DEV and TRAIN sets."""
        if not remaining_segments:
            return [], []
        
        # Calculate split point based on dev_ratio
        total_segments = len(remaining_segments)
        dev_count = int(total_segments * self.config.dev_ratio)
        
        # Ensure we have at least some segments for each set if possible
        if dev_count == 0 and total_segments > 1:
            dev_count = 1
        elif dev_count == total_segments and total_segments > 1:
            dev_count = total_segments - 1
        
        # Split the segments
        dev_segments = remaining_segments[:dev_count]
        train_segments = remaining_segments[dev_count:]
        
        print(f"    DEV: {len(dev_segments)} segments")
        print(f"    TRAIN: {len(train_segments)} segments")
        
        return dev_segments, train_segments

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
    
    def process_batch(self, input_files: List[FileProcessingInfo], 
                     output_dir: str) -> BatchStats:
        """Process multiple audio files and combine into single TEST/DEV/TRAIN outputs."""
        # Create output directories
        output_path = Path(output_dir)
        test_dir = output_path / "TEST"
        dev_dir = output_path / "DEV"
        train_dir = output_path / "TRAIN"
        
        test_dir.mkdir(exist_ok=True, parents=True)
        if self.config.create_splits:
            dev_dir.mkdir(exist_ok=True, parents=True)
            train_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize batch statistics
        self.batch_stats = BatchStats(total_files=len(input_files))
        
        print(f"\nProcessing {len(input_files)} files into unified TEST/DEV/TRAIN sets...")
        
        try:
            # Process all files together to create unified outputs
            result = self.processor.process_unified_output(
                input_files, str(test_dir), str(dev_dir), str(train_dir)
            )
            
            # Update batch statistics for unified processing
            self._update_unified_batch_stats(self.batch_stats, result, input_files)
            
            print(f"  ✓ Processing Complete")
            
        except Exception as e:
            print(f"  ✗ Error in processing: {e}")
            # Create error entry
            error_detail = {
                "filename": "unified_processing",
                "error": str(e)
            }
            self.batch_stats.test_stats.file_details.append(error_detail)
        
        # Calculate aggregate statistics
        self._finalize_batch_stats(self.batch_stats)
        
        # Create comprehensive summary CSV
        self._create_summary_csv(self.batch_stats, output_dir)
        
        return self.batch_stats
    
    def _update_unified_batch_stats(self, batch_stats: BatchStats, 
                                  result: ProcessingResult, 
                                  input_files: List[FileProcessingInfo]) -> None:
        """Update batch statistics for unified processing."""
        # Update original duration (sum of all input files)
        batch_stats.total_original_duration = result.original_duration_hours
        
        # Process TEST set
        self._process_unified_set_stats(
            batch_stats.test_stats,
            "TEST",
            result.test_audio,
            result.test_timestamps,
            result.test_speech_ms,
            result.test_non_speech_ms,
            input_files,
            result.original_duration_hours
        )
        
        # Process DEV set if available
        if result.dev_audio and len(result.dev_audio) > 0:
            dev_speech_ms = sum(ts.duration_sec * 1000 for ts in result.dev_timestamps if ts.type == "speech")
            dev_non_speech_ms = sum(ts.duration_sec * 1000 for ts in result.dev_timestamps if ts.type == "non-speech")
            
            self._process_unified_set_stats(
                batch_stats.dev_stats,
                "DEV",
                result.dev_audio,
                result.dev_timestamps,
                dev_speech_ms,
                dev_non_speech_ms,
                input_files,
                result.original_duration_hours
            )
        
        # Process TRAIN set if available
        if result.train_audio and len(result.train_audio) > 0:
            train_speech_ms = sum(ts.duration_sec * 1000 for ts in result.train_timestamps if ts.type == "speech")
            train_non_speech_ms = sum(ts.duration_sec * 1000 for ts in result.train_timestamps if ts.type == "non-speech")
            
            self._process_unified_set_stats(
                batch_stats.train_stats,
                "TRAIN",
                result.train_audio,
                result.train_timestamps,
                train_speech_ms,
                train_non_speech_ms,
                input_files,
                result.original_duration_hours
            )
    
    def _process_unified_set_stats(self, set_stats: SetStats, 
                                 set_type: str,
                                 audio: AudioSegment,
                                 timestamps: List[TimestampMapping],
                                 speech_ms: float,
                                 non_speech_ms: float,
                                 input_files: List[FileProcessingInfo],
                                 original_duration_hours: float) -> None:
        """Process statistics for unified processing."""
        duration_hours = len(audio) / 3600000
        speech_hours = speech_ms / 3600000
        non_speech_hours = non_speech_ms / 3600000
        
        # Update set totals
        set_stats.total_duration += duration_hours
        set_stats.speech_duration += speech_hours
        set_stats.non_speech_duration += non_speech_hours
        
        # Calculate ratio accuracy (only for TEST set)
        if set_type == "TEST" and duration_hours > 0:
            speech_ratio = speech_hours / duration_hours
            target_ratio = self.config.speech_ratio
            ratio_accuracy = 1 - abs(speech_ratio - target_ratio)
            self.batch_stats.speech_ratio_accuracy.append(ratio_accuracy)
        
        # Create file details for unified processing
        speech_ratio_val = speech_hours / duration_hours if duration_hours > 0 else 0
        unified_filename = f"unified_{set_type.lower()}_from_{len(input_files)}_files"
        
        file_stats = {
            "filename": unified_filename,
            "set_type": set_type,
            "original_hours": original_duration_hours,
            "duration_hours": duration_hours,
            "speech_hours": speech_hours,
            "speech_ratio": speech_ratio_val,
            "non_speech_hours": non_speech_hours,
            "source_files": [Path(f.audio_path).name for f in input_files]
        }
        
        set_stats.file_details.append(file_stats)

    def _finalize_batch_stats(self, batch_stats: BatchStats) -> None:
        """Calculate final aggregate statistics for all sets."""
        # No additional calculations needed
        pass
    
    def _create_summary_csv(self, batch_stats: BatchStats, output_dir: str) -> None:
        """Create a CSV summary file that includes all sets."""
        output_path = Path(output_dir)
        summary_file = output_path / "batch_processing_summary.csv"
        
        headers = [
            "Filename", "SplitType", "OriginalDuration(h)", "OutputDuration(h)",
            "SpeechDuration(h)", "SpeechPercentage", "SilenceDuration(h)", "SilencePercentage"
        ]
        
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            # Write error entries
            error_files = [fd for fd in batch_stats.test_stats.file_details if 'error' in fd]
            for file_detail in error_files:
                writer.writerow({"Filename": file_detail['filename'], "SplitType": "ERROR"})
            
            # Process each set type
            for set_type in ["TEST", "DEV", "TRAIN"]:
                set_stats = batch_stats.stats_by_type(set_type)
                
                for file_detail in set_stats.file_details:
                    if 'error' in file_detail:
                        continue  # Already processed errors
                    
                    speech_hours = file_detail.get('speech_hours', 0)
                    total_hours = file_detail.get('duration_hours', 0)
                    speech_percentage = (speech_hours / total_hours) * 100 if total_hours > 0 else 0
                    silence_hours = total_hours - speech_hours;
                    
                    writer.writerow({
                        "Filename": file_detail['filename'],
                        "SplitType": set_type,
                        "OriginalDuration(h)": f"{file_detail.get('original_hours', 0):.4f}",
                        "OutputDuration(h)": f"{total_hours:.4f}",
                        "SpeechDuration(h)": f"{speech_hours:.4f}",
                        "SpeechPercentage": f"{speech_percentage:.2f}",
                        "SilenceDuration(h)": f"{silence_hours:.4f}",
                        "SilencePercentage": f"{100 - speech_percentage:.2f}"
                    })
        
        print(f"\nbatch processing summary saved to {summary_file}")

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
    # Configure processing parameters
    config = ProcessingConfig(
        target_hours=0.135,              # Target X hour output for TEST set
        speech_ratio=0.5,              # 50% speech, 50% silence (1:1 ratio)
        speech_padding_ms=200,         # 200ms padding around speech
        create_splits=True,            # Create DEV/TRAIN splits
        dev_ratio=0.2,                 # 20% for DEV set
        silence_reserve_ratio=0.4,     # 40% of silence reserved for interspersing
    )
    
    # Example: Different speech ratios
    # config.speech_ratio = 0.3  # 30% speech, 70% silence
    # config.speech_ratio = 0.7  # 70% speech, 30% silence
    
    # Load config into pipeline
    pipeline = AudioProcessingPipeline(config)
    
    # Process directory with unified output approach
    stats = pipeline.process_directory(
        input_dir="input_data",
        output_dir="Recompiled_Output"
    )
    
if __name__ == "__main__":
    main()
