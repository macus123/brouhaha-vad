import csv
import datetime
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from pydub import AudioSegment

from split_seg import read_ground_truth, get_non_speech_segments, save_audio_safely


@dataclass
class AudioSegmentInfo:
    """Represents an audio segment with timing and type information."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    type: str     # "speech" or "non-speech"
    duration: float = field(init=False)  # Duration in milliseconds
    
    def __post_init__(self):
        self.duration = (self.end - self.start) * 1000


@dataclass
class TimestampMapping:
    """Maps output timestamps to original audio timestamps."""
    segment_index: int
    original_start_sec: float
    original_end_sec: float
    output_start_sec: float
    output_end_sec: float
    duration_sec: float
    type: str
    is_continuous: bool


@dataclass
class ContinuityStats:
    """Statistics about segment continuity."""
    sequences: List[Dict[str, Any]] = field(default_factory=list)
    num_sequences: int = 0
    longest_sequence: int = 0
    avg_sequence_length: float = 0.0
    total_segments: int = 0
    continuous_segments: int = 0


@dataclass
class ProcessingConfig:
    """Configuration for audio processing."""
    target_hours: float = 1.0
    speech_padding_ms: int = 200
    create_splits: bool = True
    dev_ratio: float = 0.2
    silence_reserve_ratio: float = 0.4


@dataclass
class FileProcessingInfo:
    """Information about a file to be processed."""
    audio_path: str
    ground_truth_path: str
    set_type: str = "TEST"


@dataclass
class ProcessingResult:
    """Results from processing a single audio file."""
    balanced_output: str
    original_duration_hours: float
    balanced_audio: AudioSegment
    balanced_timestamps: List[TimestampMapping]
    balanced_speech_ms: float
    balanced_non_speech_ms: float
    dev_output: Optional[str] = None
    train_output: Optional[str] = None
    dev_audio: Optional[AudioSegment] = None
    train_audio: Optional[AudioSegment] = None
    dev_timestamps: List[TimestampMapping] = field(default_factory=list)
    train_timestamps: List[TimestampMapping] = field(default_factory=list)


# @dataclass
# class BatchStats:
#     """Statistics for batch processing."""
#     total_files: int = 0
#     total_original_duration: float = 0
#     total_balanced_duration: float = 0
#     total_speech_duration: float = 0
#     total_non_speech_duration: float = 0
#     speech_ratio_accuracy: List[float] = field(default_factory=list)
#     continuity_stats: ContinuityStats = field(default_factory=ContinuityStats)
#     file_details: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class SetStats:
    """Statistics for a specific set (TEST/DEV/TRAIN)."""
    total_duration: float = 0.0
    speech_duration: float = 0.0
    non_speech_duration: float = 0.0
    continuity_stats: ContinuityStats = field(default_factory=ContinuityStats)
    file_details: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BatchStats:
    """Statistics for batch processing with separate tracking for each set."""
    total_files: int = 0
    total_original_duration: float = 0
    speech_ratio_accuracy: List[float] = field(default_factory=list)
    
    # Statistics by set type
    test_stats: SetStats = field(default_factory=SetStats)
    dev_stats: SetStats = field(default_factory=SetStats)
    train_stats: SetStats = field(default_factory=SetStats)
    
    # Map for easy access by set type
    def stats_by_type(self, set_type: str) -> SetStats:
        """Get statistics for a specific set type."""
        set_type = set_type.upper()
        if set_type == "TEST":
            return self.test_stats
        elif set_type == "DEV":
            return self.dev_stats
        elif set_type == "TRAIN":
            return self.train_stats
        else:
            raise ValueError(f"Unknown set type: {set_type}")

class AudioProcessor:
    """Main audio processing class with sophisticated algorithms."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.audio: Optional[AudioSegment] = None
    
    def format_duration(self, ms: float) -> str:
        """Format duration in milliseconds to a readable string."""
        seconds = ms / 1000
        return str(datetime.timedelta(seconds=seconds))
    
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
    
    def create_balanced_timeline(self, segments: List[AudioSegmentInfo]) -> List[AudioSegmentInfo]:
        """
        Create a balanced timeline using sophisticated silence distribution algorithm.
        This preserves the original algorithm's intelligent silence interspersing.
        """
        target_ms = int(self.config.target_hours * 3600 * 1000)
        target_per_type_ms = target_ms // 2
        
        # Separate speech and silence segments
        speech_segments = [s for s in segments if s.type == "speech"]
        silence_segments = [s for s in segments if s.type == "non-speech"]
        
        # Calculate totals
        total_speech_ms = sum(s.duration for s in speech_segments)
        total_silence_ms = sum(s.duration for s in silence_segments)
        
        print(f"Speech content: {self.format_duration(total_speech_ms)} ({total_speech_ms / (total_speech_ms + total_silence_ms) * 100:.1f}%)")
        print(f"Non-speech content: {self.format_duration(total_silence_ms)} ({total_silence_ms / (total_speech_ms + total_silence_ms) * 100:.1f}%)")
        
        # Check if we have enough content
        if total_speech_ms < target_per_type_ms or total_silence_ms < target_per_type_ms:
            print(f"Warning: Insufficient audio to reach target duration with 1:1 ratio")
            target_per_type_ms = min(total_speech_ms, total_silence_ms)
            print(f"Adjusting target to: {self.format_duration(target_per_type_ms * 2)}")
        
        # Apply sophisticated silence distribution algorithm
        return self._distribute_silence_intelligently(speech_segments, silence_segments, target_per_type_ms)
    
    def _distribute_silence_intelligently(self, speech_segments: List[AudioSegmentInfo], 
                                        silence_segments: List[AudioSegmentInfo], 
                                        target_per_type_ms: float) -> List[AudioSegmentInfo]:
        """
        Sophisticated algorithm to distribute silence throughout the audio.
        This preserves the original algorithm's intelligent approach.
        """
        # Reserve silence for interspersing
        reserved_silence_ms = target_per_type_ms * self.config.silence_reserve_ratio
        primary_silence_ms = target_per_type_ms - reserved_silence_ms
        
        # Sort segments for optimal distribution
        speech_segments.sort(key=lambda x: x.start)
        short_silence_segments = sorted(silence_segments, key=lambda x: x.duration)
        
        # Reserve short silence segments for interspersing
        reserved_silence = []
        reserved_silence_duration = 0
        
        for segment in short_silence_segments:
            if reserved_silence_duration < reserved_silence_ms:
                if reserved_silence_duration + segment.duration <= reserved_silence_ms * 1.1:
                    reserved_silence.append(segment)
                    reserved_silence_duration += segment.duration
                elif reserved_silence_duration < reserved_silence_ms * 0.9:
                    # Split segment if needed
                    remaining_needed = reserved_silence_ms - reserved_silence_duration
                    if segment.duration > remaining_needed * 2:
                        # Create split segment
                        split_segment = AudioSegmentInfo(
                            start=segment.start,
                            end=segment.start + (remaining_needed / 1000),
                            type=segment.type
                        )
                        reserved_silence.append(split_segment)
                        reserved_silence_duration += remaining_needed
                    else:
                        reserved_silence.append(segment)
                        reserved_silence_duration += segment.duration
        
        # Build primary timeline
        reserved_ids = set(id(seg) for seg in reserved_silence)
        primary_candidates = [seg for seg in silence_segments if id(seg) not in reserved_ids]
        primary_silence = sorted(primary_candidates, key=lambda x: x.start)
        
        # Create balanced segments using alternating approach
        balanced_segments = self._alternate_segments(speech_segments, primary_silence, 
                                                   target_per_type_ms, primary_silence_ms)
        
        # Intersperse reserved silence in long speech runs
        final_segments = self._intersperse_reserved_silence(balanced_segments, reserved_silence)
        
        return sorted(final_segments, key=lambda x: x.start)
    
    def _alternate_segments(self, speech_segments: List[AudioSegmentInfo], 
                          silence_segments: List[AudioSegmentInfo],
                          speech_quota: float, silence_quota: float) -> List[AudioSegmentInfo]:
        """Alternate between speech and silence segments."""
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
                # Create shortened version
                desired_duration = min(silence_quota_remaining * 0.2, first_segment.duration)
                shortened = AudioSegmentInfo(
                    start=first_segment.start,
                    end=first_segment.start + (desired_duration / 1000),
                    type=first_segment.type
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
                        type=segment.type
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
                        type=segment.type
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
        compiled_audio = AudioSegment.empty()
        timestamp_map = []
        output_time_ms = 0
        
        for i, segment in enumerate(segments):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            segment_audio = self.audio[start_ms:end_ms]
            segment_duration_ms = len(segment_audio)
            
            # Determine continuity
            is_continuous = i == 0 or abs(segment.start - segments[i - 1].end) < 0.001
            
            timestamp_map.append(TimestampMapping(
                segment_index=i,
                original_start_sec=segment.start,
                original_end_sec=segment.end,
                output_start_sec=output_time_ms / 1000,
                output_end_sec=(output_time_ms + segment_duration_ms) / 1000,
                duration_sec=segment_duration_ms / 1000,
                type=segment.type,
                is_continuous=is_continuous
            ))
            
            compiled_audio += segment_audio
            output_time_ms += segment_duration_ms
        
        return compiled_audio, timestamp_map
    
    def process_single_file(self, audio_path: str, ground_truth_path: str, 
                          output_dir: str) -> ProcessingResult:
        """Process a single audio file with sophisticated algorithms."""
        file_stem = Path(audio_path).stem
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Load and analyze audio
        all_segments, total_duration_sec = self.load_and_analyze_audio(audio_path, ground_truth_path)
        
        # Create balanced timeline
        balanced_segments = self.create_balanced_timeline(all_segments)
        
        # Compile balanced audio
        balanced_audio, balanced_timestamps = self.compile_audio_from_segments(balanced_segments)
        
        # Calculate statistics
        balanced_speech_ms = sum(ts.duration_sec * 1000 for ts in balanced_timestamps if ts.type == "speech")
        balanced_non_speech_ms = sum(ts.duration_sec * 1000 for ts in balanced_timestamps if ts.type == "non-speech")
        
        # Create output path
        balanced_output_path = output_path / f"{file_stem}_balanced_{self.config.target_hours:.1f}h.wav"
        
        # Save balanced audio
        save_audio_safely(balanced_audio, balanced_output_path)
        
        # Prepare result
        result = ProcessingResult(
            balanced_output=str(balanced_output_path),
            original_duration_hours=total_duration_sec / 3600,
            balanced_audio=balanced_audio,
            balanced_timestamps=balanced_timestamps,
            balanced_speech_ms=balanced_speech_ms,
            balanced_non_speech_ms=balanced_non_speech_ms
        )
        
        # Handle dev/train splits if enabled
        if self.config.create_splits:
            self._create_dev_train_splits(all_segments, balanced_segments, file_stem, output_path, result)
        
        return result
    
    def _create_dev_train_splits(self, all_segments: List[AudioSegmentInfo], 
                               balanced_segments: List[AudioSegmentInfo],
                               file_stem: str, output_path: Path, 
                               result: ProcessingResult) -> None:
        """Create dev/train splits from remaining segments."""
        # Find remaining segments
        used_segments = set(id(segment) for segment in balanced_segments)
        remaining_segments = [seg for seg in all_segments if id(seg) not in used_segments]
        remaining_segments.sort(key=lambda x: x.start)
        
        if not remaining_segments:
            return
        
        # Split remaining segments
        dev_segments, train_segments = self._split_remaining_segments(remaining_segments)
        
        if dev_segments:
            dev_audio, dev_timestamps = self.compile_audio_from_segments(dev_segments)
            dev_output_path = output_path / f"{file_stem}_dev.wav"
            save_audio_safely(dev_audio, dev_output_path)
            result.dev_output = str(dev_output_path)
            result.dev_audio = dev_audio
            result.dev_timestamps = dev_timestamps
        
        if train_segments:
            train_audio, train_timestamps = self.compile_audio_from_segments(train_segments)
            train_output_path = output_path / f"{file_stem}_train.wav"
            save_audio_safely(train_audio, train_output_path)
            result.train_output = str(train_output_path)
            result.train_audio = train_audio
            result.train_timestamps = train_timestamps
    
    def _split_remaining_segments(self, remaining_segments: List[AudioSegmentInfo]) -> Tuple[List[AudioSegmentInfo], List[AudioSegmentInfo]]:
        """Split remaining segments into dev and train sets."""
        # Calculate quotas
        remaining_speech_ms = sum(s.duration for s in remaining_segments if s.type == "speech")
        remaining_non_speech_ms = sum(s.duration for s in remaining_segments if s.type == "non-speech")
        
        dev_speech_ms = remaining_speech_ms * self.config.dev_ratio
        dev_non_speech_ms = remaining_non_speech_ms * self.config.dev_ratio
        
        dev_segments = []
        train_segments = []
        
        current_dev_speech_ms = 0
        current_dev_non_speech_ms = 0
        
        for segment in remaining_segments:
            is_speech = segment.type == "speech"
            
            if (is_speech and current_dev_speech_ms < dev_speech_ms) or \
               (not is_speech and current_dev_non_speech_ms < dev_non_speech_ms):
                dev_segments.append(segment)
                if is_speech:
                    current_dev_speech_ms += segment.duration
                else:
                    current_dev_non_speech_ms += segment.duration
            else:
                train_segments.append(segment)
        
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


class ContinuityAnalyzer:
    """Analyzes continuity in audio segments."""
    
    @staticmethod
    def analyze_continuity(timestamps: List[TimestampMapping]) -> ContinuityStats:
        """Analyze continuity in a sophisticated way by identifying sequences."""
        if not timestamps:
            return ContinuityStats()
        
        sequences = []
        current_sequence = []
        continuous_count = 0
        
        for i, ts in enumerate(timestamps):
            is_continuous = ts.is_continuous if i > 0 else True
            
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
                        "start_time": timestamps[seq_start_idx].output_start_sec,
                        "end_time": timestamps[seq_end_idx].output_end_sec,
                    })
                current_sequence = [i]
        
        # Add the last sequence
        if current_sequence:
            seq_start_idx = current_sequence[0]
            seq_end_idx = current_sequence[-1]
            sequences.append({
                "segments": current_sequence,
                "length": len(current_sequence),
                "start_time": timestamps[seq_start_idx].output_start_sec,
                "end_time": timestamps[seq_end_idx].output_end_sec,
            })
        
        longest_seq = max([s["length"] for s in sequences]) if sequences else 0
        avg_seq_len = sum([s["length"] for s in sequences]) / len(sequences) if sequences else 0
        
        return ContinuityStats(
            sequences=sequences,
            num_sequences=len(sequences),
            longest_sequence=longest_seq,
            avg_sequence_length=avg_seq_len,
            total_segments=len(timestamps),
            continuous_segments=continuous_count
        )


class BatchProcessor:
    """Handles batch processing of multiple audio files with comprehensive statistics."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processor = AudioProcessor(config)
        self.batch_stats = None
    
    def process_batch(self, input_files: List[FileProcessingInfo], 
                     output_dir: str) -> BatchStats:
        """Process multiple audio files in batch with comprehensive statistics."""
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
        
        # Process each file
        for i, file_info in enumerate(input_files):
            print(f"\n[{i + 1}/{len(input_files)}] Processing {file_info.audio_path}...")
            
            try:
                # Process file
                result = self.processor.process_single_file(
                    file_info.audio_path,
                    file_info.ground_truth_path,
                    str(test_dir)
                )
                
                # Move dev/train files to appropriate directories
                if self.config.create_splits:
                    self._organize_split_files(result, dev_dir, train_dir)
                
                # Update batch statistics for all sets
                self._update_batch_stats(self.batch_stats, result, file_info)
                
                print(f"  ✓ Processed successfully")
                
            except Exception as e:
                print(f"  ✗ Error processing file: {e}")
                error_detail = {
                    "filename": Path(file_info.audio_path).name,
                    "error": str(e)
                }
                self.batch_stats.test_stats.file_details.append(error_detail)
        
        # Calculate aggregate statistics
        self._finalize_batch_stats(self.batch_stats)
        
        # Create comprehensive summary CSV
        self._create_summary_csv(self.batch_stats, output_dir)
        
        return self.batch_stats
    
    def _organize_split_files(self, result: ProcessingResult, 
                            dev_dir: Path, train_dir: Path) -> None:
        """Move dev/train files to appropriate directories."""
        if result.dev_output:
            dev_src = Path(result.dev_output)
            dev_dest = dev_dir / dev_src.name
            if dev_src.exists():
                shutil.move(str(dev_src), str(dev_dest))
        
        if result.train_output:
            train_src = Path(result.train_output)
            train_dest = train_dir / train_src.name
            if train_src.exists():
                shutil.move(str(train_src), str(train_dest))
    
    def _update_batch_stats(self, batch_stats: BatchStats, 
                          result: ProcessingResult, 
                          file_info: FileProcessingInfo) -> None:
        """Update batch statistics with results from all sets (TEST/DEV/TRAIN)."""
        # Update original duration (common for all sets)
        batch_stats.total_original_duration += result.original_duration_hours
        
        # Process TEST set
        self._process_set_stats(
            batch_stats.test_stats,
            "TEST",
            result.balanced_audio,
            result.balanced_timestamps,
            result.balanced_speech_ms,
            result.balanced_non_speech_ms,
            file_info,
            result.original_duration_hours
        )
        
        # Process DEV set if available
        if result.dev_audio and len(result.dev_audio) > 0:
            dev_speech_ms = sum(ts.duration_sec * 1000 for ts in result.dev_timestamps if ts.type == "speech")
            dev_non_speech_ms = sum(ts.duration_sec * 1000 for ts in result.dev_timestamps if ts.type == "non-speech")
            
            self._process_set_stats(
                batch_stats.dev_stats,
                "DEV",
                result.dev_audio,
                result.dev_timestamps,
                dev_speech_ms,
                dev_non_speech_ms,
                file_info,
                result.original_duration_hours
            )
        
        # Process TRAIN set if available
        if result.train_audio and len(result.train_audio) > 0:
            train_speech_ms = sum(ts.duration_sec * 1000 for ts in result.train_timestamps if ts.type == "speech")
            train_non_speech_ms = sum(ts.duration_sec * 1000 for ts in result.train_timestamps if ts.type == "non-speech")
            
            self._process_set_stats(
                batch_stats.train_stats,
                "TRAIN",
                result.train_audio,
                result.train_timestamps,
                train_speech_ms,
                train_non_speech_ms,
                file_info,
                result.original_duration_hours
            )
    
    def _process_set_stats(self, set_stats: SetStats, 
                         set_type: str,
                         audio: AudioSegment,
                         timestamps: List[TimestampMapping],
                         speech_ms: float,
                         non_speech_ms: float,
                         file_info: FileProcessingInfo,
                         original_duration_hours: float) -> None:
        """Process statistics for a specific set type (TEST/DEV/TRAIN)."""
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
            ratio_accuracy = min(speech_ratio, 1 - speech_ratio) * 2
            self.batch_stats.speech_ratio_accuracy.append(ratio_accuracy)
        
        # Analyze continuity
        continuity_stats = ContinuityAnalyzer.analyze_continuity(timestamps)
        
        # Update continuity stats
        set_stats.continuity_stats.total_segments += continuity_stats.total_segments
        set_stats.continuity_stats.continuous_segments += continuity_stats.continuous_segments
        set_stats.continuity_stats.num_sequences += continuity_stats.num_sequences
        set_stats.continuity_stats.longest_sequence = max(
            set_stats.continuity_stats.longest_sequence, 
            continuity_stats.longest_sequence
        )
        
        # Create file details
        speech_ratio_val = speech_hours / duration_hours if duration_hours > 0 else 0
        file_stats = {
            "filename": Path(file_info.audio_path).name,
            "set_type": set_type,
            "original_hours": original_duration_hours,
            "duration_hours": duration_hours,
            "speech_hours": speech_hours,
            "speech_ratio": speech_ratio_val,
            "non_speech_hours": non_speech_hours,
            "continuous_segments": continuity_stats.continuous_segments,
            "total_segments": continuity_stats.total_segments,
            "num_sequences": continuity_stats.num_sequences,
            "longest_sequence": continuity_stats.longest_sequence,
        }
        
        set_stats.file_details.append(file_stats)
    
    def _finalize_batch_stats(self, batch_stats: BatchStats) -> None:
        """Calculate final aggregate statistics for all sets."""
        # Process each set type
        for set_type in ["TEST", "DEV", "TRAIN"]:
            set_stats = batch_stats.stats_by_type(set_type)
            
            # Calculate average sequence length if we have sequences
            if set_stats.continuity_stats.num_sequences > 0:
                set_stats.continuity_stats.avg_sequence_length = (
                    set_stats.continuity_stats.continuous_segments /
                    set_stats.continuity_stats.num_sequences
                )
    
    def _create_summary_csv(self, batch_stats: BatchStats, output_dir: str) -> None:
        """Create a CSV summary file that includes all sets."""
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
                    silence_hours = total_hours - speech_hours
                    
                    writer.writerow({
                        "Filename": file_detail['filename'],
                        "SplitType": set_type,
                        "OriginalDuration(h)": f"{file_detail.get('original_hours', 0):.4f}",
                        "OutputDuration(h)": f"{total_hours:.4f}",
                        "SpeechDuration(h)": f"{speech_hours:.4f}",
                        "SpeechPercentage": f"{speech_percentage:.2f}",
                        "SilenceDuration(h)": f"{silence_hours:.4f}",
                        "SilencePercentage": f"{100 - speech_percentage:.2f}",
                        "TotalSegments": file_detail.get('total_segments', 0),
                        "ContinuousSegments": file_detail.get('continuous_segments', 0),
                        "Sequences": file_detail.get('num_sequences', 0),
                        "LongestSequence": file_detail.get('longest_sequence', 0)
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
        """
        Main entry point for processing a directory of audio files.
        
        Args:
            input_dir: Input directory containing 'audio' and 'ground_truth' subdirectories
            output_dir: Output directory for processed files
            
        Returns:
            Dictionary with batch processing statistics
        """
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
        target_hours=0.135,
        speech_padding_ms=200,
        create_splits=True,
        dev_ratio=0.2,
        silence_reserve_ratio=0.4
    )
    
    # load config into pipeline
    pipeline = AudioProcessingPipeline(config)
    
    # directory for input and output, input directory expects an audio and corresponding ground_truth folder within the input_data folder
    stats = pipeline.process_directory(
        input_dir="input_data",
        output_dir="Recompiled_Output"
    )
    
    if stats:
        print(f"\nProcessed {stats.total_files} files")
        # print(f"Total output generated: {stats.total_balanced_duration:.2f} hours")


if __name__ == "__main__":
    main()
