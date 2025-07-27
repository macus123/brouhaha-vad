from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pydub import AudioSegment


@dataclass
class AudioSegmentInfo:
    """Represents an audio segment with timing and type information."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    type: str     # "speech" or "non-speech"
    source_file: str = None  # Source file path
    file_id: str = None      # Unique file identifier
    duration: float = field(init=False)  # Duration in milliseconds
    text: str = ""
    
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
    source_file: str = None  
    file_id: str = None     
    text: str = ""
    
@dataclass
class ProcessingConfig:
    def __init__(self,
                 target_hours_speech: float = 0.0675,
                 target_hours_silence: float = 0.0675, 
                 speech_padding_ms: float = 200.0,
                 create_splits: bool = True,
                 dev_ratio: float = 0.2,
                 silence_reserve_ratio: float = 0.4,
                 default_sample_rate: int = 16000):
        self.target_hours_speech = target_hours_speech
        self.target_hours_silence = target_hours_silence
        self.speech_padding_ms = speech_padding_ms
        self.create_splits = create_splits
        self.dev_ratio = dev_ratio
        self.silence_reserve_ratio = silence_reserve_ratio
        self.default_sample_rate = default_sample_rate


@dataclass
class FileProcessingInfo:
    """Information about a file to be processed."""
    audio_path: str
    ground_truth_path: str
    set_type: str = "TEST"


@dataclass
class ProcessingResult:
    """Results from processing audio files."""
    test_output: str  # Now directory path instead of single file
    test_ground_truth: str  # Now directory path instead of single file
    original_duration_hours: float
    test_audio: Optional[AudioSegment] = None  # None for temporal sequence output
    test_timestamps: Optional[List[TimestampMapping]] = None  # None for temporal sequence output
    test_speech_ms: float = 0.0
    test_non_speech_ms: float = 0.0
    dev_output: Optional[str] = None  # Directory path for temporal sequences
    train_output: Optional[str] = None  # Directory path for temporal sequences
    dev_audio: Optional[AudioSegment] = None  # None for temporal sequence output
    train_audio: Optional[AudioSegment] = None  # None for temporal sequence output
    dev_timestamps: Optional[List[TimestampMapping]] = None  # None for temporal sequence output
    train_timestamps: Optional[List[TimestampMapping]] = None  # None for temporal sequence output
    dev_ground_truth: Optional[str] = None     # Dev ground truth directory path
    train_ground_truth: Optional[str] = None   # Train ground truth directory path
    files_used: List[str] = field(default_factory=list)  # Source files used


@dataclass
class SetStats:
    """Statistics for a specific set (TEST/DEV/TRAIN)."""
    total_duration: float = 0.0
    speech_duration: float = 0.0
    non_speech_duration: float = 0.0
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
