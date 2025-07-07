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
    source_file: str = None  # Source file path
    file_id: str = None      # Source file identifier


@dataclass
class ContinuityStats:
    """Statistics about segment continuity."""
    total_segments: int = 0
    continuous_segments: int = 0
    cross_file_transitions: int = 0
    files_used: int = 0


@dataclass
class ProcessingConfig:
    """Configuration for audio processing."""
    target_hours: float = 1.0
    speech_ratio: float = 0.5  # 0.5 = 50% speech, 50% silence
    speech_padding_ms: int = 200
    create_splits: bool = True
    dev_ratio: float = 0.2
    silence_reserve_ratio: float = 0.4
    enable_multi_file_stitching: bool = True


@dataclass
class FileProcessingInfo:
    """Information about a file to be processed."""
    audio_path: str
    ground_truth_path: str
    set_type: str = "TEST"


@dataclass
class ProcessingResult:
    """Results from processing audio files."""
    test_output: str
    test_ground_truth: str
    original_duration_hours: float
    test_audio: AudioSegment
    test_timestamps: List[TimestampMapping]
    test_speech_ms: float
    test_non_speech_ms: float
    dev_output: Optional[str] = None
    train_output: Optional[str] = None
    dev_audio: Optional[AudioSegment] = None
    train_audio: Optional[AudioSegment] = None
    dev_timestamps: List[TimestampMapping] = field(default_factory=list)
    train_timestamps: List[TimestampMapping] = field(default_factory=list)
    dev_ground_truth: Optional[str] = None     # Dev ground truth file path
    train_ground_truth: Optional[str] = None   # Train ground truth file path
    files_used: List[str] = field(default_factory=list)  # Source files used


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
