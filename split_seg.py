from pathlib import Path
from pydub import AudioSegment
import os

def read_ground_truth(gt_file):
    """Read ground truth segments from file."""
    segments = []
    with open(gt_file, 'r') as f:
        for line in f:
            try:
                start, end, _ = line.strip().split('\t', 2)
                segments.append((float(start), float(end)))
            except ValueError as e:
                print(f"Error parsing line in {gt_file}: {line}")
                continue
    return sorted(segments)

def get_non_speech_segments(speech_segments, total_duration):
    """Identify non-speech segments between speech segments."""
    non_speech = []
    current_time = 0.0
    
    for start, end in speech_segments:
        if current_time < start:
            non_speech.append((current_time, start))
        current_time = end
    
    # Add final non-speech segment if needed
    if current_time < total_duration:
        non_speech.append((current_time, total_duration))
    
    return non_speech

def split_audio(audio_path, gt_path, output_base_dir):
    """Split single audio file into speech/non-speech segments."""
    print(f"Processing: {audio_path}")
    
    # Create output directories
    output_dir = Path(output_base_dir) / Path(audio_path).parent.name
    speech_dir = output_dir / "speech"
    non_speech_dir = output_dir / "non_speech"
    speech_dir.mkdir(parents=True, exist_ok=True)
    non_speech_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio and ground truth
        audio = AudioSegment.from_file(audio_path)
        total_duration = len(audio) / 1000.0  # Convert to seconds
        speech_segments = read_ground_truth(gt_path)
        non_speech_segments = get_non_speech_segments(speech_segments, total_duration)
        
        # Extract speech segments
        for i, (start, end) in enumerate(speech_segments):
            segment = audio[int(start*1000):int(end*1000)]
            output_path = speech_dir / f"{Path(audio_path).stem}_speech_{i:03d}.wav"
            segment.export(str(output_path), format="wav")
        
        # Extract non-speech segments
        for i, (start, end) in enumerate(non_speech_segments):
            segment = audio[int(start*1000):int(end*1000)]
            output_path = non_speech_dir / f"{Path(audio_path).stem}_non_speech_{i:03d}.wav"
            segment.export(str(output_path), format="wav")
            
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")

def main():
    base_dir = Path("VAD_Input")
    output_base_dir = Path("VAD_Output")
    splits = ["DEV", "TEST", "TRAIN"]
    
    # First, get all ground truth files
    ground_truth_files = set(f.stem for f in Path(base_dir / "Ground").glob('*.txt'))
    
    for split in splits:
        audio_dir = base_dir / "Audio" / split
        ground_dir = base_dir / "Ground"
        
        # Process each audio file in the split
        for audio_file in audio_dir.glob("*.wav"):
            # Check if ground truth exists using the same logic as verify_file_matching
            if audio_file.stem in ground_truth_files:
                gt_file = ground_dir / f"{audio_file.stem}.txt"
                split_audio(str(audio_file), str(gt_file), output_base_dir)
            else:
                print(f"Warning: No ground truth found for {audio_file.name}")
        
        print(f"Ground truth files found: {len(ground_truth_files)}")
        print(f"First few ground truth files: {list(ground_truth_files)[:5]}")

if __name__ == "__main__":
    main()
    