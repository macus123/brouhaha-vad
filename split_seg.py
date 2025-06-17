from pathlib import Path
from pydub import AudioSegment
import os
import torch
import torchaudio
import numpy as np

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

def save_audio_safely(audio_segment, file_path, sample_rate=16000):
    """Safely save audio using torchaudio for better format control."""
    # Check if the audio segment is empty
    if len(audio_segment) == 0:
        print(f"Warning: Empty audio segment for {file_path}")
        # Create a minimal silent segment (10ms) to avoid empty file errors
        audio_segment = AudioSegment.silent(duration=10, frame_rate=sample_rate)
    
    # Convert pydub AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    
    # Safety check for empty arrays
    if samples.size == 0:
        print(f"Warning: Empty samples array for {file_path}")
        samples = np.zeros(sample_rate // 100, dtype=np.float32)  # 10ms of silence
    
    # Convert to float32 and normalize to [-1, 1]
    if audio_segment.sample_width == 2:  # 16-bit audio
        samples = samples.astype(np.float32) / 32768.0
    elif audio_segment.sample_width == 4:  # 32-bit audio
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32) / (2**(audio_segment.sample_width*8-1))
    
    # Handle stereo to mono conversion if needed
    if audio_segment.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    
    # Safe normalization - check if array has non-zero elements first
    if samples.size > 0:
        max_abs_val = max(abs(samples.max()) if samples.size > 0 else 0, 
                          abs(samples.min()) if samples.size > 0 else 0)
        if max_abs_val > 1.0 and max_abs_val > 0:
            samples = samples / max_abs_val
    
    # Convert to torch tensor
    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    
    # Validate waveform
    if torch.isnan(waveform).any():
        print(f"Warning: NaN values found in waveform for {file_path}")
        # Replace NaNs with zeros
        waveform = torch.nan_to_num(waveform)
    
    try:
        # Save audio with explicit format settings
        torchaudio.save(
            str(file_path), 
            waveform, 
            sample_rate=sample_rate,
            encoding='PCM_S', 
            bits_per_sample=16
        )
        
        # Verify the saved file
        try:
            verification, sr = torchaudio.load(str(file_path))
            if verification.size(1) == 0:
                print(f"Warning: Verification failed, empty file for {file_path}")
                return False
        except Exception as e:
            print(f"Warning: Could not verify saved file for {file_path}: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error saving audio {file_path}: {e}")
        return False

def split_audio(audio_path, gt_path, output_base_dir):
    """Split audio file into two files: one with all speech segments, one with all non-speech segments."""
    print(f"Processing: {audio_path}")
    
    # Extract the split name (TRAIN, TEST, or DEV)
    split_name = Path(audio_path).parent.name
    
    # Create output directories for this split
    output_dir = Path(output_base_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create separate speech and non-speech subfolders
    speech_dir = output_dir / "speech"
    non_speech_dir = output_dir / "non_speech"
    speech_dir.mkdir(parents=True, exist_ok=True)
    non_speech_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ground truth directory
    ground_truth_dir = Path(output_base_dir) / "Ground"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio and ground truth
        audio = AudioSegment.from_file(audio_path)
        total_duration = len(audio) / 1000.0  # Convert to seconds
        speech_segments = read_ground_truth(gt_path)
        non_speech_segments = get_non_speech_segments(speech_segments, total_duration)
        
        # Check for empty segments and add a tiny placeholder if needed
        if not speech_segments:
            print(f"Warning: No speech segments found for {audio_path}")
            speech_segments = [(0.0, 0.01)]  # Add tiny segment to avoid empty file
            
        if not non_speech_segments:
            print(f"Warning: No non-speech segments found for {audio_path}")
            non_speech_segments = [(0.0, 0.01)]  # Add tiny segment to avoid empty file
        
        # Create combined speech segment
        speech_audio = AudioSegment.empty()
        for start, end in speech_segments:
            segment = audio[int(start*1000):int(end*1000)]
            speech_audio += segment
        
        # Create combined non-speech segment
        non_speech_audio = AudioSegment.empty()
        for start, end in non_speech_segments:
            segment = audio[int(start*1000):int(end*1000)]
            non_speech_audio += segment
        
        # Export the combined files with robust saving
        speech_output_path = speech_dir / f"{Path(audio_path).stem}.wav"
        non_speech_output_path = non_speech_dir / f"{Path(audio_path).stem}.wav"
        
        # Use the safer method to save audio files
        speech_saved = save_audio_safely(speech_audio, speech_output_path, sample_rate=audio.frame_rate)
        non_speech_saved = save_audio_safely(non_speech_audio, non_speech_output_path, sample_rate=audio.frame_rate)
        
        if not speech_saved or not non_speech_saved:
            print(f"Warning: Failed to save one or both audio files for {audio_path}")
        
        # Create metadata files in Ground directory
        speech_meta_path = ground_truth_dir / f"{Path(audio_path).stem}_speech.txt"
        non_speech_meta_path = ground_truth_dir / f"{Path(audio_path).stem}_non_speech.txt"
        
        with open(speech_meta_path, 'w') as f:
            for i, (start, end) in enumerate(speech_segments):
                f.write(f"{start}\t{end}\tSpeech segment {i+1}\n")
                
        with open(non_speech_meta_path, 'w') as f:
            for i, (start, end) in enumerate(non_speech_segments):
                f.write(f"{start}\t{end}\tNon-speech segment {i+1}\n")
        
        # Sanity check to ensure no duplication
        original_length = len(audio)
        combined_length = len(speech_audio) + len(non_speech_audio)
        
        if abs(original_length - combined_length) > 10:  # Allow small rounding differences
            print(f"Warning: Length mismatch for {audio_path}")
            print(f"Original: {original_length}ms, Combined: {combined_length}ms")
            
    except Exception as e:
        print(f"Error processing {audio_path}:{str(e)}")

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
            # Check if ground truth exists
            if audio_file.stem in ground_truth_files:
                gt_file = ground_dir / f"{audio_file.stem}.txt"
                split_audio(str(audio_file), str(gt_file), output_base_dir)
            else:
                print(f"Warning: No ground truth found for {audio_file.name}")

if __name__ == "__main__":
    main()