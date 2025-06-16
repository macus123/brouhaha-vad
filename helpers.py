import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import random

def analyze_speech_silence_ratio(rttm_file, audio_duration):
    """Calculate speech-to-silence ratio from RTTM file"""
    speech_segments = []
    total_speech = 0
    
    with open(rttm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            dur = float(parts[4])
            speech_segments.append((start, start + dur))
            total_speech += dur
    
    total_silence = audio_duration - total_speech
    ratio = total_speech / total_silence if total_silence > 0 else float('inf')
    
    return {
        'speech_duration': total_speech,
        'silence_duration': total_silence,
        'ratio': ratio,
        'speech_percentage': (total_speech / audio_duration) * 100,
        'speech_segments': speech_segments
    }

def extract_speech_silence_segments(audio_file, rttm_file, output_dir):
    """Extract speech and silence segments to separate files"""
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Get speech segments
    speech_segments = []
    with open(rttm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            dur = float(parts[4])
            speech_segments.append((start, start + dur))
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract speech segments
    speech_audio = []
    for i, (start, end) in enumerate(speech_segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        speech_audio.append(segment)
        sf.write(output_dir / f"speech_{i}.wav", segment, sr)
    
    # Calculate silence segments (gaps between speech)
    silence_segments = []
    last_end = 0
    for start, end in speech_segments:
        if start > last_end:
            silence_segments.append((last_end, start))
        last_end = end
    if duration > last_end:
        silence_segments.append((last_end, duration))
    
    # Extract silence segments
    silence_audio = []
    for i, (start, end) in enumerate(silence_segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        silence_audio.append(segment)
        sf.write(output_dir / f"silence_{i}.wav", segment, sr)
    
    return {
        'speech_files': [output_dir / f"speech_{i}.wav" for i in range(len(speech_segments))],
        'silence_files': [output_dir / f"silence_{i}.wav" for i in range(len(silence_segments))]
    }

def create_representative_audio_bucketed(speech_files, silence_files, output_file, target_ratio=1.0):
    """Create a shorter audio file with specific speech:silence ratio through intelligent selection"""
    # Load original audio data with timestamps
    audio_segments = []
    sr = None
    
    # Process speech segments - keep all of them
    for i, file in enumerate(speech_files):
        y, sr_file = librosa.load(str(file), sr=None)
        file_path = Path(file)
        index = int(file_path.stem.split('_')[1])
        audio_segments.append({
            'type': 'speech',
            'audio': y,
            'index': index,
            'duration': len(y) / sr_file
        })
        if sr is None:
            sr = sr_file
    
    # Process silence segments
    silence_segments = []
    for i, file in enumerate(silence_files):
        y, _ = librosa.load(str(file), sr=sr)
        file_path = Path(file)
        index = int(file_path.stem.split('_')[1])
        silence_segments.append({
            'type': 'silence',
            'audio': y,
            'index': index,
            'duration': len(y) / sr
        })
    
    # Calculate total durations
    speech_duration = sum(s['duration'] for s in audio_segments if s['type'] == 'speech')
    silence_duration = sum(s['duration'] for s in silence_segments)
    
    # Target silence duration based on ratio
    target_silence_duration = speech_duration / target_ratio
    
    # If we need to reduce silence, use intelligent selection
    if target_silence_duration < silence_duration:
        # Sort silence segments by duration (longest first for better selection)
        silence_segments.sort(key=lambda x: x['duration'], reverse=True)
        
        # Calculate how much silence to keep (as a percentage)
        keep_ratio = target_silence_duration / silence_duration
        
        # Bucket silence segments by length
        short_silences = [s for s in silence_segments if s['duration'] < 0.3]  # < 300ms
        medium_silences = [s for s in silence_segments if 0.3 <= s['duration'] < 1.0]  # 300ms-1s
        long_silences = [s for s in silence_segments if s['duration'] >= 1.0]  # > 1s
        
        # Strategy: Keep all short silences, trim long silences more aggressively
        silence_to_keep = []
        
        # Keep all short silences (pauses between words)
        silence_to_keep.extend(short_silences)
        short_duration = sum(s['duration'] for s in short_silences)
        
        # Calculate remaining silence needed
        remaining_silence = target_silence_duration - short_duration
        
        # If we need more silence, select from medium segments
        if remaining_silence > 0 and medium_silences:
            # Sort by duration to use shorter ones first
            medium_silences.sort(key=lambda x: x['duration'])
            medium_total = sum(s['duration'] for s in medium_silences)
            
            if medium_total <= remaining_silence:
                # We can keep all medium silences
                silence_to_keep.extend(medium_silences)
                remaining_silence -= medium_total
            else:
                # Keep as many as we can fit
                for s in medium_silences:
                    if s['duration'] <= remaining_silence:
                        silence_to_keep.append(s)
                        remaining_silence -= s['duration']
        
        # If we still need more silence, trim from long segments
        if remaining_silence > 0 and long_silences:
            # Sort by index to maintain temporal order
            long_silences.sort(key=lambda x: x['index'])
            
            for s in long_silences:
                # We'll keep a portion of each long silence
                # The portion depends on how much silence we still need
                keep_duration = min(remaining_silence, s['duration'] * 0.5)  # Keep up to 50% of long silences
                if keep_duration > 0:
                    # Take the middle portion of the silence
                    samples_to_keep = int(keep_duration * sr)
                    middle_start = (len(s['audio']) - samples_to_keep) // 2
                    middle_end = middle_start + samples_to_keep
                    
                    trimmed_audio = s['audio'][middle_start:middle_end]
                    silence_to_keep.append({
                        'type': 'silence',
                        'audio': trimmed_audio,
                        'index': s['index'],
                        'duration': keep_duration
                    })
                    remaining_silence -= keep_duration
        
        # Replace our silence segments with the selected ones
        silence_segments = silence_to_keep
    
    # Combine all segments and sort by original order
    combined_segments = audio_segments + silence_segments
    combined_segments.sort(key=lambda x: x['index'])
    
    # Concatenate audio in correct order
    final_audio = [segment['audio'] for segment in combined_segments]
    final_audio_array = np.concatenate(final_audio)
    
    # Calculate achieved metrics
    final_speech_duration = sum(s['duration'] for s in combined_segments if s['type'] == 'speech')
    final_silence_duration = sum(s['duration'] for s in combined_segments if s['type'] == 'silence')
    
    # Write to file
    sf.write(output_file, final_audio_array, sr)
    
    return {
        'output_file': output_file,
        'target_ratio': target_ratio,
        'achieved_ratio': final_speech_duration / (final_silence_duration if final_silence_duration > 0 else 1),
        'speech_duration': final_speech_duration,
        'silence_duration': final_silence_duration,
        'total_duration': final_speech_duration + final_silence_duration
    }