create_sample_dataset.ipynb:
pulls and splits atco2 audio dataset into VAD_Input/Audio/{DEV, TEST, TRAIN} splits
pulls ground truths for each of the .wav files into the following format: (speech timestamp start) (speech timestamp end) (text representation of speech)

split_seg.py:
parses and matches ground truth to corresponding .wav file based on file name and runs the following operations:
1. separates speech and non-speech portions based on ground truth annotation files
2. extracts all speech segments from a .wav file, concatenating them into a single .wav file
3. extracts all non-speech segments, concatenating them into another .wav
4. notes metadata consisting of both speech and silence timestamps in .txt format
5. handles empty audio segments by inserting minimal silence (eg. <0s of silence)

main.py:
1. processes the extracted non-speech segments with brouhaha

expected outputs from extract_noise_profiles.py:
Noise_Profiles/
├── low_frequency/         # Rumbles, HVAC sounds (20Hz-200Hz)
│   ├── standard_1s_concat.wav
│   ├── standard_2s_concat.wav
│   └── standard_5s_synthetic.wav
├── mid_frequency/         # Human activity, traffic (200Hz-2kHz)
│   └── [similar files]
├── high_frequency/        # Hisses, electronic whines (2kHz+)
│   └── [similar files]
├── broadband/             # Mixed frequency noise
│   └── [similar files]
├── noise_summary.png      # Visual breakdown of noise types
└── noise_summary.txt      # Statistical analysis of noise

from create_noise_library.py:
Noise_Library/
├── standard_noise_-20dB.wav   # Very loud noise sample
├── standard_noise_-10dB.wav   # Strong noise sample
├── standard_noise_0dB.wav     # Moderate noise sample
├── standard_noise_10dB.wav    # Subtle noise sample
├── mixed_snr_noise_library.wav  # All SNR levels in one file
└── README.txt                   # Documentation

noise_merge.py:
1. Loads a speech file
2. Adds noise padding before and after the speech (length controlled by padding_ratio param)
3. Mixes noise into the speech at a controlled SNR level
4. Creates visualisations showing each component

padding_ratio param: changes how much noise padding to add, eg. 0.5 = 50% of speech length
target_snr param: controls how prominent the noise is during the speech