"""
Example demonstrating tempo detection functionality.

This script shows how to use the tempo detection functions from the
basic-audio-analytics package to estimate the tempo (BPM) of audio files.

Usage:
    python tempo_detection_example.py /path/to/audio_file.mp3

Dependencies:
    - basic_audio_analytics
    - librosa
    - matplotlib
    - numpy
"""

import sys
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basic_audio_analytics.tempo import detect_tempo, analyze_tempo_distribution

def main():
    """Run the tempo detection example."""
    # Check if an audio file was provided
    if len(sys.argv) < 2:
        print("Please provide an audio file path.")
        print("Usage: python tempo_detection_example.py /path/to/audio_file.mp3")
        return
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        return
    
    print(f"Analyzing audio file: {audio_file}")
    
    # Load audio file
    print("Loading audio file...")
    y, sr = librosa.load(audio_file, sr=None)
    
    # Simple tempo detection with visualization
    print("Detecting tempo...")
    tempo = detect_tempo(y=y, sr=sr, visualize=True)
    print(f"Detected tempo: {tempo:.1f} BPM")
    
    # Get multiple tempo candidates
    tempo_info = detect_tempo(y=y, sr=sr, n_tempi=3)
    print("\nMultiple tempo candidates:")
    for i, (t, c) in enumerate(zip(tempo_info['tempo'], tempo_info['confidence']), 1):
        print(f"  Candidate {i}: {t:.1f} BPM (confidence: {c:.3f})")
    
    # Analyze tempo distribution for longer files
    duration = len(y) / sr
    if duration > 30:  # Only for longer audio files
        print("\nAnalyzing tempo distribution...")
        tempo_analysis = analyze_tempo_distribution(
            y=y, sr=sr, frame_length=5, visualize=True
        )
        
        print(f"Average tempo: {tempo_analysis['avg_tempo']:.1f} BPM")
        print(f"Tempo range: {tempo_analysis['min_tempo']:.1f} - {tempo_analysis['max_tempo']:.1f} BPM")
        print(f"Tempo standard deviation: {tempo_analysis['std_tempo']:.2f} BPM")

if __name__ == "__main__":
    main()
