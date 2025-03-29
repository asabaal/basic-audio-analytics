#!/usr/bin/env python3
"""
Frequency Analysis Example for FL Studio Vocals
----------------------------------------------
This script analyzes an audio file to identify its primary frequencies
and provides recommendations for EQ settings in FL Studio.

Usage:
    python frequency_analysis_example.py path/to/audio_file.wav

Requirements:
    - librosa
    - numpy
    - matplotlib
    - scipy
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import functions from our package
from src.basic_audio_analytics.frequency_analysis import identify_primary_frequencies, analyze_audio_for_fl_studio

def main():
    """Main function to run the frequency analysis example"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze primary frequencies in an audio file')
    parser.add_argument('audio_file', type=str, help='Path to the audio file to analyze')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--min-freq', type=int, default=80, help='Minimum frequency to analyze (Hz)')
    parser.add_argument('--max-freq', type=int, default=8000, help='Maximum frequency to analyze (Hz)')
    parser.add_argument('--num-peaks', type=int, default=10, help='Number of frequency peaks to identify')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found!")
        return 1
    
    print(f"\n{'='*50}")
    print(f"Analyzing audio file: {args.audio_file}")
    print(f"{'='*50}\n")
    
    # Run frequency analysis
    print("Performing primary frequency analysis...")
    results = identify_primary_frequencies(
        args.audio_file,
        n_frequencies=args.num_peaks,
        min_frequency=args.min_freq,
        max_frequency=args.max_freq,
        plot=not args.no_plot
    )
    
    # Print the results
    print("\nPrimary Frequencies:")
    print(f"{'='*20}")
    for i, (freq, mag) in enumerate(results["primary_frequencies"]):
        print(f"{i+1}. {freq:.1f} Hz (magnitude: {mag:.3f})")
    
    print("\nFrequency Distribution by Region:")
    print(f"{'='*30}")
    for region, freqs in results["frequency_regions"].items():
        if freqs:
            print(f"{region.upper()}: {', '.join([f'{f:.1f} Hz' for f in freqs])}")
        else:
            print(f"{region.upper()}: None detected")
    
    print("\nEQ Recommendations:")
    print(f"{'='*20}")
    for region, recommendation in results["eq_recommendations"].items():
        print(f"- {region.replace('_', ' ').title()}: {recommendation}")
    
    # Get FL Studio specific settings
    print("\nFL Studio Parametric EQ 2 Settings:")
    print(f"{'='*35}")
    fl_settings = analyze_audio_for_fl_studio(args.audio_file)
    for band, setting in fl_settings["fl_studio_eq_settings"].items():
        print(f"- {band}: {setting}")
    
    print(f"\n{'='*50}")
    print("Analysis complete!")
    print(f"{'='*50}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())