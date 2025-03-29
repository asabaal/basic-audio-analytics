#!/usr/bin/env python3
"""
Vocal Processing Comparison Example for FL Studio
------------------------------------------------
This script demonstrates how to use the vocal processing comparison module
to analyze the differences between pre-processed and post-processed vocal files
from FL Studio.

Usage:
    python vocal_processing_comparison_example.py pre_file.wav post_file.wav

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

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import functions from our package
from src.basic_audio_analytics.vocal_processing_comparison import (
    compare_vocal_files,
    calculate_dynamic_range,
    analyze_frequency_distribution,
    analyze_stereo_width,
    calculate_harmonic_content,
    analyze_reverb
)

def main():
    """Main function to run the vocal processing comparison example"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compare pre and post-processed vocal files from FL Studio'
    )
    parser.add_argument('pre_file', type=str, help='Path to the pre-processed audio file')
    parser.add_argument('post_file', type=str, help='Path to the post-processed audio file')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.pre_file, args.post_file]:
        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' not found!")
            return 1
    
    print(f"\n{'='*60}")
    print(f"Vocal Processing Comparison: {os.path.basename(args.pre_file)} vs {os.path.basename(args.post_file)}")
    print(f"{'='*60}\n")
    
    # Run comparison analysis
    print("Analyzing audio files...")
    results = compare_vocal_files(
        args.pre_file,
        args.post_file,
        plot=not args.no_plot
    )
    
    # Print dynamic range analysis
    print("\nDynamic Range Analysis:")
    print(f"{'='*25}")
    print(f"Pre-processing:  {results['dynamic_range']['pre']:.1f} dB")
    print(f"Post-processing: {results['dynamic_range']['post']:.1f} dB")
    print(f"Change:          {results['dynamic_range']['change']:.1f} dB")
    print(f"Analysis:        {results['text_analysis']['dynamic_range']}")
    
    # Print frequency balance analysis
    print("\nFrequency Balance Analysis:")
    print(f"{'='*30}")
    print(f"Analysis: {results['text_analysis']['frequency_balance']}")
    
    print("\nPer-band Frequency Changes:")
    for band, change_db in results['frequency_bands']['change_db'].items():
        if abs(change_db) >= 1.0:  # Only show significant changes
            direction = "+" if change_db > 0 else ""
            print(f"  {band.replace('_', ' '):12}: {direction}{change_db:.1f} dB")
    
    # Print harmonic content analysis
    print("\nHarmonic Content Analysis:")
    print(f"{'='*28}")
    print(f"Analysis: {results['text_analysis']['harmonic_content']}")
    print(f"Spectral Balance: {results['text_analysis']['spectral_balance']}")
    
    # Print detailed harmonic metrics
    print("\nHarmonic Metrics:")
    print(f"  {'Metric':20} {'Pre':>10} {'Post':>10} {'Change':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    
    h_pre = results['harmonic_content']['pre']
    h_post = results['harmonic_content']['post']
    
    print(f"  {'Harmonic Ratio':20} {h_pre['harmonic_ratio']:10.2f} {h_post['harmonic_ratio']:10.2f} {results['harmonic_content']['harmonic_ratio_change']:+10.2f}")
    print(f"  {'Spectral Centroid':20} {h_pre['spectral_centroid']:10.1f} {h_post['spectral_centroid']:10.1f} {results['harmonic_content']['spectral_centroid_change']:+10.1f}")
    print(f"  {'Spectral Flatness':20} {h_pre['spectral_flatness']*100:10.2f}% {h_post['spectral_flatness']*100:10.2f}% {(h_post['spectral_flatness']-h_pre['spectral_flatness'])*100:+10.2f}%")
    
    # Print reverb analysis
    print("\nReverb Analysis:")
    print(f"{'='*20}")
    print(f"Analysis: {results['text_analysis']['reverb']}")
    
    r_pre = results['reverb']['pre']
    r_post = results['reverb']['post']
    
    print(f"  Decay Time: {r_pre['decay_time']:.2f}s → {r_post['decay_time']:.2f}s ({results['reverb']['decay_time_change']:+.2f}s)")
    print(f"  Reverb Level: {r_pre['reverb_level']:.2f} → {r_post['reverb_level']:.2f} ({r_post['reverb_level']-r_pre['reverb_level']:+.2f})")
    
    # Print stereo analysis if applicable
    if results['stereo_width']['pre'] is not None and results['stereo_width']['post'] is not None:
        print("\nStereo Width Analysis:")
        print(f"{'='*25}")
        print(f"Analysis: {results['text_analysis']['stereo_width']}")
        print(f"  Pre-processing:  {results['stereo_width']['pre']*100:.1f}%")
        print(f"  Post-processing: {results['stereo_width']['post']*100:.1f}%")
        change = results['stereo_width']['change']*100
        print(f"  Change:          {change:+.1f}%")
    
    # Summary
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"{'='*20}")
    
    effects_found = []
    
    # Check for compression
    if results['dynamic_range']['change'] < -2:
        effects_found.append(f"Compression ({abs(results['dynamic_range']['change']):.1f}dB reduction)")
    
    # Check for EQ
    significant_eq_changes = False
    for band, change_db in results['frequency_bands']['change_db'].items():
        if abs(change_db) >= 3.0:
            significant_eq_changes = True
            break
    
    if significant_eq_changes:
        effects_found.append("Significant EQ processing")
    
    # Check for reverb
    if results['reverb']['decay_time_change'] > 0.2:
        effects_found.append(f"Reverb (decay time: {results['reverb']['post']['decay_time']:.2f}s)")
    
    # Check for saturation/distortion
    if results['harmonic_content']['harmonic_ratio_change'] > 0.2:
        effects_found.append("Saturation/harmonic enhancement")
    
    # Check for stereo effects
    if results['stereo_width']['change'] is not None and results['stereo_width']['change'] > 0.1:
        effects_found.append(f"Stereo widening ({results['stereo_width']['change']*100:.1f}%)")
    
    if effects_found:
        print("Detected effects:")
        for effect in effects_found:
            print(f"  • {effect}")
    else:
        print("No significant processing detected or minimal effects applied.")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
