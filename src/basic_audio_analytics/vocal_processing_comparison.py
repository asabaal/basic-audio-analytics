"""
Vocal Processing Comparison Module
----------------------------------
This module provides tools to analyze and compare pre-processed and post-processed
vocal recordings to quantify the effects of audio processing in FL Studio.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns
from .base import set_style

def calculate_dynamic_range(y):
    """
    Calculate the dynamic range of an audio signal.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
        
    Returns:
    --------
    float
        Dynamic range in dB
    """
    # Calculate RMS values in small frames
    frame_length = 2048
    hop_length = 512
    
    # Calculate RMS across frames
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB
    rms_db = 20 * np.log10(np.maximum(rms, 1e-7))  # Avoid log of zero
    
    # Calculate dynamic range (excluding silence)
    non_silent_frames = rms_db > -60  # Exclude frames below -60 dB
    if np.any(non_silent_frames):
        rms_db_non_silent = rms_db[non_silent_frames]
        dynamic_range = np.max(rms_db_non_silent) - np.min(rms_db_non_silent)
    else:
        dynamic_range = 0
        
    return dynamic_range

def analyze_frequency_distribution(y, sr):
    """
    Analyze the frequency distribution of an audio signal.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
        
    Returns:
    --------
    dict
        Frequency analysis results
    """
    # Compute the spectral features
    S = np.abs(librosa.stft(y))
    
    # Convert to dB scale
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Calculate mean spectrum across time
    mean_spectrum = np.mean(S, axis=1)
    
    # Normalize the spectrum
    mean_spectrum = mean_spectrum / np.max(mean_spectrum)
    
    # Define frequency bands
    bands = {
        "sub_bass": (20, 60),
        "bass": (60, 250),
        "low_mids": (250, 500),
        "mids": (500, 2000),
        "high_mids": (2000, 4000),
        "highs": (4000, 10000),
        "air": (10000, 20000)
    }
    
    # Calculate energy in each band
    band_energy = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Get indices for frequencies in band
        band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
        if len(band_indices) > 0:
            # Calculate energy as mean magnitude in band
            band_energy[band_name] = np.mean(mean_spectrum[band_indices])
        else:
            band_energy[band_name] = 0
    
    # Find peaks in the spectrum
    # Filter spectrum for higher frequency resolution
    freq_range = (freqs >= 80) & (freqs <= 8000)
    filtered_freqs = freqs[freq_range]
    filtered_spectrum = mean_spectrum[freq_range]
    
    # Find peaks
    peaks, _ = find_peaks(filtered_spectrum, height=0.1, distance=20)
    
    # Get top 10 peaks
    peak_magnitudes = filtered_spectrum[peaks]
    if len(peak_magnitudes) > 0:
        sorted_indices = np.argsort(peak_magnitudes)[::-1][:min(10, len(peak_magnitudes))]  # Get indices of top 10 peaks
        
        # Extract the primary frequency values and their magnitudes
        primary_freqs = filtered_freqs[peaks[sorted_indices]]
        primary_mags = filtered_spectrum[peaks[sorted_indices]]
    else:
        primary_freqs = np.array([])
        primary_mags = np.array([])
    
    return {
        "band_energy": band_energy,
        "primary_frequencies": list(zip(primary_freqs, primary_mags)),
        "mean_spectrum": mean_spectrum,
        "freqs": freqs
    }

def analyze_stereo_width(y):
    """
    Analyze the stereo width of an audio signal.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal (mono or stereo)
        
    Returns:
    --------
    float or None
        Stereo width (0-1) or None if mono
    """
    # Check if signal is stereo
    if len(y.shape) == 1:
        return None
    
    # Calculate mid and side signals
    mid = (y[0] + y[1]) / 2
    side = (y[0] - y[1]) / 2
    
    # Calculate RMS of mid and side
    mid_rms = np.sqrt(np.mean(mid ** 2))
    side_rms = np.sqrt(np.mean(side ** 2))
    
    # Calculate stereo width (side/mid ratio)
    if mid_rms > 0:
        stereo_width = side_rms / mid_rms
    else:
        stereo_width = 0
        
    return stereo_width

def calculate_harmonic_content(y, sr):
    """
    Calculate the harmonic content and analyze saturation.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
        
    Returns:
    --------
    dict
        Harmonic analysis results
    """
    # Calculate harmonic to noise ratio
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Calculate RMS of harmonic and percussive components
    harmonic_rms = np.sqrt(np.mean(y_harmonic ** 2))
    percussive_rms = np.sqrt(np.mean(y_percussive ** 2))
    
    # Calculate harmonic to percussive ratio
    if percussive_rms > 0:
        harmonic_ratio = harmonic_rms / percussive_rms
    else:
        harmonic_ratio = float('inf')
    
    # Calculate spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    
    # Calculate spectral flatness (indicates noise vs tonal content)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mean_flatness = np.mean(flatness)
    
    return {
        "harmonic_ratio": harmonic_ratio,
        "spectral_centroid": mean_centroid,
        "spectral_flatness": mean_flatness
    }

def analyze_reverb(y, sr):
    """
    Analyze the reverb/ambience in an audio signal.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
        
    Returns:
    --------
    dict
        Reverb analysis results
    """
    # Calculate the RMS envelope
    frame_length = 2048
    hop_length = 512
    
    # Get the RMS envelope
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find peaks in the RMS envelope
    peaks, _ = find_peaks(rms, height=0.1 * np.max(rms), distance=sr//hop_length//4)
    
    # If no peaks found, return default values
    if len(peaks) == 0:
        return {
            "decay_time": 0,
            "reverb_level": 0
        }
    
    # Analyze decay time after peaks
    decay_times = []
    for peak in peaks:
        # Get RMS values after peak
        if peak + 50 < len(rms):  # Ensure there's enough data after the peak
            post_peak = rms[peak:peak + 50]
            
            # Calculate decay rate
            try:
                # Convert to dB
                post_peak_db = 20 * np.log10(np.maximum(post_peak, 1e-7))
                
                # Linear regression to find decay rate
                times = np.arange(len(post_peak)) * hop_length / sr
                
                # Use only valid values for regression
                valid_indices = np.isfinite(post_peak_db)
                if np.sum(valid_indices) > 10:  # Need enough points
                    valid_times = times[valid_indices]
                    valid_db = post_peak_db[valid_indices]
                    
                    # Simple linear regression
                    polyfit = np.polyfit(valid_times, valid_db, 1)
                    slope = polyfit[0]
                    
                    # RT60 estimation: time for 60dB decay
                    if slope < 0:  # Only count negative slopes
                        decay_time = -60 / slope
                        if 0.1 < decay_time < 10:  # Reasonable range
                            decay_times.append(decay_time)
            except Exception as e:
                # Skip errors in regression
                continue
    
    # Calculate median decay time
    if decay_times:
        median_decay = np.median(decay_times)
    else:
        median_decay = 0
    
    # Estimate reverb level based on ratio of decay to attack
    reverb_level = 0
    if len(peaks) > 1:
        # Calculate average distance between peaks in seconds
        peak_distances = (peaks[1:] - peaks[:-1]) * hop_length / sr
        avg_distance = np.mean(peak_distances)
        
        # If decay time is significant relative to peak distance
        if avg_distance > 0:
            reverb_level = min(1.0, median_decay / (5 * avg_distance))
    
    return {
        "decay_time": median_decay,
        "reverb_level": reverb_level
    }

def plot_comparison_results(y_pre, y_post, sr, 
                           freq_analysis_pre, freq_analysis_post,
                           dynamic_range_pre, dynamic_range_post,
                           stereo_width_pre, stereo_width_post,
                           harmonic_analysis_pre, harmonic_analysis_post,
                           reverb_analysis_pre, reverb_analysis_post):
    """
    Plot comparison results between pre and post-processed vocals.
    
    Parameters:
    -----------
    Multiple parameters from the comparison analysis
    """
    # Set style for better visualization
    colors = set_style()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 20))
    
    # 1. Waveform Comparison
    ax1 = plt.subplot(5, 1, 1)
    librosa.display.waveshow(y_pre, sr=sr, alpha=0.7, color='#00A7E1', ax=ax1, label='Pre-processing')
    librosa.display.waveshow(y_post, sr=sr, alpha=0.7, color='#FFA400', ax=ax1, label='Post-processing')
    ax1.set_title('Waveform Comparison', color='white', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Amplitude', color='white')
    
    # Add dynamic range annotation
    textstr = f'Dynamic Range:\nPre: {dynamic_range_pre:.1f}dB\nPost: {dynamic_range_post:.1f}dB\nChange: {dynamic_range_post-dynamic_range_pre:.1f}dB'
    props = dict(boxstyle='round', facecolor='#1A1A1A', alpha=0.8, edgecolor='#404040')
    ax1.text(0.02, 0.97, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color='white')
    
    # 2. Frequency Spectrum Comparison
    ax2 = plt.subplot(5, 1, 2)
    ax2.semilogx(freq_analysis_pre["freqs"], freq_analysis_pre["mean_spectrum"], alpha=0.7, color='#00A7E1', label='Pre-processing')
    ax2.semilogx(freq_analysis_post["freqs"], freq_analysis_post["mean_spectrum"], alpha=0.7, color='#FFA400', label='Post-processing')
    ax2.set_title('Frequency Spectrum Comparison', color='white', fontsize=14)
    ax2.set_xlabel('Frequency (Hz)', color='white')
    ax2.set_ylabel('Magnitude', color='white')
    ax2.set_xlim(20, 20000)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    # 3. Frequency Band Energy Comparison
    bands = ["sub_bass", "bass", "low_mids", "mids", "high_mids", "highs", "air"]
    band_labels = ["Sub Bass\n20-60Hz", "Bass\n60-250Hz", "Low Mids\n250-500Hz", 
                  "Mids\n500-2kHz", "High Mids\n2-4kHz", "Highs\n4-10kHz", "Air\n10-20kHz"]
    
    pre_energies = [freq_analysis_pre["band_energy"].get(band, 0) for band in bands]
    post_energies = [freq_analysis_post["band_energy"].get(band, 0) for band in bands]
    
    ax3 = plt.subplot(5, 1, 3)
    x = np.arange(len(bands))
    width = 0.35
    
    ax3.bar(x - width/2, pre_energies, width, label='Pre-processing', color='#00A7E1', alpha=0.7)
    ax3.bar(x + width/2, post_energies, width, label='Post-processing', color='#FFA400', alpha=0.7)
    
    ax3.set_title('Frequency Band Energy Comparison', color='white', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(band_labels, color='white')
    ax3.set_ylabel('Relative Energy', color='white')
    ax3.legend(loc='upper right')
    
    # Calculate and annotate dB changes
    for i, (pre, post) in enumerate(zip(pre_energies, post_energies)):
        if pre > 0 and post > 0:
            change_db = 20 * np.log10(post / pre)
            if abs(change_db) >= 1.0:  # Only show significant changes
                color = '#00FF00' if change_db > 0 else '#FF6666'
                ax3.annotate(f"{change_db:.1f}dB", 
                            xy=(i, max(pre, post) + 0.02),
                            ha='center', va='bottom',
                            color=color, fontweight='bold')
    
    # 4. Harmonic and Spectral Analysis
    ax4 = plt.subplot(5, 1, 4)
    
    # Create bars for harmonic analysis
    harmonic_features = ["Harmonic Ratio", "Spectral Centroid/5000", "Flatness*100"]
    pre_values = [
        harmonic_analysis_pre["harmonic_ratio"] / 2,
        harmonic_analysis_pre["spectral_centroid"] / 5000,
        harmonic_analysis_pre["spectral_flatness"] * 100
    ]
    post_values = [
        harmonic_analysis_post["harmonic_ratio"] / 2,
        harmonic_analysis_post["spectral_centroid"] / 5000,
        harmonic_analysis_post["spectral_flatness"] * 100
    ]
    
    x = np.arange(len(harmonic_features))
    ax4.bar(x - width/2, pre_values, width, label='Pre-processing', color='#00A7E1', alpha=0.7)
    ax4.bar(x + width/2, post_values, width, label='Post-processing', color='#FFA400', alpha=0.7)
    
    ax4.set_title('Harmonic Content Analysis', color='white', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(harmonic_features, color='white')
    ax4.set_ylabel('Value (Normalized)', color='white')
    ax4.legend(loc='upper right')
    
    # 5. Reverb Analysis
    ax5 = plt.subplot(5, 1, 5)
    
    # Create bars for reverb analysis
    reverb_features = ["Decay Time (s)", "Reverb Level"]
    pre_values = [reverb_analysis_pre["decay_time"], reverb_analysis_pre["reverb_level"]]
    post_values = [reverb_analysis_post["decay_time"], reverb_analysis_post["reverb_level"]]
    
    x = np.arange(len(reverb_features))
    ax5.bar(x - width/2, pre_values, width, label='Pre-processing', color='#00A7E1', alpha=0.7)
    ax5.bar(x + width/2, post_values, width, label='Post-processing', color='#FFA400', alpha=0.7)
    
    ax5.set_title('Reverb Analysis', color='white', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(reverb_features, color='white')
    ax5.set_ylabel('Value', color='white')
    ax5.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def compare_vocal_files(pre_file, post_file, plot=True):
    """
    Compare pre-processed and post-processed vocal files and provide analysis.
    
    Parameters:
    -----------
    pre_file : str
        Path to pre-processed audio file
    post_file : str
        Path to post-processed audio file
    plot : bool
        Whether to generate comparison plots
        
    Returns:
    --------
    dict
        Comparison analysis results
    """
    # Load audio files
    y_pre, sr_pre = librosa.load(pre_file, sr=None)
    y_post, sr_post = librosa.load(post_file, sr=None)
    
    # Ensure same sample rate
    if sr_pre != sr_post:
        y_post = librosa.resample(y=y_post, orig_sr=sr_post, target_sr=sr_pre)
        sr_post = sr_pre
    
    # Ensure same length for comparison
    min_length = min(len(y_pre), len(y_post))
    y_pre = y_pre[:min_length]
    y_post = y_post[:min_length]
    
    # 1. Dynamic Range Analysis
    dynamic_range_pre = calculate_dynamic_range(y_pre)
    dynamic_range_post = calculate_dynamic_range(y_post)
    
    # 2. Frequency Distribution Analysis
    freq_analysis_pre = analyze_frequency_distribution(y_pre, sr_pre)
    freq_analysis_post = analyze_frequency_distribution(y_post, sr_post)
    
    # 3. Stereo Width Analysis
    stereo_width_pre = analyze_stereo_width(y_pre)
    stereo_width_post = analyze_stereo_width(y_post)
    
    # 4. Harmonic Content Analysis
    harmonic_analysis_pre = calculate_harmonic_content(y_pre, sr_pre)
    harmonic_analysis_post = calculate_harmonic_content(y_post, sr_post)
    
    # 5. Reverb Analysis
    reverb_analysis_pre = analyze_reverb(y_pre, sr_pre)
    reverb_analysis_post = analyze_reverb(y_post, sr_post)
    
    # Generate plots if requested
    if plot:
        plot_comparison_results(
            y_pre, y_post, sr_pre,
            freq_analysis_pre, freq_analysis_post,
            dynamic_range_pre, dynamic_range_post,
            stereo_width_pre, stereo_width_post,
            harmonic_analysis_pre, harmonic_analysis_post,
            reverb_analysis_pre, reverb_analysis_post
        )
    
    # Calculate frequency band changes
    band_changes = {}
    for band, energy_pre in freq_analysis_pre["band_energy"].items():
        energy_post = freq_analysis_post["band_energy"][band]
        # Calculate relative change in dB
        if energy_pre > 0 and energy_post > 0:
            change_db = 20 * np.log10(energy_post / energy_pre)
        else:
            change_db = 0
        band_changes[band] = change_db
    
    # Prepare comparison results
    comparison = {
        "dynamic_range": {
            "pre": dynamic_range_pre,
            "post": dynamic_range_post,
            "change": dynamic_range_post - dynamic_range_pre
        },
        "frequency_bands": {
            "pre": freq_analysis_pre["band_energy"],
            "post": freq_analysis_post["band_energy"],
            "change_db": band_changes
        },
        "stereo_width": {
            "pre": stereo_width_pre,
            "post": stereo_width_post,
            "change": (stereo_width_post - stereo_width_pre) if stereo_width_pre is not None and stereo_width_post is not None else None
        },
        "harmonic_content": {
            "pre": harmonic_analysis_pre,
            "post": harmonic_analysis_post,
            "spectral_centroid_change": harmonic_analysis_post["spectral_centroid"] - harmonic_analysis_pre["spectral_centroid"],
            "harmonic_ratio_change": harmonic_analysis_post["harmonic_ratio"] - harmonic_analysis_pre["harmonic_ratio"]
        },
        "reverb": {
            "pre": reverb_analysis_pre,
            "post": reverb_analysis_post,
            "decay_time_change": reverb_analysis_post["decay_time"] - reverb_analysis_pre["decay_time"]
        }
    }
    
    # Generate textual analysis
    analysis_text = generate_comparison_text(comparison)
    comparison["text_analysis"] = analysis_text
    
    return comparison

def generate_comparison_text(comparison):
    """
    Generate textual analysis from the comparison results.
    
    Parameters:
    -----------
    comparison : dict
        Comparison results from compare_vocal_files
        
    Returns:
    --------
    dict
        Textual analysis of changes
    """
    analysis = {}
    
    # Dynamic Range Analysis
    dr_change = comparison["dynamic_range"]["change"]
    if abs(dr_change) < 1:
        analysis["dynamic_range"] = "Dynamic range remains largely unchanged"
    elif dr_change < 0:
        decrease_percent = abs(dr_change) / comparison["dynamic_range"]["pre"] * 100 if comparison["dynamic_range"]["pre"] > 0 else 0
        analysis["dynamic_range"] = f"Dynamic range reduced by {abs(dr_change):.1f}dB ({decrease_percent:.1f}%) due to compression"
    else:
        increase_percent = dr_change / comparison["dynamic_range"]["pre"] * 100 if comparison["dynamic_range"]["pre"] > 0 else 0
        analysis["dynamic_range"] = f"Dynamic range increased by {dr_change:.1f}dB ({increase_percent:.1f}%), unusual for vocal processing"
    
    # Frequency Analysis
    freq_changes = []
    for band, change_db in comparison["frequency_bands"]["change_db"].items():
        if abs(change_db) >= 1.0:  # Only significant changes
            direction = "boosted" if change_db > 0 else "reduced"
            freq_changes.append(f"{band.replace('_', ' ')} {direction} by {abs(change_db):.1f}dB")
    
    if freq_changes:
        analysis["frequency_balance"] = "Frequency balance changes: " + "; ".join(freq_changes)
    else:
        analysis["frequency_balance"] = "Minimal changes to frequency balance"
    
    # Stereo Width Analysis
    if comparison["stereo_width"]["pre"] is not None and comparison["stereo_width"]["post"] is not None:
        width_change = comparison["stereo_width"]["change"]
        if abs(width_change) < 0.05:
            analysis["stereo_width"] = "Stereo width remains largely unchanged"
        elif width_change > 0:
            analysis["stereo_width"] = f"Stereo width increased by {width_change*100:.1f}%, likely due to reverb or stereo effects"
        else:
            analysis["stereo_width"] = f"Stereo width decreased by {abs(width_change)*100:.1f}%"
    else:
        analysis["stereo_width"] = "Stereo analysis not applicable (mono files)"
    
    # Harmonic Content Analysis
    harmonic_ratio_change = comparison["harmonic_content"]["harmonic_ratio_change"]
    centroid_change = comparison["harmonic_content"]["spectral_centroid_change"]
    
    if abs(harmonic_ratio_change) > 0.1 or abs(centroid_change) > 200:
        if harmonic_ratio_change > 0.1:
            analysis["harmonic_content"] = "Increased harmonic content, likely due to saturation/distortion"
        elif harmonic_ratio_change < -0.1:
            analysis["harmonic_content"] = "Reduced harmonic content, possibly due to filtering"
        
        if centroid_change > 200:
            analysis["spectral_balance"] = "Brighter sound (increased high frequencies)"
        elif centroid_change < -200:
            analysis["spectral_balance"] = "Warmer sound (reduced high frequencies)"
        else:
            analysis["spectral_balance"] = "Minimal change in overall brightness"
    else:
        analysis["harmonic_content"] = "Harmonic content relatively unchanged"
        analysis["spectral_balance"] = "No significant change in spectral balance"
    
    # Reverb Analysis
    decay_change = comparison["reverb"]["decay_time_change"]
    
    if comparison["reverb"]["post"]["decay_time"] > 0.5 and comparison["reverb"]["pre"]["decay_time"] < 0.2:
        analysis["reverb"] = f"Reverb added with approximately {comparison['reverb']['post']['decay_time']:.2f}s decay time"
    elif decay_change > 0.1:
        analysis["reverb"] = f"Reverb increased by {decay_change:.2f}s decay time"
    else:
        analysis["reverb"] = "No significant reverb changes detected"
    
    return analysis
