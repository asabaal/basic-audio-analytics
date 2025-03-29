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
    sorted_indices = np.argsort(peak_magnitudes)[::-1][:10]  # Get indices of top 10 peaks
    
    # Extract the primary frequency values and their magnitudes
    primary_freqs = filtered_freqs[peaks[sorted_indices]]
    primary_mags = filtered_spectrum[peaks[sorted_indices]]
    
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
            except:
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
