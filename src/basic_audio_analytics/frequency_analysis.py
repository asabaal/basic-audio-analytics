import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def identify_primary_frequencies(audio_file, n_frequencies=5, min_frequency=80, max_frequency=8000, 
                                 fmin=80, fmax=8000, threshold=0.5, plot=True):
    """
    Identifies the primary frequencies in an audio file, particularly useful for vocal tracks.
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file
    n_frequencies : int
        Number of primary frequencies to identify
    min_frequency : int
        Minimum frequency to consider (Hz)
    max_frequency : int
        Maximum frequency to consider (Hz)
    fmin : int
        Minimum frequency for plotting
    fmax : int
        Maximum frequency for plotting
    threshold : float
        Threshold for peak detection (0.0-1.0)
    plot : bool
        Whether to plot the spectrum with identified peaks
    
    Returns:
    --------
    dict
        Containing primary frequencies and their magnitudes
    """
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
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
    
    # Filter by frequency range
    freq_range = (freqs >= min_frequency) & (freqs <= max_frequency)
    filtered_freqs = freqs[freq_range]
    filtered_spectrum = mean_spectrum[freq_range]
    
    # Find peaks
    peaks, _ = find_peaks(filtered_spectrum, height=threshold)
    
    # Get the top n_frequencies peaks
    peak_magnitudes = filtered_spectrum[peaks]
    sorted_indices = np.argsort(peak_magnitudes)[::-1][:n_frequencies]  # Get indices of top n peaks
    
    # Extract the primary frequency values and their magnitudes
    primary_freqs = filtered_freqs[peaks[sorted_indices]]
    primary_mags = filtered_spectrum[peaks[sorted_indices]]
    
    # Determine frequency bands
    bass_region = (primary_freqs >= 60) & (primary_freqs <= 250)
    low_mid_region = (primary_freqs > 250) & (primary_freqs <= 500)
    mid_region = (primary_freqs > 500) & (primary_freqs <= 2000)
    high_mid_region = (primary_freqs > 2000) & (primary_freqs <= 4000)
    high_region = (primary_freqs > 4000)
    
    # Create regions dictionary
    regions = {
        "bass": primary_freqs[bass_region].tolist(),
        "low_mid": primary_freqs[low_mid_region].tolist(),
        "mid": primary_freqs[mid_region].tolist(),
        "high_mid": primary_freqs[high_mid_region].tolist(),
        "high": primary_freqs[high_region].tolist()
    }
    
    # Generate EQ recommendations
    eq_recommendations = generate_eq_recommendations(regions)
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_freqs, filtered_spectrum, color='#87CEFA')
        plt.scatter(primary_freqs, primary_mags, color='red', marker='x', s=100, zorder=5)
        
        # Label the peaks
        for i, (freq, mag) in enumerate(zip(primary_freqs, primary_mags)):
            plt.annotate(f'{freq:.0f} Hz', 
                        xy=(freq, mag),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', 
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.7))
        
        plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7)
        plt.xlim(fmin, fmax)
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Audio Spectrum with Primary Frequencies')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return {
        "primary_frequencies": list(zip(primary_freqs, primary_mags)),
        "frequency_regions": regions,
        "eq_recommendations": eq_recommendations
    }

def generate_eq_recommendations(regions):
    """
    Generate EQ recommendations based on identified frequency regions
    
    Parameters:
    -----------
    regions : dict
        Dictionary of frequency regions
    
    Returns:
    --------
    dict
        EQ recommendations
    """
    recommendations = {}
    
    # High-pass filter recommendation
    if not regions["bass"] and not regions["low_mid"]:
        recommendations["high_pass"] = "Consider using a high-pass filter around 120-150Hz to remove rumble"
    elif not regions["bass"] and regions["low_mid"]:
        recommendations["high_pass"] = "Use a gentle high-pass filter around 80-100Hz to preserve low-mids"
    elif regions["bass"]:
        min_bass = min(regions["bass"]) if regions["bass"] else 80
        recommendations["high_pass"] = f"Use a high-pass filter below {max(40, min_bass-20)}Hz to remove sub-bass rumble"
    
    # Bass recommendations
    if regions["bass"]:
        recommendations["bass"] = f"Consider a gentle boost around {regions['bass'][0]}Hz for warmth"
    
    # Mid recommendations
    if regions["mid"]:
        recommendations["mid"] = f"The vocal presence is centered around {regions['mid'][0]}Hz"
        if len(regions["mid"]) > 1:
            recommendations["mid"] += f" with secondary presence at {regions['mid'][1]}Hz"
    
    # High-mid recommendations
    if regions["high_mid"]:
        if len(regions["high_mid"]) > 2:
            recommendations["high_mid"] = "Potential harshness in high-mids, consider a small cut around 2-4kHz"
        else:
            recommendations["high_mid"] = f"For clarity, consider a small boost around {regions['high_mid'][0]}Hz"
    
    # High recommendations
    if regions["high"]:
        recommendations["high"] = "Good high frequency content, consider a gentle shelf boost above 6kHz for air"
    else:
        recommendations["high"] = "Limited high frequency content, consider a shelf boost above 8kHz for air"
    
    return recommendations

def analyze_audio_for_fl_studio(audio_file):
    """
    Wrapper function specifically for FL Studio EQ recommendations
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file
    
    Returns:
    --------
    dict
        FL Studio specific EQ settings
    """
    # Run the analysis
    results = identify_primary_frequencies(audio_file)
    
    # Extract FL Studio Parametric EQ 2 settings
    eq_settings = {
        "Band 1 (HP)": "Set to high-pass mode",
        "Band 2 (Bass)": "N/A",
        "Band 3 (Low-Mid)": "N/A",
        "Band 4 (Mid)": "N/A", 
        "Band 5 (High-Mid)": "N/A",
        "Band 6 (High)": "N/A",
        "Band 7 (Treble)": "N/A"
    }
    
    # High-pass filter (Band 1)
    if "high_pass" in results["eq_recommendations"]:
        hp_rec = results["eq_recommendations"]["high_pass"]
        hp_freq = 80  # Default
        
        # Extract frequency from recommendation
        import re
        freq_match = re.search(r'(\d+)Hz', hp_rec)
        if freq_match:
            hp_freq = int(freq_match.group(1))
        
        eq_settings["Band 1 (HP)"] = f"High-pass at {hp_freq}Hz with 12dB/oct slope"
    
    # Bass (Band 2)
    if "bass" in results["eq_recommendations"] and results["frequency_regions"]["bass"]:
        bass_freq = results["frequency_regions"]["bass"][0]
        eq_settings["Band 2 (Bass)"] = f"Boost around {int(bass_freq)}Hz by 2-3dB with medium Q"
    
    # Low-mid (Band 3)
    if results["frequency_regions"]["low_mid"]:
        low_mid_freq = results["frequency_regions"]["low_mid"][0]
        eq_settings["Band 3 (Low-Mid)"] = f"Adjust around {int(low_mid_freq)}Hz as needed"
    
    # Mid (Band 4)
    if results["frequency_regions"]["mid"]:
        mid_freq = results["frequency_regions"]["mid"][0]
        eq_settings["Band 4 (Mid)"] = f"Focus on {int(mid_freq)}Hz - this is the vocal center"
    
    # High-mid (Band 5)
    if "high_mid" in results["eq_recommendations"]:
        if "cut" in results["eq_recommendations"]["high_mid"].lower():
            eq_settings["Band 5 (High-Mid)"] = "Cut around 2-4kHz by 2-3dB with narrow Q"
        elif results["frequency_regions"]["high_mid"]:
            high_mid_freq = results["frequency_regions"]["high_mid"][0]
            eq_settings["Band 5 (High-Mid)"] = f"Boost around {int(high_mid_freq)}Hz by 1-2dB for clarity"
    
    # High (Band 7)
    if "high" in results["eq_recommendations"]:
        if "boost" in results["eq_recommendations"]["high"].lower():
            eq_settings["Band 7 (Treble)"] = "Add a gentle shelf boost above 8kHz for air"
    
    return {
        "analysis_results": results,
        "fl_studio_eq_settings": eq_settings
    }