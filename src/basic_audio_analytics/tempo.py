import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Tuple, List, Dict, Optional, Union

def detect_tempo(
    audio_file: str = None, 
    y: np.ndarray = None, 
    sr: int = 22050,
    onset_method: str = 'spectral_flux',
    hop_length: int = 512,
    start_bpm: float = 120.0,
    min_tempo: float = 60.0,
    max_tempo: float = 240.0,
    visualize: bool = False,
    n_tempi: int = 1
) -> Union[float, Dict]:
    """
    Detect the tempo (BPM - beats per minute) of an audio file or audio array.
    
    Parameters
    ----------
    audio_file : str, optional
        Path to the audio file to analyze. If provided, y and sr are ignored.
    y : np.ndarray, optional
        Audio time series. Required if audio_file is not provided.
    sr : int, optional
        Sampling rate of the audio time series. Default is 22050.
    onset_method : str, optional
        Method for detecting onsets. Options are:
        - 'spectral_flux': Based on spectral difference (default)
        - 'energy': Based on energy changes
        - 'hfc': High Frequency Content
        - 'complex': Complex domain onset detection
        - 'melflux': Mel-frequency spectrogram difference
        - 'rms': Root-mean-square energy
    hop_length : int, optional
        Number of samples between successive frames. Default is 512.
    start_bpm : float, optional
        Initial tempo estimate in BPM. Default is 120.0.
    min_tempo : float, optional
        Minimum tempo to consider in BPM. Default is 60.0.
    max_tempo : float, optional
        Maximum tempo to consider in BPM. Default is 240.0.
    visualize : bool, optional
        Whether to visualize the tempo detection process. Default is False.
    n_tempi : int, optional
        Number of tempi to return. Default is 1 (only the most confident estimate).
        If greater than 1, returns a dictionary with multiple tempo candidates and their confidence.
    
    Returns
    -------
    float or dict
        If n_tempi=1, returns the estimated tempo in BPM.
        If n_tempi>1, returns a dictionary with:
        - 'tempo': List of estimated tempi in descending order of confidence
        - 'confidence': List of confidence values (summing to 1.0)
        - 'onset_strength': The onset strength envelope 
        - 'onset_times': Times of detected onsets in seconds
    
    Examples
    --------
    >>> # Detect tempo from file
    >>> bpm = detect_tempo('song.mp3')
    >>> print(f"The estimated tempo is {bpm:.1f} BPM")
    
    >>> # Detect with visualization
    >>> bpm = detect_tempo('song.mp3', visualize=True)
    
    >>> # Get multiple tempo candidates
    >>> tempo_info = detect_tempo('song.mp3', n_tempi=3)
    >>> print(f"Top tempo: {tempo_info['tempo'][0]:.1f} BPM with confidence {tempo_info['confidence'][0]:.2f}")
    
    Notes
    -----
    This function uses a multi-stage approach for tempo detection:
    1. Extract the onset strength envelope
    2. Identify peaks in the onset strength (potential beats)
    3. Calculate tempo using auto-correlation and dynamic programming
    4. Refine the result with beat tracking
    
    For more accurate results, consider using longer audio segments (>30 seconds).
    """
    # Load audio file if provided, otherwise use the provided audio time series
    if audio_file is not None:
        y, sr = librosa.load(audio_file, sr=sr)
    elif y is None:
        raise ValueError("Either audio_file or y must be provided")
        
    # Step 1: Extract onset strength envelope
    onset_env = _get_onset_strength(y, sr, hop_length, onset_method)
    
    # Step 2: Calculate the static tempo estimate
    tempo_static = librosa.beat.tempo(
        onset_envelope=onset_env, 
        sr=sr, 
        hop_length=hop_length,
        start_bpm=start_bpm,
        min_tempo=min_tempo,
        max_tempo=max_tempo
    )[0]
    
    # Step 3: Get dynamic tempo estimate with beat tracking
    tempo, beats = _get_dynamic_tempo(
        y, sr, onset_env, hop_length, 
        start_bpm=tempo_static,
        min_tempo=min_tempo,
        max_tempo=max_tempo
    )
    
    # Step 4: Calculate onset times
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    
    # Step 5: Get peaks from onset strength (for visualization)
    peaks, _ = find_peaks(onset_env, distance=sr/hop_length/4)  # At least 250ms between peaks
    peak_times = onset_times[peaks]
    peak_values = onset_env[peaks]
    
    # Visualize if requested
    if visualize:
        _visualize_tempo_detection(
            y, sr, onset_env, onset_times, tempo, 
            peak_times, peak_values, beats
        )
    
    # Return results
    if n_tempi == 1:
        return tempo
    else:
        # Calculate multiple tempo candidates
        ac_tempo = _get_tempo_candidates(onset_env, sr, hop_length, n_tempi, 
                                         min_tempo, max_tempo)
        
        return {
            'tempo': ac_tempo['tempo'],
            'confidence': ac_tempo['confidence'],
            'onset_strength': onset_env,
            'onset_times': onset_times
        }

def _get_onset_strength(
    y: np.ndarray, 
    sr: int, 
    hop_length: int, 
    method: str = 'spectral_flux'
) -> np.ndarray:
    """
    Calculate onset strength envelope using different methods.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    hop_length : int
        Number of samples between frames
    method : str
        Onset detection method
        
    Returns
    -------
    np.ndarray
        Onset strength envelope
    """
    # Normalize audio
    y = librosa.util.normalize(y)
    
    if method == 'spectral_flux':
        # Compute spectrogram
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        # Compute spectral flux onset strength
        onset_env = librosa.onset.onset_strength(
            S=librosa.amplitude_to_db(S, ref=np.max), 
            sr=sr, 
            hop_length=hop_length
        )
    elif method == 'energy':
        # Energy-based onset detection
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length,
            feature=librosa.feature.rms
        )
    elif method == 'hfc':
        # High-frequency content onset detection
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        onset_env = librosa.onset.onset_strength(
            S=librosa.power_to_db(S**2, ref=np.max), 
            sr=sr, 
            hop_length=hop_length
        )
    elif method == 'complex':
        # Complex domain onset detection
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length, 
            feature=librosa.feature.spectral_flatness
        )
    elif method == 'melflux':
        # Mel-frequency spectrogram difference
        S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
        onset_env = librosa.onset.onset_strength(
            S=librosa.power_to_db(S, ref=np.max), 
            sr=sr, 
            hop_length=hop_length
        )
    elif method == 'rms':
        # RMS energy
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length,
            feature=librosa.feature.rms
        )
    else:
        raise ValueError(f"Unknown onset method: {method}")
    
    return onset_env

def _get_dynamic_tempo(
    y: np.ndarray, 
    sr: int, 
    onset_env: np.ndarray, 
    hop_length: int,
    start_bpm: float = 120.0,
    min_tempo: float = 60.0,
    max_tempo: float = 240.0
) -> Tuple[float, np.ndarray]:
    """
    Calculate dynamic tempo estimate with beat tracking.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    onset_env : np.ndarray
        Onset strength envelope
    hop_length : int
        Number of samples between frames
    start_bpm : float
        Initial tempo estimate in BPM
    min_tempo : float
        Minimum tempo to consider in BPM
    max_tempo : float
        Maximum tempo to consider in BPM
        
    Returns
    -------
    float
        Estimated tempo in BPM
    np.ndarray
        Beat locations in frames
    """
    # Use beat tracking to refine tempo estimate
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=100,  # Make the beat tracker more rigid
        trim=False,
        bpm=None,
        units='frames'
    )
    
    # Ensure tempo is within bounds
    if tempo < min_tempo:
        tempo *= 2
    elif tempo > max_tempo:
        tempo *= 0.5
        
    return tempo, beats

def _get_tempo_candidates(
    onset_env: np.ndarray, 
    sr: int, 
    hop_length: int, 
    n_tempi: int = 3,
    min_tempo: float = 60.0,
    max_tempo: float = 240.0
) -> Dict:
    """
    Calculate multiple tempo candidates using auto-correlation.
    
    Parameters
    ----------
    onset_env : np.ndarray
        Onset strength envelope
    sr : int
        Sampling rate
    hop_length : int
        Number of samples between frames
    n_tempi : int
        Number of tempo candidates to return
    min_tempo : float
        Minimum tempo to consider in BPM
    max_tempo : float
        Maximum tempo to consider in BPM
        
    Returns
    -------
    Dict
        Dictionary with tempo candidates and confidence scores
    """
    # Calculate auto-correlation
    ac = librosa.autocorrelate(onset_env, max_size=4 * sr // hop_length)
    
    # Convert lag indices to tempi
    tempo_lags = np.arange(1, len(ac))
    tempi = 60 * sr / (hop_length * tempo_lags)
    
    # Filter to the target tempo range
    valid_tempi = (tempi >= min_tempo) & (tempi <= max_tempo)
    tempi = tempi[valid_tempi]
    valid_ac = ac[1:][valid_tempi]
    
    # Remove negative correlations
    valid_ac = np.maximum(0, valid_ac)
    
    if len(valid_ac) == 0:
        # No valid candidates, return default
        return {
            'tempo': [120.0],
            'confidence': [1.0]
        }
    
    # Sort tempo candidates
    sort_idx = np.argsort(valid_ac)[::-1]
    
    # Limit to top n candidates
    n_tempi = min(n_tempi, len(sort_idx))
    top_tempi = tempi[sort_idx[:n_tempi]]
    top_scores = valid_ac[sort_idx[:n_tempi]]
    
    # Normalize confidence scores
    if np.sum(top_scores) > 0:
        confidence = top_scores / np.sum(top_scores)
    else:
        confidence = np.ones(len(top_scores)) / len(top_scores)
    
    return {
        'tempo': top_tempi.tolist(),
        'confidence': confidence.tolist()
    }

def _visualize_tempo_detection(
    y: np.ndarray, 
    sr: int, 
    onset_env: np.ndarray, 
    onset_times: np.ndarray,
    tempo: float, 
    peak_times: np.ndarray, 
    peak_values: np.ndarray,
    beats: np.ndarray
) -> None:
    """
    Visualize the tempo detection process.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    onset_env : np.ndarray
        Onset strength envelope
    onset_times : np.ndarray
        Times corresponding to onset_env
    tempo : float
        Estimated tempo in BPM
    peak_times : np.ndarray
        Times of peak onsets
    peak_values : np.ndarray
        Values of peak onsets
    beats : np.ndarray
        Beat locations in frames
    """
    # Ensure dark mode for consistent styling with other visualizations
    plt.style.use('dark_background')
    
    # Create figure with multiple plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#00ff9f')
    ax1.set_title('Audio Waveform', fontsize=14, color='white')
    ax1.label_outer()
    
    # Plot onset strength
    ax2.plot(onset_times, onset_env, color='#00a7e1', label='Onset Strength')
    ax2.scatter(peak_times, peak_values, color='#ff00ff', s=30, label='Peaks')
    
    # Plot beat times if available
    if beats.size > 0:
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
        ax2.vlines(beat_times, 0, onset_env.max(), color='#ffa400', 
                  linestyle='--', alpha=0.7, label='Detected Beats')
    
    ax2.set_title(f'Onset Strength & Beat Detection (Tempo: {tempo:.1f} BPM)', 
                 fontsize=14, color='white')
    ax2.legend(loc='upper right')
    ax2.label_outer()
    
    # Plot click track based on detected tempo
    duration = len(y) / sr
    beat_period = 60 / tempo
    click_times = np.arange(0, duration, beat_period)
    
    # Generate click track waveform
    click_track = np.zeros_like(y)
    for time in click_times:
        sample = int(time * sr)
        if sample < len(click_track) - 100:  # Ensure we don't go out of bounds
            click_track[sample:sample+100] = np.linspace(1, 0, 100)
    
    librosa.display.waveshow(click_track, sr=sr, ax=ax3, color='#ffa400')
    ax3.set_title('Generated Click Track at Detected Tempo', fontsize=14, color='white')
    
    # Customize all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Time (s)', fontsize=12, color='white')
        ax.grid(True, color='#404040', linestyle=':', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color('#404040')
        ax.tick_params(colors='white')
    
    ax1.set_ylabel('Amplitude', fontsize=12, color='white')
    ax2.set_ylabel('Magnitude', fontsize=12, color='white')
    ax3.set_ylabel('Amplitude', fontsize=12, color='white')
    
    # Add explanation text
    analysis_details = f"""
    Tempo Analysis Details:
    • Detected Tempo: {tempo:.1f} BPM
    • Beat Period: {beat_period:.3f} seconds
    • Analyzed Duration: {duration:.1f} seconds
    • Number of Beats Detected: {len(beats)}
    """
    
    plt.figtext(0.02, 0.02, analysis_details, fontsize=12, color='white',
               bbox=dict(facecolor='#1A1A1A', alpha=0.8, 
                       pad=10, edgecolor='#404040'))
    
    plt.tight_layout()
    plt.show()

def analyze_tempo_distribution(
    audio_file: str = None, 
    y: np.ndarray = None,
    sr: int = 22050,
    frame_length: int = 10,
    hop_length: int = 512,
    min_tempo: float = 60.0,
    max_tempo: float = 240.0,
    visualize: bool = True
) -> Dict:
    """
    Analyze tempo variation throughout an audio file.
    
    Parameters
    ----------
    audio_file : str, optional
        Path to the audio file to analyze. If provided, y and sr are ignored.
    y : np.ndarray, optional
        Audio time series. Required if audio_file is not provided.
    sr : int, optional
        Sampling rate of the audio time series. Default is 22050.
    frame_length : int, optional
        Length of each analysis window in seconds. Default is 10.
    hop_length : int, optional
        Number of samples between successive frames. Default is 512.
    min_tempo : float, optional
        Minimum tempo to consider in BPM. Default is 60.0.
    max_tempo : float, optional
        Maximum tempo to consider in BPM. Default is 240.0.
    visualize : bool, optional
        Whether to visualize the tempo distribution. Default is True.
    
    Returns
    -------
    Dict
        Dictionary with tempo analysis results:
        - 'tempo_segments': List of tempo values for each segment
        - 'segment_times': Start time of each segment
        - 'avg_tempo': Average tempo across all segments
        - 'min_tempo': Minimum tempo observed
        - 'max_tempo': Maximum tempo observed
        - 'std_tempo': Standard deviation of tempo
    
    Examples
    --------
    >>> # Analyze tempo distribution for a song
    >>> results = analyze_tempo_distribution('song.mp3', frame_length=5)
    >>> print(f"Average tempo: {results['avg_tempo']:.1f} BPM")
    >>> print(f"Tempo range: {results['min_tempo']:.1f} - {results['max_tempo']:.1f} BPM")
    
    Notes
    -----
    This function divides the audio into overlapping segments and estimates
    the tempo for each segment. It's useful for analyzing songs with tempo
    changes or for estimating tempo stability.
    """
    # Load audio file if provided, otherwise use the provided audio time series
    if audio_file is not None:
        y, sr = librosa.load(audio_file, sr=sr)
    elif y is None:
        raise ValueError("Either audio_file or y must be provided")
    
    # Calculate segment parameters
    frame_samples = int(frame_length * sr)
    half_frame = frame_samples // 2  # 50% overlap
    
    tempo_values = []
    segment_times = []
    
    # Process each segment
    for i in range(0, len(y) - frame_samples, half_frame):
        segment = y[i:i+frame_samples]
        segment_time = i / sr
        
        # Skip segments with very low energy
        if np.mean(np.abs(segment)) < 0.01:
            continue
            
        # Get tempo for this segment
        try:
            tempo = detect_tempo(
                y=segment, 
                sr=sr, 
                hop_length=hop_length,
                min_tempo=min_tempo,
                max_tempo=max_tempo,
                visualize=False
            )
            
            tempo_values.append(tempo)
            segment_times.append(segment_time)
        except Exception as e:
            print(f"Error analyzing segment at {segment_time}s: {e}")
    
    # Calculate statistics
    if tempo_values:
        avg_tempo = np.mean(tempo_values)
        min_tempo_val = np.min(tempo_values)
        max_tempo_val = np.max(tempo_values)
        std_tempo = np.std(tempo_values)
    else:
        avg_tempo = min_tempo_val = max_tempo_val = std_tempo = 0
    
    # Visualize if requested
    if visualize and tempo_values:
        _visualize_tempo_distribution(tempo_values, segment_times, 
                                     avg_tempo, min_tempo_val, max_tempo_val)
    
    return {
        'tempo_segments': tempo_values,
        'segment_times': segment_times,
        'avg_tempo': avg_tempo,
        'min_tempo': min_tempo_val,
        'max_tempo': max_tempo_val,
        'std_tempo': std_tempo
    }

def _visualize_tempo_distribution(
    tempo_values: List[float], 
    segment_times: List[float],
    avg_tempo: float, 
    min_tempo: float, 
    max_tempo: float
) -> None:
    """
    Visualize tempo distribution across the audio file.
    
    Parameters
    ----------
    tempo_values : List[float]
        List of tempo values for each segment
    segment_times : List[float]
        Start time of each segment
    avg_tempo : float
        Average tempo across all segments
    min_tempo : float
        Minimum tempo observed
    max_tempo : float
        Maximum tempo observed
    """
    # Ensure dark mode for consistent styling with other visualizations
    plt.style.use('dark_background')
    
    # Create figure with multiple plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot tempo over time
    ax1.plot(segment_times, tempo_values, color='#00ff9f', linewidth=2)
    ax1.axhline(y=avg_tempo, color='#ffa400', linestyle='--', 
               label=f'Average: {avg_tempo:.1f} BPM')
    
    # Add tempo range as a shaded area
    ax1.fill_between(segment_times, 
                    [min_tempo] * len(segment_times),
                    [max_tempo] * len(segment_times),
                    color='#00a7e1', alpha=0.2, 
                    label=f'Range: {min_tempo:.1f} - {max_tempo:.1f} BPM')
    
    ax1.set_title('Tempo Variation Over Time', fontsize=14, color='white')
    ax1.set_xlabel('Time (s)', fontsize=12, color='white')
    ax1.set_ylabel('Tempo (BPM)', fontsize=12, color='white')
    ax1.legend(loc='upper right')
    ax1.grid(True, color='#404040', linestyle=':', alpha=0.3)
    
    # Plot tempo histogram
    bins = np.arange(min(60, min(tempo_values)-5), max(240, max(tempo_values)+5), 2)
    ax2.hist(tempo_values, bins=bins, color='#00a7e1', alpha=0.7)
    ax2.axvline(x=avg_tempo, color='#ffa400', linestyle='--', linewidth=2,
               label=f'Average: {avg_tempo:.1f} BPM')
    
    ax2.set_title('Tempo Distribution Histogram', fontsize=14, color='white')
    ax2.set_xlabel('Tempo (BPM)', fontsize=12, color='white')
    ax2.set_ylabel('Frequency', fontsize=12, color='white')
    ax2.legend(loc='upper right')
    ax2.grid(True, color='#404040', linestyle=':', alpha=0.3)
    
    # Customize all axes
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('#404040')
        ax.tick_params(colors='white')
    
    # Add explanation text
    analysis_details = f"""
    Tempo Distribution Analysis:
    • Average Tempo: {avg_tempo:.1f} BPM
    • Min Tempo: {min_tempo:.1f} BPM
    • Max Tempo: {max_tempo:.1f} BPM
    • Standard Deviation: {np.std(tempo_values):.2f} BPM
    • Tempo Stability: {1 - np.std(tempo_values) / avg_tempo if avg_tempo > 0 else 0:.2f} (0-1 scale)
    • Total Analyzed Segments: {len(tempo_values)}
    """
    
    plt.figtext(0.02, 0.02, analysis_details, fontsize=12, color='white',
               bbox=dict(facecolor='#1A1A1A', alpha=0.8, 
                       pad=10, edgecolor='#404040'))
    
    plt.tight_layout()
    plt.show()
