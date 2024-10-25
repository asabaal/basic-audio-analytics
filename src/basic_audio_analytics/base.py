import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.signal.windows import hamming

def plot_waveform(amplitude, sample_rate):
    # Display waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(amplitude, sr=sample_rate, color='#00ff9f')  # Using a bright cyan-green color
    plt.title('Audio Waveform', color='white')

    # Customize grid and spines for better visibility
    plt.grid(True, color='#303030')
    for spine in plt.gca().spines.values():
        spine.set_color('#404040')

    plt.show()    

def plot_amplitude_distributions(amplitude):
    import matplotlib.pyplot as plt
    # Set dark style for consistency
    plt.style.use('dark_background')

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Original amplitude distribution (linear)
    sns.histplot(amplitude, bins=100, color='#00ff9f', alpha=0.6, ax=ax1, stat='probability', common_norm=False)
    ax1.set_title('Original Amplitude Distribution (Linear)', color='white')
    ax1.set_xlabel('Amplitude', color='white')
    ax1.set_ylabel('Percentage', color='white')
    ax1.tick_params(colors='white')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y * 100)))
    for spine in ax1.spines.values():
        spine.set_color('#404040')

    # Plot 2: Absolute amplitude distribution (linear)
    sns.histplot(np.abs(amplitude), bins=100, color='#ff00ff', alpha=0.6, ax=ax2, stat='probability', common_norm=False)
    ax2.set_title('Absolute Amplitude Distribution (Linear)', color='white')
    ax2.set_xlabel('Absolute Amplitude', color='white')
    ax2.set_ylabel('Percentage', color='white')
    ax2.tick_params(colors='white')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y * 100)))
    for spine in ax2.spines.values():
        spine.set_color('#404040')

    # Plot 3: Original amplitude distribution (log scale)
    sns.histplot(amplitude, bins=100, color='#00ff9f', alpha=0.6, ax=ax3, stat='probability', common_norm=False)
    ax3.set_title('Original Amplitude Distribution (Log Scale)', color='white')
    ax3.set_xlabel('Amplitude', color='white')
    ax3.set_ylabel('Percentage (Log Scale)', color='white')
    ax3.set_yscale('log')
    ax3.tick_params(colors='white')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1e}%'.format(y * 100)))
    for spine in ax3.spines.values():
        spine.set_color('#404040')

    # Plot 4: Absolute amplitude distribution (log scale)
    sns.histplot(np.abs(amplitude), bins=100, color='#ff00ff', alpha=0.6, ax=ax4, stat='probability', common_norm=False)
    ax4.set_title('Absolute Amplitude Distribution (Log Scale)', color='white')
    ax4.set_xlabel('Absolute Amplitude', color='white')
    ax4.set_ylabel('Percentage (Log Scale)', color='white')
    ax4.set_yscale('log')
    ax4.tick_params(colors='white')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1e}%'.format(y * 100)))
    for spine in ax4.spines.values():
        spine.set_color('#404040')

    plt.tight_layout()
    plt.show()

def plot_spectrogram(S_db, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, 
                            sr=sample_rate, 
                            x_axis='time', 
                            y_axis='hz',
                            cmap='magma')  # Using magma colormap which looks great in dark mode

    # Customize colorbar
    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.set_ylabel('Amplitude (dB)', color='white')

    # Set title and customize appearance
    plt.title('Spectrogram', color='white')

    # Customize grid and spines for better visibility
    for spine in plt.gca().spines.values():
        spine.set_color('#404040')

    plt.show()    

def plot_log_spectrograms(amplitude, S_db, sample_rate):
    # Create a figure with multiple views of the same data
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Top plot: Original spectrogram but with log frequency scaling
    img1 = librosa.display.specshow(S_db, 
                            sr=sample_rate,
                            x_axis='time',
                            y_axis='log',  # Changed to log scale
                            ax=ax1,
                            cmap='magma')
    ax1.set_title('Spectrogram (Log Frequency Scale)', color='white')
    ax1.tick_params(colors='white')
    cbar1 = fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
    cbar1.ax.yaxis.set_tick_params(color='white')
    cbar1.ax.set_ylabel('Amplitude (dB)', color='white')

    # Bottom plot: Mel spectrogram (better matches human perception)
    mel_spect = librosa.feature.melspectrogram(y=amplitude, sr=sample_rate)
    mel_db = librosa.power_to_db(mel_spect, ref=np.max)
    img2 = librosa.display.specshow(mel_db,
                                sr=sample_rate,
                                x_axis='time',
                                y_axis='mel',
                                ax=ax2,
                                cmap='magma')
    ax2.set_title('Mel Spectrogram', color='white')
    ax2.tick_params(colors='white')
    cbar2 = fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
    cbar2.ax.yaxis.set_tick_params(color='white')
    cbar2.ax.set_ylabel('Amplitude (dB)', color='white')

    # Customize spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('#404040')

    plt.tight_layout()
    plt.show()            

def plot_mel_conversion():
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Generate frequencies
    freq_hz = np.linspace(0, 8000, 1000)
    freq_mel = 2595 * np.log10(1 + freq_hz/700)

    # Plot Hz to Mel relationship
    ax1.plot(freq_hz, freq_mel, color='#00ff9f', linewidth=2)
    ax1.set_title('Frequency (Hz) to Mel Scale', color='white')
    ax1.set_xlabel('Frequency (Hz)', color='white')
    ax1.set_ylabel('Mel Scale', color='white')
    ax1.grid(True, color='#404040')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#404040')

    # Plot some reference points
    reference_hz = [250, 500, 1000, 2000, 4000, 8000]
    reference_mel = 2595 * np.log10(1 + np.array(reference_hz)/700)
    ax1.scatter(reference_hz, reference_mel, color='#ff00ff', zorder=5)
    for hz, mel in zip(reference_hz, reference_mel):
        ax1.annotate(f'{hz}Hz', (hz, mel), color='white', 
                    xytext=(10, 10), textcoords='offset points')

    # Plot the same relationship on log scale for x-axis
    ax2.plot(freq_hz, freq_mel, color='#00ff9f', linewidth=2)
    ax2.set_title('Frequency (Hz) to Mel Scale (Log Scale)', color='white')
    ax2.set_xlabel('Frequency (Hz)', color='white')
    ax2.set_ylabel('Mel Scale', color='white')
    ax2.set_xscale('log')
    ax2.grid(True, color='#404040')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#404040')

    # Plot reference points on log scale too
    ax2.scatter(reference_hz, reference_mel, color='#ff00ff', zorder=5)
    for hz, mel in zip(reference_hz, reference_mel):
        ax2.annotate(f'{hz}Hz', (hz, mel), color='white', 
                    xytext=(10, 10), textcoords='offset points')

    plt.tight_layout()
    plt.show()    

def plot_vocal_range_spectrograms(amplitude, S_db, sample_rate):
    # Create zoomed views focusing on vocal ranges
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Regular spectrogram zoomed to vocal range
    img1 = librosa.display.specshow(S_db, 
                            sr=sample_rate,
                            x_axis='time',
                            y_axis='log',  # Log scale to better see harmonics
                            ax=ax1,
                            cmap='magma',
                            vmin=-60)  # Adjust contrast to better see details
    ax1.set_ylim([80, 8000])  # Set the y-axis limits after creating the plot
    ax1.set_title('Spectrogram (Zoomed to Vocal Range)', color='white')
    ax1.tick_params(colors='white')
    cbar1 = fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
    cbar1.ax.yaxis.set_tick_params(color='white')

    # Mel spectrogram zoomed similarly
    mel_spect = librosa.feature.melspectrogram(y=amplitude, sr=sample_rate, n_mels=256)  # More mel bands for detail
    mel_db = librosa.power_to_db(mel_spect, ref=np.max)
    img2 = librosa.display.specshow(mel_db,
                                sr=sample_rate,
                                x_axis='time',
                                y_axis='mel',
                                ax=ax2,
                                cmap='magma',
                                vmin=-60)
    ax2.set_title('Mel Spectrogram (Focused on Vocal Range)', color='white')
    ax2.tick_params(colors='white')
    cbar2 = fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
    cbar2.ax.yaxis.set_tick_params(color='white')

    # Customize spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('#404040')

    plt.tight_layout()
    plt.show()
    
def plot_mfccs(mfccs, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, 
                            sr=sample_rate, 
                            x_axis='time',
                            cmap='plasma')  # Using plasma colormap which works well for MFCCs

    # Customize colorbar
    cbar = plt.colorbar(format='%+2.0f')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.set_ylabel('MFCC Amplitude', color='white')

    # Set title and customize appearance
    plt.title('MFCCs', color='white')

    # Customize grid and spines for better visibility
    for spine in plt.gca().spines.values():
        spine.set_color('#404040')

    # Add y-axis label
    plt.ylabel('MFCC Coefficients', color='white')

    plt.show()        

def visualize_framing_process(y, sr, start_time=1.0):
    """
    Visualize the audio framing process in detail
    
    Parameters:
    y: audio signal
    sr: sample rate
    start_time: time in seconds to start visualization (default 1.0)
    """
    # Frame parameters
    frame_length = 2048  # ~46ms at 44.1kHz
    hop_length = 512    # 75% overlap
    
    # Calculate some values for visualization
    samples_per_sec = sr
    start_sample = int(start_time * sr)
    
    # Get several consecutive frames
    num_frames = 4
    frames = []
    frame_starts = []
    
    for i in range(num_frames):
        frame_start = start_sample + (i * hop_length)
        frame_starts.append(frame_start)
        frames.append(y[frame_start:frame_start + frame_length])
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # 1. Show full signal context
    plt.subplot(4, 1, 1)
    time_axis = np.arange(len(y)) / sr
    plt.plot(time_axis, y)
    
    # Highlight the region we're examining
    region_start = start_time
    region_end = start_time + (frame_length + (num_frames-1)*hop_length)/sr
    plt.axvspan(region_start, region_end, color='yellow', alpha=0.3)
    
    plt.title('Full Audio Signal with Highlighted Analysis Region')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    
    # 2. Zoom in on the region with frame boundaries
    plt.subplot(4, 1, 2)
    zoom_samples = y[start_sample:start_sample + frame_length + (num_frames-1)*hop_length]
    zoom_time = np.arange(len(zoom_samples)) / sr + start_time
    plt.plot(zoom_time, zoom_samples)
    
    # Show frame boundaries
    for i, frame_start in enumerate(frame_starts):
        frame_time_start = frame_start / sr
        frame_time_end = (frame_start + frame_length) / sr
        plt.axvspan(frame_time_start, frame_time_end, 
                   color=f'C{i+1}', alpha=0.2, 
                   label=f'Frame {i+1}')
    
    plt.title('Zoomed Region Showing Frame Overlap')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # 3. Individual frames with windows
    plt.subplot(4, 1, 3)
    window = hamming(frame_length)
    
    for i, frame in enumerate(frames[:2]):  # Show just first 2 frames for clarity
        frame_time = np.arange(len(frame)) / sr
        plt.plot(frame_time, frame + i*0.5, 
                label=f'Frame {i+1}', 
                alpha=0.7)
        plt.plot(frame_time, window*np.max(np.abs(frame))*0.5 + i*0.5, 
                '--', label=f'Window {i+1}', 
                alpha=0.7)
    
    plt.title('Individual Frames with Hamming Windows')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # 4. Windowed frames
    plt.subplot(4, 1, 4)
    
    for i, frame in enumerate(frames[:2]):
        windowed = frame * window
        frame_time = np.arange(len(frame)) / sr
        plt.plot(frame_time, windowed + i*0.5, 
                label=f'Windowed Frame {i+1}')
    
    plt.title('Frames After Windowing')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def explain_framing_math():
    """
    Print detailed mathematical explanation of the framing process
    """
    explanation = """
    AUDIO FRAMING MATHEMATICAL EXPLANATION
    
    1. Frame Length Selection:
       - Frame length (N) = 2048 samples
       - At 44.1kHz sampling rate, this is ~46ms: (2048/44100) seconds
       - Chosen to be a power of 2 for efficient FFT
       - Long enough to capture lowest frequencies of interest
       - Short enough to assume signal is relatively stationary
    
    2. Hop Length Selection:
       - Hop length (H) = 512 samples
       - Creates 75% overlap: (2048-512)/2048 = 0.75
       - Overlap helps prevent artifacts and ensure smooth analysis
    
    3. Frame Extraction:
       For frame index i:
       - Start sample: s[i] = i * H
       - Frame samples: x[i][n] = x[s[i] + n], n = 0 to N-1
    
    4. Hamming Window:
       w[n] = 0.54 - 0.46 * cos(2πn/(N-1))
       where n goes from 0 to N-1
    
    5. Windowed Frame:
       x_windowed[i][n] = x[i][n] * w[n]
    
    KEY RELATIONSHIPS:
    - Number of frames = ⌊(L - N)/H + 1⌋
      where L is signal length
    - Time of frame i = (i * H)/sr seconds
    - Frequency resolution = sr/N Hz
    
    PRACTICAL IMPLICATIONS:
    - 2048 samples gives frequency resolution of ~21.5 Hz at 44.1kHz
    - 512 hop length means we get a new frame every ~11.6ms
    - 75% overlap helps ensure we don't miss transient events
    """
    print(explanation)

def plot_window_properties(frame_length=2048):
    """
    Visualize properties of the Hamming window
    """
    window = hamming(frame_length)
    
    plt.figure(figsize=(15, 8))
    
    # Time domain
    plt.subplot(1, 2, 1)
    plt.plot(window)
    plt.title('Hamming Window (Time Domain)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Frequency domain
    window_fft = np.abs(np.fft.rfft(window))
    window_db = 20 * np.log10(window_fft/np.max(window_fft))
    freq = np.linspace(0, 0.5, len(window_db))
    
    plt.subplot(1, 2, 2)
    plt.plot(freq, window_db)
    plt.title('Hamming Window Frequency Response')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.ylim(-120, 5)
    
    plt.tight_layout()
    plt.show()    

def create_test_signal(duration=3, sr=22050):
    """Create a test signal with multiple frequencies"""
    t = np.linspace(0, duration, int(sr * duration))
    # Create a signal with two frequencies
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    return signal, sr

def set_style():
    """Set consistent style parameters for better visualization"""
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = '#1A1A1A'
    plt.rcParams['axes.facecolor'] = '#1A1A1A'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#404040'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = '#404040'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.3
    
    return {
        'frame': '#00A7E1',      # Bright blue
        'window': '#FFA400',     # Bright orange
        'windowed': '#7FBA00',   # Bright green
        'grid': '#404040'        # Dark gray
    }

def visualize_windowing_effect(y, sr, start_time=1.0):
    """
    Visualize how the Hamming window affects a single frame of audio
    
    Parameters:
    y: audio signal array
    sr: sample rate
    start_time: time in seconds to start visualization (default 1.0)
    """
    colors = set_style()
    frame_length = 2048
    start_sample = int(start_time * sr)
    
    # Ensure we don't go beyond signal length
    if start_sample + frame_length > len(y):
        start_sample = len(y) - frame_length
        print(f"Adjusted start time to {start_sample/sr:.2f} seconds to fit signal length")
    
    # Get a single frame
    frame = y[start_sample:start_sample + frame_length]
    
    # Create normalized Hamming window
    window = hamming(len(frame))
    window = window / np.sum(window[::frame_length//4])  # Normalize for 75% overlap
    
    windowed_frame = frame * window
    
    # Create time axis in milliseconds
    time_ms = np.arange(len(frame)) * 1000 / sr
    
    fig = plt.figure(figsize=(15, 12))
    
    plot_params = {
        'linewidth': 2.5,
        'alpha': 0.9
    }
    
    # 1. Original Frame
    ax1 = plt.subplot(4, 1, 1)
    plt.plot(time_ms, frame, color=colors['frame'], **plot_params)
    plt.title('Original Audio Frame', pad=20, fontsize=14, color='white')
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, color=colors['grid'], linestyle=':', alpha=0.3)
    
    # 2. Hamming Window
    ax2 = plt.subplot(4, 1, 2)
    plt.plot(time_ms, window, color=colors['window'], **plot_params)
    plt.title('Hamming Window', pad=20, fontsize=14, color='white')
    plt.ylabel('Window\nAmplitude', fontsize=12)
    plt.grid(True, color=colors['grid'], linestyle=':', alpha=0.3)
    
    # 3. Overlay of Frame and Window
    ax3 = plt.subplot(4, 1, 3)
    plt.plot(time_ms, frame, color=colors['frame'], 
            label='Original Frame', **plot_params)
    plt.plot(time_ms, window * np.max(np.abs(frame)), 
            color=colors['window'], linestyle='--', 
            label='Scaled Window', **plot_params)
    plt.title('Original Frame with Window Overlay', pad=20, fontsize=14, color='white')
    plt.ylabel('Amplitude', fontsize=12)
    legend = plt.legend(fontsize=12, framealpha=0.8)
    for text in legend.get_texts():
        text.set_color('white')
    plt.grid(True, color=colors['grid'], linestyle=':', alpha=0.3)
    
    # 4. Windowed Result
    ax4 = plt.subplot(4, 1, 4)
    plt.plot(time_ms, windowed_frame, color=colors['windowed'], **plot_params)
    plt.title('Final Windowed Frame (Original × Window)', pad=20, fontsize=14, color='white')
    plt.xlabel('Time (milliseconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, color=colors['grid'], linestyle=':', alpha=0.3)
    
    # Add connecting arrows
    plt.figtext(0.02, 0.5, '→ Multiplication →', rotation=90, 
               va='center', ha='center', fontsize=14, color='white',
               bbox=dict(facecolor='#1A1A1A', alpha=0.8, pad=5,
                        edgecolor='#404040'))
    
    plt.tight_layout()
    
    # Add explanation text
    explanation = f"""
    Frame Analysis at {start_time:.2f}s:
    • Frame length: {frame_length} samples ({frame_length/sr*1000:.1f}ms)
    • Window: Normalized Hamming
    • Sample rate: {sr} Hz
    
    Signal stats for this frame:
    • Max amplitude: {np.max(np.abs(frame)):.3f}
    • RMS amplitude: {np.sqrt(np.mean(frame**2)):.3f}
    """
    plt.figtext(0.02, 0.02, explanation, fontsize=12, color='white',
                bbox=dict(facecolor='#1A1A1A', alpha=0.95, 
                         pad=10, edgecolor='#404040'))
    
    plt.show()

# Usage example:
# visualize_windowing_effect(amplitude, sample_rate, start_time=1.0)

def plot_overlap_effect():
    """
    Visualize how overlapping windowed frames sum to reconstruct the signal
    """
    colors = set_style()
    
    # Create a simple synthetic signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    
    # Frame parameters
    frame_length = 200
    hop_length = 50  # 75% overlap
    
    # Create normalized Hamming window
    # The window should sum to 1.0 when overlapped properly
    window = hamming(frame_length)
    window = window / np.sum(window[::hop_length])  # Normalize for perfect reconstruction
    
    # Pre-allocate arrays
    window_sum = np.zeros(400)  # Just show first 400 samples for clarity
    frames = []
    
    # Process frames that affect the first 400 samples
    for i in range(8):  # Increased number of frames to show full overlap
        start = i * hop_length
        if start < 400:  # Only process frames that affect our display window
            # Add window to the sum at the correct position
            end = min(start + frame_length, 400)
            window_sum[start:end] += window[:end-start]
            
            # Get frame for display (but only keep first 4 for clarity)
            frame = signal[start:start + frame_length]
            if len(frame) == frame_length and i < 4:
                frames.append(frame * window)
    
    # Plotting
    fig = plt.figure(figsize=(15, 10))
    
    plot_params = {
        'linewidth': 2.5,
        'alpha': 0.9
    }
    
    # 1. Show first few frames
    plt.subplot(3, 1, 1)
    frame_colors = ['#00A7E1', '#FFA400', '#7FBA00', '#FF69B4']
    for i, frame in enumerate(frames):
        start = i * hop_length
        plt.plot(np.arange(start, start + len(frame)), frame, 
                color=frame_colors[i], label=f'Frame {i+1}', **plot_params)
    plt.title('Individual Overlapping Windowed Frames', pad=20, fontsize=14, color='white')
    legend = plt.legend(fontsize=12, framealpha=0.8)
    for text in legend.get_texts():
        text.set_color('white')
    plt.ylim(-1.1, 1.1)
    
    # 2. Show correct window overlap
    plt.subplot(3, 1, 2)
    plt.plot(window_sum, color='#FFA400', label='Window Sum', **plot_params)
    plt.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='Perfect Overlap = 1.0')
    plt.title('Window Overlap (First 400 Samples)', pad=20, fontsize=14, color='white')
    legend = plt.legend(fontsize=12, framealpha=0.8)
    for text in legend.get_texts():
        text.set_color('white')
    plt.ylim(0.0, 1.2)  # Adjusted to show overlap more clearly
    
    # 3. Show reconstruction (full signal)
    reconstruction = np.zeros_like(signal)
    full_window_sum = np.zeros_like(signal)
    
    # Process all frames for full reconstruction
    for i in range((len(signal) - frame_length) // hop_length + 1):
        start = i * hop_length
        frame = signal[start:start + frame_length]
        if len(frame) == frame_length:
            windowed_frame = frame * window
            reconstruction[start:start + frame_length] += windowed_frame
            full_window_sum[start:start + frame_length] += window
    
    # Normalize
    reconstruction /= np.maximum(full_window_sum, 1e-10)
    
    plt.subplot(3, 1, 3)
    plt.plot(signal, color='#00A7E1', label='Original Signal', 
            linewidth=2.5, alpha=0.7)
    plt.plot(reconstruction, color='#7FBA00', 
            label='Reconstructed Signal', 
            linewidth=2.5, alpha=0.9)
    plt.title('Signal Reconstruction from Overlapping Frames', 
             pad=20, fontsize=14, color='white')
    legend = plt.legend(fontsize=12, framealpha=0.8)
    for text in legend.get_texts():
        text.set_color('white')
    plt.ylim(-1.1, 1.1)
    
    # Add explanation text
    explanation = """
    Overlap-Add Reconstruction:
    • Normalized Hamming window ensures
      sum equals 1.0 in steady state
    • 75% overlap (hop = window_length/4)
    • Perfect reconstruction achieved
      where window sum = 1.0
    
    Note the smooth transition at edges
    and perfect reconstruction in the
    middle region.
    """
    plt.figtext(0.02, 0.02, explanation, fontsize=12, color='white',
                bbox=dict(facecolor='#1A1A1A', alpha=0.95, 
                         pad=10, edgecolor='#404040'))
    
    plt.tight_layout()
    plt.show()    

def analyze_window_coverage():
    """
    Analyze and visualize window coverage and reconstruction error
    """
    set_style()
    
    # Parameters
    frame_length = 200
    hop_length = 50  # 75% overlap
    signal_length = 1000
    
    # Create normalized Hamming window
    window = hamming(frame_length)
    window = window / np.sum(window[::hop_length])
    
    # Calculate window sum across entire signal
    window_sum = np.zeros(signal_length)
    
    # Calculate theoretical number of full frames needed
    n_frames = (signal_length - frame_length) // hop_length + 1
    
    # Calculate window sum and keep track of contributing frames at each point
    frame_coverage = np.zeros(signal_length)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= signal_length:
            window_sum[start:end] += window
            frame_coverage[start:end] += 1
    
    # Calculate theoretical error bounds
    # Error is proportional to |1 - window_sum|
    reconstruction_error = np.abs(1 - window_sum)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Window Sum across entire signal
    plt.subplot(3, 1, 1)
    plt.plot(window_sum, color='#FFA400', linewidth=2.5, label='Window Sum')
    plt.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='Perfect Coverage (1.0)')
    plt.title('Window Sum Across Entire Signal', pad=20, fontsize=14, color='white')
    plt.ylabel('Sum Value')
    plt.legend(fontsize=12, framealpha=0.8)
    plt.grid(True)
    
    # 2. Number of overlapping frames
    plt.subplot(3, 1, 2)
    plt.plot(frame_coverage, color='#00A7E1', linewidth=2.5, label='Overlapping Frames')
    plt.title('Number of Overlapping Frames', pad=20, fontsize=14, color='white')
    plt.ylabel('Frame Count')
    plt.legend(fontsize=12, framealpha=0.8)
    plt.grid(True)
    
    # 3. Reconstruction error
    plt.subplot(3, 1, 3)
    plt.semilogy(reconstruction_error, color='#7FBA00', linewidth=2.5, label='Reconstruction Error')
    plt.title('Theoretical Reconstruction Error (Log Scale)', pad=20, fontsize=14, color='white')
    plt.ylabel('Error (|1 - window_sum|)')
    plt.xlabel('Sample Number')
    plt.legend(fontsize=12, framealpha=0.8)
    plt.grid(True)
    
    # Add analytical explanation
    explanation = f"""
    Perfect Reconstruction Analysis:
    
    • Ramp-up region: [0, {frame_length-1}] samples
    • Perfect reconstruction: [{frame_length}, {signal_length-frame_length}] samples
    • Ramp-down region: [{signal_length-frame_length+1}, {signal_length-1}] samples
    
    For 75% overlap (hop={hop_length}):
    • Frames needed for perfect reconstruction: {frame_length//hop_length}
    • Maximum overlapping frames: {int(frame_length/hop_length)}
    • Theoretical error in perfect region: {np.min(reconstruction_error[frame_length:-frame_length]):.2e}
    
    Error bounds are symmetric at start/end due to
    window symmetry and linear frame accumulation.
    """
    
    plt.figtext(0.02, 0.02, explanation, fontsize=12, color='white',
                bbox=dict(facecolor='#1A1A1A', alpha=0.95, 
                         pad=10, edgecolor='#404040'))
    
    plt.tight_layout()
    plt.show()    