import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal.windows import hamming

class StyleManager:
    """Manages consistent styling for all visualizations"""
    
    @staticmethod
    def set_style():
        """Set consistent dark theme style parameters"""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': '#1A1A1A',
            'axes.facecolor': '#1A1A1A',
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'axes.edgecolor': '#404040',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'grid.color': '#404040',
            'grid.linestyle': ':',
            'grid.alpha': 0.3
        })
        
        return {
            'frame': '#00A7E1',      # Bright blue
            'window': '#FFA400',     # Bright orange
            'windowed': '#7FBA00',   # Bright green
            'grid': '#404040'        # Dark gray
        }
    
    @staticmethod
    def customize_spines(ax):
        """Apply consistent spine styling to an axis"""
        for spine in ax.spines.values():
            spine.set_color('#404040')
    
    @staticmethod
    def customize_colorbar(cbar, label='Amplitude (dB)'):
        """Apply consistent styling to a colorbar"""
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.set_ylabel(label, color='white')
    
    @staticmethod
    def add_legend(ax, fontsize=12):
        """Add and style a legend"""
        legend = ax.legend(fontsize=fontsize, framealpha=0.8)
        for text in legend.get_texts():
            text.set_color('white')
        return legend

class AudioPlotter:
    """Base class for audio visualization"""
    
    def __init__(self):
        self.colors = StyleManager.set_style()
    
    def create_figure(self, figsize=(12, 4)):
        """Create a new figure with consistent styling"""
        return plt.figure(figsize=figsize)
    
    def show_plot(self):
        """Display the plot with consistent layout"""
        plt.tight_layout()
        plt.show()

class WaveformPlotter(AudioPlotter):
    """Handles waveform visualization"""
    
    def plot_waveform(self, amplitude, sample_rate):
        """Display audio waveform"""
        self.create_figure()
        librosa.display.waveshow(amplitude, sr=sample_rate, color='#00ff9f')
        plt.title('Audio Waveform', color='white')
        plt.grid(True, color='#303030')
        StyleManager.customize_spines(plt.gca())
        self.show_plot()

class SpectrogramPlotter(AudioPlotter):
    """Handles spectrogram visualization"""
    
    def plot_spectrogram(self, S_db, sample_rate, title='Spectrogram'):
        """Display basic spectrogram"""
        self.create_figure()
        img = librosa.display.specshow(
            S_db, sr=sample_rate, x_axis='time', y_axis='hz', cmap='magma'
        )
        cbar = plt.colorbar(img, format='%+2.0f dB')
        StyleManager.customize_colorbar(cbar)
        plt.title(title, color='white')
        StyleManager.customize_spines(plt.gca())
        self.show_plot()
    
    def plot_log_spectrograms(self, amplitude, S_db, sample_rate):
        """Display multiple spectrogram views"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Log frequency spectrogram
        img1 = librosa.display.specshow(
            S_db, sr=sample_rate, x_axis='time', y_axis='log', ax=ax1, cmap='magma'
        )
        ax1.set_title('Spectrogram (Log Frequency Scale)', color='white')
        ax1.tick_params(colors='white')
        cbar1 = fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
        StyleManager.customize_colorbar(cbar1)
        
        # Mel spectrogram
        mel_spect = librosa.feature.melspectrogram(y=amplitude, sr=sample_rate)
        mel_db = librosa.power_to_db(mel_spect, ref=np.max)
        img2 = librosa.display.specshow(
            mel_db, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax2, cmap='magma'
        )
        ax2.set_title('Mel Spectrogram', color='white')
        ax2.tick_params(colors='white')
        cbar2 = fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
        StyleManager.customize_colorbar(cbar2)
        
        for ax in [ax1, ax2]:
            StyleManager.customize_spines(ax)
        
        self.show_plot()

class FramingVisualizer(AudioPlotter):
    """Base class for framing visualizations"""
    
    def __init__(self, frame_length=2048, hop_length=512):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = self._create_normalized_window()
    
    def _create_normalized_window(self):
        """Create a normalized Hamming window"""
        window = hamming(self.frame_length)
        return window / np.sum(window[::self.hop_length])
    
    def visualize_windowing_effect(self, y, sr, start_time=1.0):
        """Visualize how windowing affects a single frame"""
        colors = self.colors
        start_sample = int(start_time * sr)
        
        # Ensure we don't go beyond signal length
        if start_sample + self.frame_length > len(y):
            start_sample = len(y) - self.frame_length
            print(f"Adjusted start time to {start_sample/sr:.2f} seconds to fit signal length")
        
        # Get a single frame
        frame = y[start_sample:start_sample + self.frame_length]
        windowed_frame = frame * self.window
        
        # Create time axis in milliseconds
        time_ms = np.arange(len(frame)) * 1000 / sr
        
        # Create visualization
        fig = plt.figure(figsize=(15, 12))
        plot_params = {'linewidth': 2.5, 'alpha': 0.9}
        
        # 1. Original Frame
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(time_ms, frame, color=colors['frame'], **plot_params)
        ax1.set_title('Original Audio Frame', pad=20, fontsize=14, color='white')
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.grid(True, color=colors['grid'], linestyle=':', alpha=0.3)
        
        # 2. Hamming Window
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(time_ms, self.window, color=colors['window'], **plot_params)
        ax2.set_title('Hamming Window', pad=20, fontsize=14, color='white')
        ax2.set_ylabel('Window\nAmplitude', fontsize=12)
        ax2.grid(True, color=colors['grid'], linestyle=':', alpha=0.3)
        
        # 3. Overlay
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(time_ms, frame, color=colors['frame'], label='Original Frame', **plot_params)
        ax3.plot(time_ms, self.window * np.max(np.abs(frame)), 
                color=colors['window'], linestyle='--', label='Scaled Window', **plot_params)
        ax3.set_title('Original Frame with Window Overlay', pad=20, fontsize=14, color='white')
        ax3.set_ylabel('Amplitude', fontsize=12)
        StyleManager.add_legend(ax3)
        ax3.grid(True, color=colors['grid'], linestyle=':', alpha=0.3)
        
        # 4. Windowed Result
        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(time_ms, windowed_frame, color=colors['windowed'], **plot_params)
        ax4.set_title('Final Windowed Frame (Original × Window)', pad=20, fontsize=14, color='white')
        ax4.set_xlabel('Time (milliseconds)', fontsize=12)
        ax4.set_ylabel('Amplitude', fontsize=12)
        ax4.grid(True, color=colors['grid'], linestyle=':', alpha=0.3)
        
        # Add connecting arrows
        plt.figtext(0.02, 0.5, '→ Multiplication →', rotation=90, 
                   va='center', ha='center', fontsize=14, color='white',
                   bbox=dict(facecolor='#1A1A1A', alpha=0.8, pad=5,
                            edgecolor='#404040'))
        
        # Add explanation text
        explanation = f"""
        Frame Analysis at {start_time:.2f}s:
        • Frame length: {self.frame_length} samples ({self.frame_length/sr*1000:.1f}ms)
        • Window: Normalized Hamming
        • Sample rate: {sr} Hz
        
        Signal stats for this frame:
        • Max amplitude: {np.max(np.abs(frame)):.3f}
        • RMS amplitude: {np.sqrt(np.mean(frame**2)):.3f}
        """
        plt.figtext(0.02, 0.02, explanation, fontsize=12, color='white',
                    bbox=dict(facecolor='#1A1A1A', alpha=0.95, 
                             pad=10, edgecolor='#404040'))
        
        self.show_plot()
        
def create_test_signal(duration=3, sr=22050):
    """Create a test signal with multiple frequencies"""
    t = np.linspace(0, duration, int(sr * duration))
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    return signal, sr

class AmplitudePlotter(AudioPlotter):
    """Handles amplitude distribution visualizations"""
    
    def plot_amplitude_distributions(self, amplitude):
        """Plot various views of amplitude distribution"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        axes = [ax1, ax2, ax3, ax4]
        
        # Configure plots
        plots = [
            (amplitude, 'Original Amplitude Distribution (Linear)', '#00ff9f', False),
            (np.abs(amplitude), 'Absolute Amplitude Distribution (Linear)', '#ff00ff', False),
            (amplitude, 'Original Amplitude Distribution (Log Scale)', '#00ff9f', True),
            (np.abs(amplitude), 'Absolute Amplitude Distribution (Log Scale)', '#ff00ff', True)
        ]
        
        for ax, (data, title, color, log_scale) in zip(axes, plots):
            sns.histplot(data, bins=100, color=color, alpha=0.6, ax=ax, 
                        stat='probability', common_norm=False)
            ax.set_title(title, color='white')
            ax.set_xlabel('Amplitude' if 'Original' in title else 'Absolute Amplitude', 
                         color='white')
            ax.set_ylabel('Percentage' + (' (Log Scale)' if log_scale else ''), 
                         color='white')
            ax.tick_params(colors='white')
            if log_scale:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(
                    lambda y, _: '{:.1e}%'.format(y * 100)))
            else:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(
                    lambda y, _: '{:.1f}%'.format(y * 100)))
            StyleManager.customize_spines(ax)
        
        self.show_plot()

class MelVisualizer(AudioPlotter):
    """Handles Mel-scale visualizations"""
    
    def plot_mel_conversion(self):
        """Visualize the conversion between Hz and Mel scales"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Generate frequencies
        freq_hz = np.linspace(0, 8000, 1000)
        freq_mel = 2595 * np.log10(1 + freq_hz/700)
        reference_hz = [250, 500, 1000, 2000, 4000, 8000]
        reference_mel = 2595 * np.log10(1 + np.array(reference_hz)/700)
        
        # Plot linear scale
        self._plot_mel_scale(ax1, freq_hz, freq_mel, reference_hz, reference_mel, False)
        # Plot log scale
        self._plot_mel_scale(ax2, freq_hz, freq_mel, reference_hz, reference_mel, True)
        
        self.show_plot()
    
    def _plot_mel_scale(self, ax, freq_hz, freq_mel, ref_hz, ref_mel, log_scale):
        """Helper method for plotting mel scale conversion"""
        ax.plot(freq_hz, freq_mel, color='#00ff9f', linewidth=2)
        ax.scatter(ref_hz, ref_mel, color='#ff00ff', zorder=5)
        
        for hz, mel in zip(ref_hz, ref_mel):
            ax.annotate(f'{hz}Hz', (hz, mel), color='white', 
                       xytext=(10, 10), textcoords='offset points')
        
        ax.set_title(f'Frequency (Hz) to Mel Scale{" (Log Scale)" if log_scale else ""}', 
                    color='white')
        ax.set_xlabel('Frequency (Hz)', color='white')
        ax.set_ylabel('Mel Scale', color='white')
        if log_scale:
            ax.set_xscale('log')
        ax.grid(True, color='#404040')
        ax.tick_params(colors='white')
        StyleManager.customize_spines(ax)

class VocalRangeVisualizer(SpectrogramPlotter):
    """Handles vocal range spectrogram visualizations"""
    
    def plot_vocal_range_spectrograms(self, amplitude, S_db, sample_rate):
        """Plot spectrograms focused on vocal range"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Regular spectrogram zoomed to vocal range
        img1 = librosa.display.specshow(S_db, sr=sample_rate, x_axis='time',
                                      y_axis='log', ax=ax1, cmap='magma', vmin=-60)
        ax1.set_ylim([80, 8000])
        self._setup_vocal_range_plot(fig, ax1, img1, 'Spectrogram (Zoomed to Vocal Range)')
        
        # Mel spectrogram zoomed similarly
        mel_spect = librosa.feature.melspectrogram(y=amplitude, sr=sample_rate, n_mels=256)
        mel_db = librosa.power_to_db(mel_spect, ref=np.max)
        img2 = librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time',
                                      y_axis='mel', ax=ax2, cmap='magma', vmin=-60)
        self._setup_vocal_range_plot(fig, ax2, img2, 'Mel Spectrogram (Focused on Vocal Range)')
        
        self.show_plot()
    
    def _setup_vocal_range_plot(self, fig, ax, img, title):
        """Helper method for setting up vocal range plots"""
        ax.set_title(title, color='white')
        ax.tick_params(colors='white')
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        StyleManager.customize_colorbar(cbar)
        StyleManager.customize_spines(ax)

class MFCCPlotter(AudioPlotter):
    """Handles MFCC visualizations"""
    
    def plot_mfccs(self, mfccs, sample_rate):
        """Plot Mel-frequency cepstral coefficients"""
        self.create_figure()
        img = librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time',
                                     cmap='plasma')
        
        cbar = plt.colorbar(img, format='%+2.0f')
        StyleManager.customize_colorbar(cbar, label='MFCC Amplitude')
        
        plt.title('MFCCs', color='white')
        plt.ylabel('MFCC Coefficients', color='white')
        StyleManager.customize_spines(plt.gca())
        
        self.show_plot()

class FramingProcessVisualizer(FramingVisualizer):
    """Advanced framing process visualization"""
    
    def visualize_framing_process(self, y, sr, start_time=1.0):
        """Visualize the complete audio framing process"""
        samples_per_sec = sr
        start_sample = int(start_time * sr)
        
        # Get consecutive frames
        num_frames = 4
        frames = []
        frame_starts = []
        
        for i in range(num_frames):
            frame_start = start_sample + (i * self.hop_length)
            frame_starts.append(frame_start)
            frames.append(y[frame_start:frame_start + self.frame_length])
        
        self._plot_framing_process(y, frames, frame_starts, start_time, sr)
    
    def _plot_framing_process(self, y, frames, frame_starts, start_time, sr):
        """Helper method for plotting the framing process"""
        plt.figure(figsize=(15, 12))
        
        # Full signal context
        ax1 = plt.subplot(4, 1, 1)
        time_axis = np.arange(len(y)) / sr
        ax1.plot(time_axis, y)
        region_start = start_time
        region_end = start_time + (self.frame_length + (len(frames)-1)*self.hop_length)/sr
        ax1.axvspan(region_start, region_end, color='yellow', alpha=0.3)
        ax1.set_title('Full Audio Signal with Highlighted Analysis Region')
        
        # Zoomed region
        ax2 = plt.subplot(4, 1, 2)
        zoom_samples = y[frame_starts[0]:frame_starts[0] + 
                        self.frame_length + (len(frames)-1)*self.hop_length]
        zoom_time = np.arange(len(zoom_samples)) / sr + start_time
        ax2.plot(zoom_time, zoom_samples)
        
        for i, frame_start in enumerate(frame_starts):
            frame_time_start = frame_start / sr
            frame_time_end = (frame_start + self.frame_length) / sr
            ax2.axvspan(frame_time_start, frame_time_end, 
                       color=f'C{i+1}', alpha=0.2, label=f'Frame {i+1}')
        
        ax2.set_title('Zoomed Region Showing Frame Overlap')
        StyleManager.add_legend(ax2)
        
        # Individual frames with windows
        ax3 = plt.subplot(4, 1, 3)
        self._plot_individual_frames(ax3, frames[:2], sr)
        
        # Windowed frames
        ax4 = plt.subplot(4, 1, 4)
        self._plot_windowed_frames(ax4, frames[:2], sr)
        
        self.show_plot()
    
    def _plot_individual_frames(self, ax, frames, sr):
        """Plot individual frames with their windows"""
        for i, frame in enumerate(frames):
            frame_time = np.arange(len(frame)) / sr
            ax.plot(frame_time, frame + i*0.5, label=f'Frame {i+1}', alpha=0.7)
            ax.plot(frame_time, self.window*np.max(np.abs(frame))*0.5 + i*0.5, 
                   '--', label=f'Window {i+1}', alpha=0.7)
        
        ax.set_title('Individual Frames with Hamming Windows')
        StyleManager.add_legend(ax)
    
    def _plot_windowed_frames(self, ax, frames, sr):
        """Plot frames after windowing"""
        for i, frame in enumerate(frames):
            windowed = frame * self.window
            frame_time = np.arange(len(frame)) / sr
            ax.plot(frame_time, windowed + i*0.5, label=f'Windowed Frame {i+1}')
        
        ax.set_title('Frames After Windowing')
        StyleManager.add_legend(ax)

class OverlapAnalyzer(FramingVisualizer):
    """Analyzes and visualizes frame overlap effects"""
    
    def plot_overlap_effect(self):
        """Visualize how overlapping windowed frames sum to reconstruct the signal"""
        # Create a simple synthetic signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        
        # Create figure with multiple views
        fig = plt.figure(figsize=(15, 10))
        plot_params = {'linewidth': 2.5, 'alpha': 0.9}
        
        # 1. Show first few frames
        ax1 = plt.subplot(3, 1, 1)
        frame_colors = ['#00A7E1', '#FFA400', '#7FBA00', '#FF69B4']
        
        # Process first few frames
        for i in range(4):
            start = i * self.hop_length
            frame = signal[start:start + self.frame_length]
            if len(frame) == self.frame_length:
                windowed = frame * self.window
                ax1.plot(np.arange(start, start + len(frame)), 
                        windowed, 
                        color=frame_colors[i], 
                        label=f'Frame {i+1}', 
                        **plot_params)
        
        ax1.set_title('Individual Overlapping Windowed Frames', 
                     pad=20, fontsize=14, color='white')
        StyleManager.add_legend(ax1)
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, color='#404040')
        
        # 2. Show window overlap
        ax2 = plt.subplot(3, 1, 2)
        window_sum = np.zeros(400)  # Show first 400 samples
        
        # Calculate window sum
        for i in range(8):  # Process more frames for complete overlap
            start = i * self.hop_length
            if start < 400:
                end = min(start + self.frame_length, 400)
                window_sum[start:end] += self.window[:end-start]
        
        ax2.plot(window_sum, color='#FFA400', label='Window Sum', **plot_params)
        ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, 
                   label='Perfect Overlap = 1.0')
        ax2.set_title('Window Overlap (First 400 Samples)', 
                     pad=20, fontsize=14, color='white')
        StyleManager.add_legend(ax2)
        ax2.set_ylim(0.0, 1.2)
        ax2.grid(True, color='#404040')
        
        # 3. Show reconstruction
        ax3 = plt.subplot(3, 1, 3)
        reconstruction = self._reconstruct_signal(signal)
        
        ax3.plot(signal[:400], color='#00A7E1', 
                label='Original Signal', linewidth=2.5, alpha=0.7)
        ax3.plot(reconstruction[:400], color='#7FBA00', 
                label='Reconstructed Signal', **plot_params)
        ax3.set_title('Signal Reconstruction from Overlapping Frames', 
                     pad=20, fontsize=14, color='white')
        StyleManager.add_legend(ax3)
        ax3.set_ylim(-1.1, 1.1)
        ax3.grid(True, color='#404040')
        
        # Add explanation
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
    
    def _plot_overlap_analysis(self, signal, frames, window_sum):
        """Plot overlap analysis results"""
        fig = plt.figure(figsize=(15, 10))
        plot_params = {'linewidth': 2.5, 'alpha': 0.9}
        
        # Individual frames
        ax1 = plt.subplot(3, 1, 1)
        frame_colors = ['#00A7E1', '#FFA400', '#7FBA00', '#FF69B4']
        for i, frame in enumerate(frames):
            ax1.plot(np.arange(i * self.hop_length, 
                             i * self.hop_length + len(frame)), 
                    frame, color=frame_colors[i], 
                    label=f'Frame {i+1}', **plot_params)
        
        ax1.set_title('Individual Overlapping Windowed Frames', 
                     pad=20, fontsize=14, color='white')
        StyleManager.add_legend(ax1)
        ax1.set_ylim(-1.1, 1.1)
        
        # Window overlap
        ax2 = plt.subplot(3, 1, 2)
        self._plot_window_sum(ax2, window_sum, plot_params)
        
        # Signal reconstruction
        ax3 = plt.subplot(3, 1, 3)
        self._plot_reconstruction(ax3, signal, plot_params)
        
        self._add_overlap_explanation()
        self.show_plot()
    
    def _plot_window_sum(self, ax, window_sum, plot_params):
        """Plot window sum analysis"""
        ax.plot(window_sum, color='#FFA400', label='Window Sum', **plot_params)
        ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, 
                  label='Perfect Overlap = 1.0')
        ax.set_title('Window Overlap (First 400 Samples)', 
                    pad=20, fontsize=14, color='white')
        StyleManager.add_legend(ax)
        ax.set_ylim(0.0, 1.2)
    
    def _plot_reconstruction(self, ax, signal, plot_params):
        """Plot signal reconstruction"""
        reconstruction = self._reconstruct_signal(signal)
        ax.plot(signal[:400], color='#00A7E1', 
                label='Original Signal', linewidth=2.5, alpha=0.7)
        ax.plot(reconstruction[:400], color='#7FBA00', 
                label='Reconstructed Signal', **plot_params)
        ax.set_title('Signal Reconstruction from Overlapping Frames', 
                    pad=20, fontsize=14, color='white')
        StyleManager.add_legend(ax)
        ax.set_ylim(-1.1, 1.1)

    def _reconstruct_signal(self, signal):
        """Reconstruct signal from overlapping frames"""
        reconstruction = np.zeros_like(signal)
        window_sum = np.zeros_like(signal)
        
        for i in range((len(signal) - self.frame_length) // self.hop_length + 1):
            start = i * self.hop_length
            frame = signal[start:start + self.frame_length]
            if len(frame) == self.frame_length:
                windowed_frame = frame * self.window
                reconstruction[start:start + self.frame_length] += windowed_frame
                window_sum[start:start + self.frame_length] += self.window
        
        # Normalize by window sum
        reconstruction /= np.maximum(window_sum, 1e-10)
        return reconstruction
    
    def _add_overlap_explanation(self):
        """Add explanation text to overlap analysis plot"""
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
    
    def analyze_window_coverage(self):
        """Analyze and visualize window coverage and reconstruction error"""
        # Parameters
        signal_length = 1000
        
        # Calculate window sum and coverage
        window_sum = np.zeros(signal_length)
        frame_coverage = np.zeros(signal_length)
        
        # Calculate theoretical number of frames
        n_frames = (signal_length - self.frame_length) // self.hop_length + 1
        
        # Calculate window sum and frame coverage
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            if end <= signal_length:
                window_sum[start:end] += self.window
                frame_coverage[start:end] += 1
        
        # Calculate error only for the valid region
        valid_region = slice(self.frame_length, signal_length - self.frame_length)
        min_error = np.min(np.abs(1 - window_sum[valid_region])) if len(window_sum[valid_region]) > 0 else 0
        reconstruction_error = np.abs(1 - window_sum)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # Plot window sum
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(window_sum, color='#FFA400', linewidth=2.5, label='Window Sum')
        ax1.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, 
                   label='Perfect Coverage (1.0)')
        ax1.set_title('Window Sum Across Entire Signal', 
                     pad=20, fontsize=14, color='white')
        ax1.set_ylabel('Sum Value', color='white')
        StyleManager.add_legend(ax1)
        ax1.grid(True, color='#404040')
        
        # Plot frame coverage
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(frame_coverage, color='#00A7E1', linewidth=2.5, 
                label='Overlapping Frames')
        ax2.set_title('Number of Overlapping Frames', 
                     pad=20, fontsize=14, color='white')
        ax2.set_ylabel('Frame Count', color='white')
        StyleManager.add_legend(ax2)
        ax2.grid(True, color='#404040')
        
        # Plot reconstruction error
        ax3 = plt.subplot(3, 1, 3)
        ax3.semilogy(reconstruction_error, color='#7FBA00', linewidth=2.5, 
                    label='Reconstruction Error')
        ax3.set_title('Theoretical Reconstruction Error (Log Scale)', 
                     pad=20, fontsize=14, color='white')
        ax3.set_ylabel('Error (|1 - window_sum|)', color='white')
        ax3.set_xlabel('Sample Number', color='white')
        StyleManager.add_legend(ax3)
        ax3.grid(True, color='#404040')
        
        # Add analysis explanation
        explanation = f"""
        Perfect Reconstruction Analysis:
        
        • Ramp-up region: [0, {self.frame_length-1}] samples
        • Perfect reconstruction: [{self.frame_length}, {signal_length-self.frame_length}] samples
        • Ramp-down region: [{signal_length-self.frame_length+1}, {signal_length-1}] samples
        
        For 75% overlap (hop={self.hop_length}):
        • Frames needed for perfect reconstruction: {self.frame_length//self.hop_length}
        • Maximum overlapping frames: {int(self.frame_length/self.hop_length)}
        • Theoretical error in perfect region: {min_error:.2e}
        
        Error bounds are symmetric at start/end due to
        window symmetry and linear frame accumulation.
        """
        plt.figtext(0.02, 0.02, explanation, fontsize=12, color='white',
                   bbox=dict(facecolor='#1A1A1A', alpha=0.95, 
                            pad=10, edgecolor='#404040'))
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_coverage(self, signal_length):
        """Calculate window sum and frame coverage"""
        window_sum = np.zeros(signal_length)
        frame_coverage = np.zeros(signal_length)
        n_frames = (signal_length - self.frame_length) // self.hop_length + 1
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            if end <= signal_length:
                window_sum[start:end] += self.window
                frame_coverage[start:end] += 1
        
        return window_sum, frame_coverage
    
    def _plot_window_coverage_sum(self, ax, window_sum):
        """Plot window sum analysis"""
        ax.plot(window_sum, color='#FFA400', linewidth=2.5, label='Window Sum')
        ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, 
                  label='Perfect Coverage (1.0)')
        ax.set_title('Window Sum Across Entire Signal', 
                    pad=20, fontsize=14, color='white')
        ax.set_ylabel('Sum Value')
        StyleManager.add_legend(ax)
        ax.grid(True)
    
    def _plot_frame_coverage(self, ax, frame_coverage):
        """Plot frame coverage analysis"""
        ax.plot(frame_coverage, color='#00A7E1', linewidth=2.5, 
               label='Overlapping Frames')
        ax.set_title('Number of Overlapping Frames', 
                    pad=20, fontsize=14, color='white')
        ax.set_ylabel('Frame Count')
        StyleManager.add_legend(ax)
        ax.grid(True)
    
    def _plot_reconstruction_error(self, ax, reconstruction_error):
        """Plot reconstruction error analysis"""
        ax.semilogy(reconstruction_error, color='#7FBA00', linewidth=2.5, 
                   label='Reconstruction Error')
        ax.set_title('Theoretical Reconstruction Error (Log Scale)', 
                    pad=20, fontsize=14, color='white')
        ax.set_ylabel('Error (|1 - window_sum|)')
        ax.set_xlabel('Sample Number')
        StyleManager.add_legend(ax)
        ax.grid(True)
    
    def _add_coverage_analysis_explanation(self, signal_length):
        """Add explanation text to coverage analysis plot"""
        explanation = f"""
        Perfect Reconstruction Analysis:
        
        • Ramp-up region: [0, {self.frame_length-1}] samples
        • Perfect reconstruction: [{self.frame_length}, {signal_length-self.frame_length}] samples
        • Ramp-down region: [{signal_length-self.frame_length+1}, {signal_length-1}] samples
        
        For 75% overlap (hop={self.hop_length}):
        • Frames needed for perfect reconstruction: {self.frame_length//self.hop_length}
        • Maximum overlapping frames: {int(self.frame_length/self.hop_length)}
        • Theoretical error in perfect region: {0.001:.2e}
        
        Error bounds are symmetric at start/end due to
        window symmetry and linear frame accumulation.
        """
        plt.figtext(0.02, 0.02, explanation, fontsize=12, color='white',
                   bbox=dict(facecolor='#1A1A1A', alpha=0.95, 
                           pad=10, edgecolor='#404040'))