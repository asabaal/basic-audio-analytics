from basic_audio_analytics.base import (
    plot_waveform,
    plot_amplitude_distributions,
    plot_spectrogram,
    plot_log_spectrograms,
    plot_mel_conversion,
    plot_vocal_range_spectrograms,
    plot_mfccs,
    visualize_framing_process,
    explain_framing_math,
    plot_window_properties,
    create_test_signal,
    set_style,
    visualize_windowing_effect,
    plot_overlap_effect,
    analyze_window_coverage
)

# Import tempo detection functions
from basic_audio_analytics.tempo import (
    detect_tempo,
    analyze_tempo_distribution
)

__all__ = [
    'plot_waveform',
    'plot_amplitude_distributions',
    'plot_spectrogram',
    'plot_log_spectrograms',
    'plot_mel_conversion',
    'plot_vocal_range_spectrograms',
    'plot_mfccs',
    'visualize_framing_process',
    'explain_framing_math',
    'plot_window_properties',
    'create_test_signal',
    'set_style',
    'visualize_windowing_effect',
    'plot_overlap_effect',
    'analyze_window_coverage',
    'detect_tempo',
    'analyze_tempo_distribution'
]
