"""
Tests for the tempo detection module.
"""

import unittest
import numpy as np
import librosa

from basic_audio_analytics.tempo import (
    detect_tempo,
    analyze_tempo_distribution,
    _get_onset_strength,
    _get_dynamic_tempo,
    _get_tempo_candidates
)

class TestTempoDetection(unittest.TestCase):
    """Test cases for tempo detection functions."""

    def setUp(self):
        """Set up test data."""
        # Create a simple synthetic signal with 120 BPM (2 Hz) tempo
        sr = 22050
        duration = 10  # seconds
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create clicks at 120 BPM (0.5s apart)
        beat_period = 0.5  # 120 BPM = 2 Hz = 0.5s period
        num_beats = int(duration / beat_period)
        
        # Initialize empty signal
        self.y = np.zeros_like(t)
        
        # Add pulses at beat positions
        for i in range(num_beats):
            beat_pos = int(i * beat_period * sr)
            if beat_pos + 100 < len(self.y):
                self.y[beat_pos:beat_pos+100] = np.linspace(1.0, 0, 100)
        
        # Add some noise
        self.y += 0.05 * np.random.randn(len(self.y))
        
        # Store parameters
        self.sr = sr
        self.expected_tempo = 120.0
    
    def test_onset_strength(self):
        """Test onset strength extraction."""
        onset_env = _get_onset_strength(self.y, self.sr, 512, 'spectral_flux')
        
        # Check that onset_env is not empty and has reasonable values
        self.assertGreater(len(onset_env), 0)
        self.assertGreater(np.max(onset_env), 0)
    
    def test_tempo_detection_basic(self):
        """Test basic tempo detection functionality."""
        tempo = detect_tempo(y=self.y, sr=self.sr, visualize=False)
        
        # Check that tempo is close to expected (120 BPM)
        self.assertIsNotNone(tempo)
        # Allow a 10% margin of error for tempo estimation
        self.assertGreaterEqual(tempo, self.expected_tempo * 0.9)
        self.assertLessEqual(tempo, self.expected_tempo * 1.1)
    
    def test_multiple_tempi(self):
        """Test multiple tempo candidate detection."""
        result = detect_tempo(y=self.y, sr=self.sr, n_tempi=3, visualize=False)
        
        # Check that the result is a dictionary with the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('tempo', result)
        self.assertIn('confidence', result)
        self.assertIn('onset_strength', result)
        self.assertIn('onset_times', result)
        
        # Check that we have 3 tempo candidates
        self.assertEqual(len(result['tempo']), 3)
        self.assertEqual(len(result['confidence']), 3)
        
        # Check that confidences sum to 1.0
        self.assertAlmostEqual(sum(result['confidence']), 1.0, places=6)
    
    def test_tempo_distribution(self):
        """Test tempo distribution analysis."""
        # Create a longer signal with a tempo change
        sr = 22050
        duration = 20  # seconds
        t = np.linspace(0, duration, int(sr * duration))
        
        # First half at 120 BPM
        y1 = np.zeros(int(len(t)/2))
        beat_period1 = 0.5  # 120 BPM
        num_beats1 = int((duration/2) / beat_period1)
        
        for i in range(num_beats1):
            beat_pos = int(i * beat_period1 * sr)
            if beat_pos + 100 < len(y1):
                y1[beat_pos:beat_pos+100] = np.linspace(1.0, 0, 100)
        
        # Second half at 160 BPM
        y2 = np.zeros(int(len(t)/2))
        beat_period2 = 0.375  # 160 BPM
        num_beats2 = int((duration/2) / beat_period2)
        
        for i in range(num_beats2):
            beat_pos = int(i * beat_period2 * sr)
            if beat_pos + 100 < len(y2):
                y2[beat_pos:beat_pos+100] = np.linspace(1.0, 0, 100)
        
        # Combine both parts and add noise
        y = np.concatenate([y1, y2])
        y += 0.05 * np.random.randn(len(y))
        
        # Analyze tempo distribution
        result = analyze_tempo_distribution(
            y=y, sr=sr, frame_length=5, visualize=False
        )
        
        # Check that the result contains the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('tempo_segments', result)
        self.assertIn('segment_times', result)
        self.assertIn('avg_tempo', result)
        self.assertIn('min_tempo', result)
        self.assertIn('max_tempo', result)
        self.assertIn('std_tempo', result)
        
        # Check that we detected the tempo change (std_tempo should be significant)
        self.assertGreater(result['std_tempo'], 5.0)
        
        # Check that min and max tempos are in the expected range
        # Allow a 15% margin for error in this more complex test
        self.assertLessEqual(abs(result['min_tempo'] - 120.0) / 120.0, 0.15)
        self.assertLessEqual(abs(result['max_tempo'] - 160.0) / 160.0, 0.15)

if __name__ == '__main__':
    unittest.main()
