"""
This module contains tools, functions to preprocess data (video), prepare for
feeding model, construct predictions.

Pre-processing:
    Input:
        Video file: Contains arbitrary stream visual or audio, both, no any
    Criterion:
        - No audio -> Can not detect.
        - audio, not visual -> Use audio full stream only
        - audio, visual but not speaker ->
        - audio, visual, speaker but not cropped mouth ->
        - audio, visual, speaker, mouth -> Use both visual and audio, visual hav
    Output:
        Stream segments:
            - visual:
                Duration: 3 seconds
                Frame rate: 25
            - audio:
                Duration: 3 seconds
                Sample rate: 16000
"""
import os
from .tasker import Tasker
from .checker import Checker
from .normalizer import Normalizer
from .as_detector import DemoASDetector
from .cropper import DemoCropper
from .combiner import Combiner
