import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transcribe 
import tempfile
import numpy as np
import soundfile as sf
import pytest
import torch
import argparse
from pathlib import Path



def test_get_device_config_auto():
    config = transcribe.get_device_config("auto")
    assert config == {"device_map": "auto"}

def test_get_device_config_xpu():
    config = transcribe.get_device_config("xpu")
    assert config["device"] == 0
    assert config["torch_dtype"] == torch.float16

def test_get_device_config_cpu():
    config = transcribe.get_device_config("cpu")
    assert config["device"] == -1
    assert config["torch_dtype"] == torch.float32

def test_get_device_config_invalid():
    with pytest.raises(ValueError):
        transcribe.get_device_config("invalid_device")

def test_preprocess_audio(tmp_path):
    # Create a dummy WAV file with a sine wave.
    sample_rate = 44100
    duration = 1  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq = 440  # A tone at 440 Hz
    audio_data = 0.5 * np.sin(2 * np.pi * freq * t)
    file_path = tmp_path / "test.wav"
    sf.write(file_path, audio_data, sample_rate)
    
    # Call preprocess_audio (which converts the file to WAV, normalizes, and resamples)
    processed_audio, processed_sr = transcribe.preprocess_audio(str(file_path), verbose=False)
    
    # Check that the sample rate is now 16kHz and that we got a numpy array.
    assert processed_sr == 16000
    assert isinstance(processed_audio, np.ndarray)

def test_transcribe_local(monkeypatch):
    # Create dummy audio data: 1 second of silence at 16kHz.
    audio_data = np.zeros(16000)
    sample_rate = 16000

    # Define fake functions for WhisperProcessor and pipeline.
    # These will override the actual model calls to return dummy outputs.
    import transformers

    def fake_from_pretrained(model_name):
        class FakeProcessor:
            def get_decoder_prompt_ids(self, language, task):
                return [0, 1, 2]
        return FakeProcessor()

    def fake_pipeline(task, **kwargs):
        if task == "automatic-speech-recognition":
            def fake_pipe(audio_path):
                return [{"text": "dummy transcription"}]
            return fake_pipe
        else:
            # For other tasks, return a dummy callable.
            return lambda *args, **kwargs: []
    
    monkeypatch.setattr(transformers.WhisperProcessor, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)

    transcription = transcribe.transcribe_local(audio_data, sample_rate, "fake_model", verbose=False)
    assert transcription == "dummy transcription"

def test_analyze_with_local_llm(monkeypatch):
    # Define a fake pipeline for text-generation.
    def fake_pipeline(task, **kwargs):
        if task == "text-generation":
            def fake_generator(prompt):
                return [{"generated_text": "dummy analysis"}]
            return fake_generator
        else:
            return lambda *args, **kwargs: []
    
    monkeypatch.setattr(transcribe, "pipeline", fake_pipeline)
    
    dummy_transcript = "This is a dummy transcript."
    analysis = transcribe.analyze_with_local_llm(
        dummy_transcript,
        "fake_model",
        supplementary_instructions="extra instructions",
        verbose=False
    )
    assert analysis == "dummy analysis"

def test_save_output(tmp_path):
    # Create dummy args to simulate command-line arguments.
    dummy_args = argparse.Namespace(
        output_file=str(tmp_path / "output.md"),
        audio_file="dummy_audio.wav",
        transcription_method="local",
        analysis_method="local",
        whisper_model="dummy_whisper_model",
        llm_model="dummy_llm_model",
        api_llm_model="dummy_api_llm_model"  # not used for local methods
    )
    dummy_transcript = "Dummy transcript"
    dummy_analysis = "Dummy analysis"
    
    output_path = transcribe.save_output(dummy_transcript, dummy_analysis, dummy_args)
    # Verify the output file exists.
    assert output_path.exists()
    
    content = output_path.read_text(encoding="utf-8")
    # Check that key sections and metadata are present.
    assert "# Meeting Analysis" in content
    assert "Dummy transcript" in content
    assert "Dummy analysis" in content
    assert "dummy_audio.wav" in content
    assert "dummy_whisper_model" in content
    assert "dummy_llm_model" in content
