#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path
import torch
import numpy as np
from transformers import pipeline
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Configuration settings for LLM analysis
LLM_CONFIG = {
    # Local LLM settings
    "local_temperature": 0.7,
    "local_max_tokens": 2000,
    
    # Analysis prompts
    "analysis_prompt": """
    You are a helpful assistant analyzing a meeting transcript. Please provide:
    
    1. A brief summary of the meeting (2-3 sentences)
    2. Key topics discussed
    3. Action items with owners (if specified)
    4. Decisions made
    5. Next steps
    
    Format your response in markdown.
    """
}

def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe and analyze meeting audio recordings")
    parser.add_argument("audio_file", type=str, help="Path to the audio recording file")
    
    # Transcription options
    parser.add_argument("--whisper-model", type=str, 
                        default=os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo"),
                        help="Whisper model to use for local transcription")
    
    # Analysis options
    parser.add_argument("--llm-model", type=str, 
                        default=os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
                        help="HuggingFace model to use for local analysis")
    
    # Device options
    parser.add_argument("--device", type=str, 
                        default="auto",
                        help="Device to use for inference (cpu, xdu, cuda, cuda:0, etc.)")
    
    # Output options
    parser.add_argument("--output-file", type=str, 
                        default=os.getenv("OUTPUT_FILE", "meeting_analysis.md"),
                        help="Path to save the output file")
    
    # Additional options
    parser.add_argument("--supplementary-instructions", type=str, 
                        default=os.getenv("SUPPLEMENTARY_INSTRUCTIONS", ""),
                        help="Additional instructions for the analysis LLM")
    parser.add_argument("--verbose", action="store_true",
                        default=os.getenv("DEBUG", "false").lower() == "true",
                        help="Enable verbose output")
    
    return parser.parse_args()

def get_device_config(device_str):
    """
    Returns a configuration dictionary for the device to be used by the transformers pipeline,
    automatically setting the torch_dtype based on the device type.

    Parameters:
        device_str (str): The device argument, e.g., "auto", "cpu", "cuda", "cuda:0", "xpu", "xpu:0", etc.
    """
    device_str = device_str.lower().strip()
    
    if device_str == "auto":
        return {"device_map": "auto"}
    
    try:
        dev = torch.device(device_str)
    except Exception as e:
        raise ValueError(f"Invalid device string: {device_str}") from e
    
    # Set dtype based on the device type.
    if dev.type == "xpu":
        if not torch.xpu.is_available():
            raise ValueError("XPU is not available on this system.")
        chosen_dtype = torch.float16  # Use half precision for XPU acceleration.
    elif dev.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system.")
        chosen_dtype = torch.float16  # Use half precision for faster inference.
    else:
        chosen_dtype = torch.float32  # Default for CPU and any other devices.
    
    # Build configuration dictionary for device.
    if device_str.startswith("xpu"):
        try:
            device_index = int(device_str.split(":")[1])
        except (IndexError, ValueError):
            device_index = 0
        return {"device": device_index, "torch_dtype": chosen_dtype}
    elif device_str.startswith("cuda"):
        try:
            device_index = int(device_str.split(":")[1])
        except (IndexError, ValueError):
            device_index = 0
        return {"device": device_index, "torch_dtype": chosen_dtype}

    elif device_str in ("cpu", "cpu:0"):
        return {"device": -1, "torch_dtype": chosen_dtype}
    else:
        # For any other device string, pass it as is.
        return {"device": device_str, "torch_dtype": chosen_dtype}

def preprocess_audio(input_file, verbose=False):
    """Preprocess audio: convert to WAV, normalize, remove silence, and resample to 16kHz"""
    if verbose:
        print(f"Preprocessing audio file: {input_file}")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        normalized_path = temp_path / "normalized.wav"
        
        # Convert to WAV if not already
        input_extension = Path(input_file).suffix.lower()
        if input_extension != '.wav':
            if verbose:
                print("Converting to WAV format...")
            audio = AudioSegment.from_file(input_file)
            audio.export(normalized_path, format="wav")
        else:
            # If already WAV, just copy
            audio = AudioSegment.from_file(input_file)
            audio.export(normalized_path, format="wav")
        
        # Normalize audio
        if verbose:
            print("Normalizing audio...")
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)
        
        # Remove silence
        if verbose:
            print("Removing silence...")
        chunks = split_on_silence(
            normalized_audio,
            min_silence_len=500,  # minimum silence length in ms
            silence_thresh=-40,   # consider anything quieter than -40 dBFS as silence
            keep_silence=300      # keep 300ms of silence at the start and end
        )
        
        # Combine all non-silent chunks
        processed_audio = AudioSegment.empty()
        for chunk in chunks:
            processed_audio += chunk
        
        # WhisperFeatureExtractor was trained using a sampling rate of 16000
        processed_audio = processed_audio.set_frame_rate(16000) 
        
        final_path = temp_path / "processed.wav"
        processed_audio.export(final_path, format="wav")
        
        # Read the processed audio for returning
        audio_data, sample_rate = sf.read(final_path)
        
        if verbose:
            print(f"Audio preprocessing complete. Duration: {len(processed_audio)/1000:.2f} seconds, Sample Rate: {sample_rate} Hz")
        
        return audio_data, sample_rate

def transcribe_local(audio_data, sample_rate, model_name, verbose=False):
    """Transcribe audio using a local Whisper model via the Transformers pipeline."""
    from transformers import pipeline, WhisperProcessor

    if verbose:
        print(f"Transcribing using local model: {model_name}")

    # Save the preprocessed audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        # Use soundfile to write out the audio data
        import soundfile as sf
        sf.write(temp_path, audio_data, sample_rate)

    try:
        # Load the processor to get the forced_decoder_ids for English.
        processor = WhisperProcessor.from_pretrained(model_name)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

        # Initialize the pipeline with chunking.
        # chunk_length_s=30 splits the audio into 30-second segments.
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device_map="auto",
            generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
        )

        # Perform transcription on the temporary file.
        result = pipe(temp_path)

        # Clean up temporary file
        os.unlink(temp_path)

        # If the result is a list of segments, join them together.
        if isinstance(result, list):
            transcription = " ".join(segment.get("text", "") for segment in result)
        else:
            transcription = result.get("text", "")

        if verbose:
            print("Local transcription complete")
        return transcription

    except Exception as e:
        print(f"Error during local transcription: {e}")
        # Clean up temporary file in case of error.
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

def analyze_with_local_llm(transcript, model_name, supplementary_instructions="", verbose=False):
    """Analyze transcript using a local LLM with instruct model"""

    # Compose the prompt
    prompt = LLM_CONFIG["analysis_prompt"]
    if supplementary_instructions:
        prompt += "\n\nAdditional instructions: " + supplementary_instructions
    prompt += "\n\nTranscript:\n" + transcript + "\n\nAnalysis:"

    if verbose:
        print("Prompt for instruct model:")
        print(prompt)
    
    generator = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        do_sample=True,
        max_new_tokens=LLM_CONFIG["local_max_tokens"],
        temperature=LLM_CONFIG["local_temperature"],
        model_kwargs={"low_cpu_mem_usage": True}
    )
    
    # Generate the analysis.
    result = generator(prompt)
    analysis_text = result[0]["generated_text"]
    
    # Remove the prompt from the generated text if it is present.
    if analysis_text.startswith(prompt):
        analysis_text = analysis_text[len(prompt):].strip()
    
    if verbose:
        print("Local analysis complete")
    
    return analysis_text

def save_output(transcript, analysis, args):
    """Save the transcript and analysis to a file"""
    output_path = Path(args.output_file)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Meeting Analysis\n\n")
        
        # Write metadata
        f.write("## Metadata\n\n")
        f.write(f"- **Audio File**: {args.audio_file}\n")
        f.write(f"- **Whisper Model**: {args.whisper_model}\n")
        f.write(f"- **LLM Model**: {args.llm_model}\n")
        
        # Write analysis
        f.write("## Analysis\n\n")
        f.write(analysis.strip())
        f.write("\n\n")
        
        # Write transcript
        f.write("## Transcript\n\n")
        f.write(transcript)
    
    return output_path

# Global verbose flag to simplify logging
verbose_global = False

def main():
    global verbose_global
    args = parse_args()
    verbose_global = args.verbose
    audio_path = args.audio_file
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    try:
        # Step 1: Preprocess audio
        audio_data, sample_rate = preprocess_audio(audio_path, verbose=args.verbose)
        
        # Step 2: Transcribe audio
        transcript = transcribe_local(audio_data, sample_rate, args.whisper_model, verbose=args.verbose)
        
        if args.verbose:
            print(f"Transcript length: {len(transcript)} characters")
        
        # Step 3: Analyze transcript

        analysis = analyze_with_local_llm(
            transcript, 
            args.llm_model,
            supplementary_instructions=args.supplementary_instructions,
            verbose=args.verbose
        )
        
        # Step 4: Save output
        output_path = save_output(transcript, analysis, args)
        
        print(f"Analysis complete! Results saved to: {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()