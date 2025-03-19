# Whisper Meeing Sumamry

This is a personal project developed to explore Intel ARC GPU support in PyTorch 2.6 while experimenting with meeting transcription and analysis using [Whisper](https://github.com/openai/whisper) and Large Language Models (LLMs) powered by [Hugging Face Transformers](https://huggingface.co/docs/transformers/index). Although it was optimized for Intel ARC, the tool is designed to work on other architectures as well—but your mileage may vary.

## Installation

Install the required dependencies:

### For Intel Arc GPU with PyTorch 2.6+ built-in XPU support (Windows or Linux)

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/test/xpu
python -m pip install -r requirements.txt
```

## Command-Line Options

### Positional Argument:
- **`audio_file`**: Path to the audio recording file.

### Optional Arguments:
- **`--whisper-model`**: Whisper model for local transcription.  
  *Default:* `WHISPER_MODEL` or `"openai/whisper-large-v3-turbo"`.
- **`--llm-model`**: HuggingFace model for local analysis.  
  *Default:* `LLM_MODEL` or `"mistralai/Mistral-7B-Instruct-v0.3"`.
- **`--device`**: Inference device (`cpu`, `cuda`, `mps`, `xpu`).  
  *Default:* `"auto"`.
- **`--output-file`**: Output file path for analysis.  
  *Default:* `OUTPUT_FILE` or `"meeting_analysis.md"`.
- **`--supplementary-instructions`**: Additional instructions for LLM analysis.
- **`--verbose`**: Enable verbose logging.

## Environment Variables

- **`WHISPER_MODEL`**: Whisper model for API transcription.
- **`LLM_MODEL`**: LLM model for API analysis.
- **`HUGGINGFACE_TOKEN`**: API key for Hugging Face models.
- **`OPENAI_API_KEY`**: API key for OpenAI models.

## License

This project is licensed under the **MIT License**.

