from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import os
import torch

# def create_model(model_id="meta-llama/Llama-3.2-3B-Instruct"):
#     """Creates and returns a text-generation model using Meta's Llama 3.2-3B."""
#     model = pipeline(
#         "text-generation",
#         model=model_id,
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )
#     return model

def download_quantize_and_save_model(model_name: str, save_directory: str):
    """
    Downloads a pre-trained model, applies 8-bit quantization, and saves it to the specified directory.

    Args:
        model_name (str): The name of the pre-trained model (e.g., "gpt2", "meta-llama/Llama-2-7b").
        save_directory (str): The directory where the quantized model and tokenizer will be saved.
    """
    # Step 1: Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload if needed
    )

    # Step 2: Load the tokenizer
    print(f"Downloading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 3: Load the quantized model
    print(f"Downloading and quantizing model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"  # Automatically map layers to GPU/CPU
    )

    # Step 4: Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Step 5: Save the quantized model and tokenizer
    print(f"Saving quantized model and tokenizer to: {save_directory}")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print("Model and tokenizer have been successfully saved.")

def load_quantized_model_and_tokenizer(save_directory: str):
    """
    Loads a quantized model and tokenizer from the specified directory.

    Args:
        save_directory (str): The directory where the quantized model and tokenizer are saved.

    Returns:
        model: The loaded quantized model.
        tokenizer: The loaded tokenizer.
        text_generator: A text generation pipeline using the loaded model and tokenizer.
    """
    # Step 1: Reload the quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload if needed
    )

    # Step 2: Reload the tokenizer
    print(f"Loading tokenizer from: {save_directory}")
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    # Step 3: Reload the quantized model
    print(f"Loading quantized model from: {save_directory}")
    model = AutoModelForCausalLM.from_pretrained(
        save_directory,
        quantization_config=quantization_config,  # Use the same quantization config
        device_map="auto"  # Automatically map layers to GPU/CPU
    )

    # Step 4: Create a text generation pipeline
    print("Creating text generation pipeline...")
    text_generator = pipeline(
        "text-generation",  # Task type
        model=model,
        tokenizer=tokenizer,
    )

    print("Model, tokenizer, and text generation pipeline have been successfully loaded.")
    return text_generator