import argparse
import ast
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

def read_token_file(file_path):
    """Reads a text file and returns the parsed list.."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            # convert string to list
            return ast.literal_eval(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def get_huggingface_token():
    """Retrieve the Hugging Face token from environment variables."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN is not set in environment variables.")
    return hf_token

def set_mirror_url():
    """Set the Hugging Face mirror URL."""
    mirror_url = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    os.environ["HF_ENDPOINT"] = mirror_url
    print(f"Using mirror URL: {mirror_url}")

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer with specified configurations."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        do_sample=False,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    print("-"*10)
    embedding_size = model.get_input_embeddings().weight.shape
    print(f"Embedding layer size before resize: {embedding_size}")
    lm_head_size = model.lm_head.weight.shape
    print(f"LM head size before resize: {lm_head_size}")
    print("-"*10)
    return model, tokenizer

def add_new_tokens(tokenizer, model, new_tokens):
    """Add new tokens to the tokenizer and resize the model embeddings."""
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"1. Added {num_added_tokens} new tokens to tokenizer.")

    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token embeddings
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        for i in range(len(new_tokens)):
            embedding_layer.weight[-(i+1)] = torch.mean(embedding_layer.weight[:-num_added_tokens], dim=0)

    return tokenizer, model

def save_model_and_tokenizer(model, tokenizer, tokenizer_path="./update_tokenizer", model_path="./updated_model"):
    """Save the updated tokenizer and model."""
    tokenizer.save_pretrained(tokenizer_path)
    model.save_pretrained(model_path)
    print(f"2. Saved updated tokenizer and model to {tokenizer_path} and {model_path}.")

def reload_model_and_tokenizer(tokenizer_path, model_path, device_map={"": 0}, lora=False, low_attention=False):
    """Reload the model and tokenizer from saved files."""
    if low_attention:
        compute_dtype = torch.float16
        compute_dtype = torch.bfloat16
        attn_implementation = 'sdpa'
    elif torch.cuda.is_bf16_supported():
        os.system('pip install flash_attn')
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'
    print(compute_dtype, attn_implementation)
    if lora:
        print("Using Quantization.")
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # device_map="auto",
            device_map=device_map,
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            use_cache=False,
            low_cpu_mem_usage=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("Without Using Quantization.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=compute_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("3. Reloaded model and tokenizer.")
    return model, tokenizer

def generate_text(model, tokenizer, text, max_length=50):
    """Generate text using the trained model."""
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            use_cache=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Merge lists from multiple text files.")
    parser.add_argument('--path', nargs='+', help="Absolute paths to text files containing new tokens", required=False)

    args = parser.parse_args()
    new_tokens = []
    if args.path:
        for file_path in args.path:
            new_tokens.extend(read_token_file(file_path))

    print("Merged token list: ", len(new_tokens),new_tokens)
    # new_tokens = ["k8s-cluster", "kubernetes", "devops", "microservices", "containerization", "serverless", 
    #              "service mesh", "observability", "gitops", "scalability"]

    # set_mirror_url()
    hf_token = get_huggingface_token()

    model_name = "./models/llama-3.2-3b-instruct"
    model, tokenizer = load_model_and_tokenizer(model_name)

    tokenizer_path = "./models/llama-3.2-3b-instruct-graph/update_tokenizer"
    model_path = "./models/llama-3.2-3b-instruct-graph/updated_model"

    if new_tokens:
        tokenizer, model = add_new_tokens(tokenizer, model, new_tokens)
        save_model_and_tokenizer(model, tokenizer, tokenizer_path=tokenizer_path, model_path=model_path)
        # model, tokenizer = reload_model_and_tokenizer("./update_tokenizer", "./updated_model")
        print("4. Model and tokenizer reloaded.")

    '''text = "今天天气很好，我想去"
    output_text = generate_text(model, tokenizer, text)
    print(output_text)'''

if __name__ == "__main__":
    main()