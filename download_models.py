from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/CodeLlama-7b-Python-hf"  # or the specific variant you have access to

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # or use `torch_device="cuda"` if manually setting
    cache_dir="/home/jovyan/workspace/viper/pretrained_models"  # optional: if using custom storage
)
