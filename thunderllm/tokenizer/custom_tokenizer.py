from transformers import AutoTokenizer


def get_custom_tokenizer(tokenizer_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_path)
