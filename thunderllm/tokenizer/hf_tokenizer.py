from transformers import LlamaTokenizerFast


def get_hf_tokenizer(tokenizer_type: str):
    if tokenizer_type == 'hf-llama-tokenizer':
        return LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer")
