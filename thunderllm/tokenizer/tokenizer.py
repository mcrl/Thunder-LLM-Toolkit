def get_tokenizer(tokenizer_type: str, tokenizer_path: str):
    if tokenizer_type.startswith('hf-'):
        from .hf_tokenizer import get_hf_tokenizer
        return get_hf_tokenizer(tokenizer_type)
    elif tokenizer_type == 'custom':
        from .custom_tokenizer import get_custom_tokenizer
        return get_custom_tokenizer(tokenizer_path)
    raise ValueError(f'Unknown tokenizer type: {tokenizer_type}')
