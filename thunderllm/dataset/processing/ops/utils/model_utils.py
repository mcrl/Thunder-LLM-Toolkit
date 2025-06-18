import fnmatch
import os
from functools import partial
from pickle import UnpicklingError
from typing import Optional, Union

import multiprocess as mp
import wget
from loguru import logger

from torch.cuda import device_count, is_available

MODEL_ZOO = {}


def prepare_fasttext_model(model_name='lid.176.bin'):
    """
    Prepare and load a fasttext model.

    :param model_name: input model name
    :return: model instance.
    """
    import fasttext

    logger.info('Loading fasttext language identification model...')
    try:
        ft_model = fasttext.load_model(model_name)
    except:  # noqa: E722
        ft_model = fasttext.load_model(model_name, force=True)
    return ft_model


def prepare_mecab_model(dic_path='', rc_path=''):
    from konlpy.tag import Mecab

    logger.info('Loading sentencepiece model...')
    HOME = os.environ['HOME']
    if not dic_path:
        dic_path = f"{HOME}/mecab/lib/mecab/dic/mecab-ko-dic"
    if not rc_path:
        rc_path = f"{HOME}/mecab/etc/mecabrc"

    mecab_model = Mecab(f"{dic_path} -r {rc_path}")
    return mecab_model


def prepare_sentencepiece_model(model_name):
    """
    Prepare and load a sentencepiece model.

    :param model_path: input model path
    :return: model instance
    """
    import sentencepiece

    logger.info('Loading sentencepiece model...')
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    try:
        sentencepiece_model.load(model_path)
    except:  # noqa: E722
        sentencepiece_model.load(model_name, force=True)
    return sentencepiece_model


def prepare_kenlm_model(model_name, name_pattern='{}.arpa.bin'):
    """
    Prepare and load a kenlm model.

    :param model_name: input model name in formatting syntax.
    :param lang: language to render model name
    :return: model instance.
    """
    import kenlm

    model_name = name_pattern.format(model_name)

    logger.info('Loading kenlm language model...')
    try:
        kenlm_model = kenlm.Model(model_name)
    except:  # noqa: E722
        kenlm_model = kenlm.Model(model_name, force=True)
    return kenlm_model


def prepare_nltk_model(lang, name_pattern='punkt.{}.pickle'):
    """
    Prepare and load a nltk punkt model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    from nltk.data import load

    nltk_to_punkt = {
        'en': 'english',
        'fr': 'french',
        'pt': 'portuguese',
        'es': 'spanish'
    }
    assert lang in nltk_to_punkt.keys(
    ), 'lang must be one of the following: {}'.format(
        list(nltk_to_punkt.keys()))
    model_name = name_pattern.format(nltk_to_punkt[lang])

    logger.info('Loading nltk punkt split model...')
    try:
        nltk_model = load(model_name)
    except:  # noqa: E722
        nltk_model = load(model_name, force=True)
    return nltk_model


def prepare_huggingface_model(pretrained_model_name_or_path,
                              return_model=True,
                              trust_remote_code=False):
    """
    Prepare and load a HuggingFace model with the correspoding processor.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :param trust_remote_code: passed to transformers
    :return: a tuple (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    import transformers
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

    if return_model:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        if hasattr(config, 'auto_map'):
            class_name = next(
                (k for k in config.auto_map if k.startswith('AutoModel')),
                'AutoModel')
        else:
            # TODO: What happens if more than one
            class_name = config.architectures[0]

        model_class = getattr(transformers, class_name)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

    return (model, processor) if return_model else processor


MODEL_FUNCTION_MAPPING = {
    'fasttext': prepare_fasttext_model,
    'sentencepiece': prepare_sentencepiece_model,
    'kenlm': prepare_kenlm_model,
    'nltk': prepare_nltk_model,
    'huggingface': prepare_huggingface_model,
    'mecab': prepare_mecab_model,
}


def prepare_model(model_type, **model_kwargs):
    assert (model_type in MODEL_FUNCTION_MAPPING.keys()
            ), 'model_type must be one of the following: {}'.format(
                list(MODEL_FUNCTION_MAPPING.keys()))
    global MODEL_ZOO
    model_func = MODEL_FUNCTION_MAPPING[model_type]
    orig_args = model_func.__code__.co_varnames
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in orig_args}
    model_key = partial(model_func, **model_kwargs)
    # always instantiate once for possible caching
    model_objects = model_key()
    MODEL_ZOO[model_key] = model_objects
    return model_key


def move_to_cuda(model, rank):
    # Assuming model can be either a single module or a tuple of modules
    if not isinstance(model, tuple):
        model = (model, )

    for module in model:
        if callable(getattr(module, 'to', None)):
            logger.debug(
                f'Moving {module.__class__.__name__} to CUDA device {rank}')
            module.to(f'cuda:{rank}')


def get_model(model_key=None, rank=None, use_cuda=False):
    if model_key is None:
        return None

    global MODEL_ZOO
    if model_key not in MODEL_ZOO:
        logger.debug(
            f'{model_key} not found in MODEL_ZOO ({mp.current_process().name})'
        )
        MODEL_ZOO[model_key] = model_key()
    if use_cuda:
        rank = 0 if rank is None else rank
        rank = rank % device_count()
        move_to_cuda(MODEL_ZOO[model_key], rank)
    return MODEL_ZOO[model_key]
