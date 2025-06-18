from . import (document_deduplicator, document_minhash_deduplicator,
               document_simhash_deduplicator,)
from .document_deduplicator import DocumentDeduplicator
from .document_minhash_deduplicator import DocumentMinhashDeduplicator
from .document_simhash_deduplicator import DocumentSimhashDeduplicator

__all__ = [
    'DocumentMinhashDeduplicator',
    'DocumentDeduplicator',
    'DocumentSimhashDeduplicator'
]
