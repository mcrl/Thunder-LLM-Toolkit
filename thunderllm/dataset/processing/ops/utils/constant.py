from enum import Enum

DEFAULT_PREFIX = '__snu__'


class Fields(object):
    stats = DEFAULT_PREFIX + 'stats__'
    context = DEFAULT_PREFIX + 'context__'


class HashKeys(object):
    hash = DEFAULT_PREFIX + 'hash'
    minhash = DEFAULT_PREFIX + 'minhash'
    simhash = DEFAULT_PREFIX + 'simhash'

    # duplicate flag
    is_duplicate = DEFAULT_PREFIX + 'is_duplicate'


class InterVars(object):
    # text
    lines = DEFAULT_PREFIX + 'lines'
    words = DEFAULT_PREFIX + 'words'
    refined_words = DEFAULT_PREFIX + 'refined_words'
