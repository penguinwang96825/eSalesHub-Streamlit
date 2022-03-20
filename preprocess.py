import re
import string
import pysbd
import itertools
import warnings
import unicodedata


with open('stopwords-en.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = f.read().splitlines()


ALL_LETTERS = string.ascii_letters + " .,;'-"
PATTERN = r"""
    (?x)                    # set flag to allow verbose regexps
    (?:[A-Z]\.)+            # abbreviations, e.g. U.S.A.
    | \$?\d+(?:\.\d+)?%?    # currency and percentages, $12.40, 50%
    | \w+(?:-\w+)+          # words with internal hyphens
    | \w+(?:'[a-z])         # words with apostrophes
    | \.\.\.                # ellipsis
    |(?:Mr|mr|Mrs|mrs|Dr|dr|Ms|ms)\.     # honorifics
    | \w+                   # normal words
    | [,.!?\\-]             # specific punctuation
"""


def word_tokenize(text):

    def tokenise_(t):
        warnings.filterwarnings('ignore')
        t = any2unicode(t)
        # Sentence boundary disambiguation
        seg = pysbd.Segmenter(language="en", clean=False)
        tokens = [re.findall(PATTERN, sentence) for sentence in seg.segment(t)]
        return list(itertools.chain(*tokens))

    if isinstance(text, str):
        return tokenise_(text)
    elif isinstance(text, list):
        return [tokenise_(t) for t in text]


def decontracted(text):
    """Expanding English language contractions"""
    # specific
    text = re.sub(r"won(\'|\’)t", "will not", text)
    text = re.sub(r"can(\'|\’)t", "can not", text)

    # general
    text = re.sub(r"n(\'|\’)t", " not", text)
    text = re.sub(r"(\'|\’)re", " are", text)
    text = re.sub(r"(\'|\’)s", " is", text)
    text = re.sub(r"(\'|\’)d", " would", text)
    text = re.sub(r"(\'|\’)ll", " will", text)
    text = re.sub(r"(\'|\’)l", " will", text)
    text = re.sub(r"(\'|\’)t", " not", text)
    text = re.sub(r"(\'|\’)ve", " have", text)
    text = re.sub(r"(\'|\’)v", " have", text)
    text = re.sub(r"(\'|\’)m", " am", text)
    return text


def remove_stopwords(text):
    pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')
    text = pattern.sub('', text)
    return text


def remove_digits(text):
    text = re.sub('[0-9]', '', text)
    return text


def clean_dialogue(text):
    text = decontracted(text)
    text = remove_digits(text)
    text = remove_stopwords(text)
    return text


def any2unicode(text, encoding='utf8', errors='strict'):
    """
    Convert a string (bytestring in `encoding` or unicode), to unicode.
    
    References
    ----------
    1. https://tedboy.github.io/nlps/_modules/gensim/utils.html#any2unicode
    """
    unicode = str
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def unicode2ascii(text):
    """
    Turn a Unicode string to plain ASCII

    References
    ----------
    1. https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )
