import spacy
from CocCocTokenizer import PyTokenizer


spacy_en = spacy.load('en_core_web_sm')
spacy_vi = PyTokenizer(load_nontone_data=True)


def tokenize_vi(text):
    return [tok for tok in spacy_vi.word_tokenize(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
