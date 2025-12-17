import re
from collections import Counter


# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
# import nltk
#
#
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')

import spacy
from nltk.corpus import stopwords
import re
import string
# Load the small English spaCy model
nlp = spacy.load('en_core_web_sm')
def term_frequency(doc: str):
    doc_lower= doc.lower()
    print(doc_lower)
    translator = str.maketrans('', '', string.punctuation)
    remove_punct = doc_lower.translate(translator)
    print(remove_punct)
    tokens = re.findall(r'\b\w+\b', remove_punct)

    # tokenize_white_space = remove_punct.replace(" ", "")
    sorted_token = dict(sorted(Counter(tokens).items()))
    sorted_tokens = list(sorted_token.values())
    print(sorted_tokens)


def clean_raw_text(text:str):
    # cleaned = re.sub(r"http\S+|\d+|[,!.]", "", text.lower())
    text_lower  = text.lower()
    translator  = str.maketrans('','', string.punctuation)
    text_remove_url = re.sub(r"http\S+", "", text_lower)
    text_remove_punctuation = text_remove_url.translate(translator)
    text_remove_digit = re.sub(r"\d+", "", text_remove_punctuation)

def part_of_speech():
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    # for token in doc:
    #     print(token.text, token.pos_, token.tag_)
    print(doc[0].text, doc[0].pos_, doc[0].tag_)

def entity_level():
    doc = nlp("Softbank invested $2.3 billion in India in 2023.")
    for token in doc:
        print(token.text, token.ent_type_)


if __name__=='__main__':
    # term_frequency("Hello, hello world! NLP world.")
    # clean_raw_text('''Machine learning Rocks! Visit https://example.com now.
# Data points: 1,234 are collected.''')
#     part_of_speech()
    entity_level()


