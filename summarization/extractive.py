from nltk.tokenize import sent_tokenize, word_tokenize
from heapq import nlargest
import nltk

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def extractive_summary(text: str, top_n: int = 3) -> str:
    """Generates an extractive summary using word frequency scoring."""
    sentences = sent_tokenize(text)
    if len(sentences) <= top_n:
        return text

    word_freq = {}
    for word in word_tokenize(text.lower()):
        if word.isalnum():
            word_freq[word] = word_freq.get(word, 0) + 1

    max_freq = max(word_freq.values(), default=1)
    for word in word_freq:
        word_freq[word] /= max_freq

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    summary_sentences = nlargest(top_n, sentence_scores, key=sentence_scores.get)
    return " ".join(summary_sentences)
