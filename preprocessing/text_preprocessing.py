import re
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def extract_abstract_from_pdf(pdf_path):
    """Extracts the abstract section from a PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # Flexible regex for Abstract section
        abstract_match = re.search(r'(?i)abstract[\s:\n]*(.*?)(?=\n\s*[1I]\.?\s*Introduction|\n\s*[A-Z][a-z]{2,})', text, re.DOTALL)

        if abstract_match:
            return abstract_match.group(1).strip()
        return "Abstract not found."
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def cosine_similarity_tfidf(sent1, sent2):
    """Computes cosine similarity between two sentences using TF-IDF."""
    vectorizer = TfidfVectorizer().fit_transform([sent1, sent2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

def clean_redundancy(text, threshold=0.8):
    """Removes redundant similar sentences from the text."""
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    seen = []

    for sentence in sentences:
        # Preprocess for comparison
        comp_sentence = re.sub(r'\s+', ' ', sentence.strip()).lower()
        comp_sentence = re.sub(r'[^a-zA-Z0-9 ]', '', comp_sentence)
        
        if not any(cosine_similarity_tfidf(comp_sentence, s) > threshold for s in seen):
            seen.append(comp_sentence)
            cleaned_sentences.append(sentence)

    return " ".join(cleaned_sentences)

def simplify_jargon(text):
    """Replaces common academic jargon with simpler terms."""
    jargon_dict = {
        "synergy": "cooperation",
        "utilize": "use",
        "leverage": "take advantage of",
        "paradigm": "model",
        "robust": "strong",
        "iterate": "repeat",
        "streamline": "simplify",
        "facilitate": "help",
        "implement": "carry out",
        "core competency": "main strength"
    }

    for jargon, simple in jargon_dict.items():
        text = re.sub(rf"\b{jargon}\b", simple, text, flags=re.IGNORECASE)

    return text

def preprocess_text(text):
    """Applies all preprocessing steps to the text."""
    text = simplify_jargon(text)
    text = clean_redundancy(text)
    return text
