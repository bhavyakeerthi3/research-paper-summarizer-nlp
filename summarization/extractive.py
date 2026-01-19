from sklearn.feature_extraction.text import TfidfVectorizer

def extractive_summary(text: str, top_n: int = 3) -> str:
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if len(s) > 20]

    if not sentences:
        return ""

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(sentences)

    scores = matrix.sum(axis=1).A1
    ranked = [sentences[i] for i in scores.argsort()[-top_n:]]

    return ". ".join(ranked)
