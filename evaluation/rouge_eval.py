from rouge_score import rouge_scorer

def compute_rouge(reference: str, predicted: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=True
    )
    return scorer.score(reference, predicted)
