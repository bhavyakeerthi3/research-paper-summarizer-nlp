from preprocessing.text_preprocessing import preprocess_text, extract_abstract_from_pdf
from summarization.extractive import extractive_summary
from summarization.t5_abstractive import t5_summary
from summarization.bart_abstractive import bart_summary
from evaluation.rouge_eval import compute_rouge
import os

def run_pipeline(text, reference_summary=None):
    """Runs the complete summarization and evaluation pipeline."""
    print("\n--- Preprocessing Text ---")
    cleaned_text = preprocess_text(text)
    print(f"Original length: {len(text)}, Cleaned length: {len(cleaned_text)}")

    print("\n--- Generating Extractive Summary ---")
    ext_summary = extractive_summary(cleaned_text)
    print(f"Extractive: {ext_summary}")

    print("\n--- Generating T5 Abstractive Summary ---")
    t5_abs_summary = t5_summary(cleaned_text)
    print(f"T5: {t5_abs_summary}")

    print("\n--- Generating BART Abstractive Summary ---")
    bart_abs_summary = bart_summary(cleaned_text)
    print(f"BART: {bart_abs_summary}")

    if reference_summary:
        print("\n--- Evaluating T5 Summary ---")
        t5_rouge = compute_rouge(reference_summary, t5_abs_summary)
        print(f"T5 ROUGE: {t5_rouge}")

        print("\n--- Evaluating BART Summary ---")
        bart_rouge = compute_rouge(reference_summary, bart_abs_summary)
        print(f"BART ROUGE: {bart_rouge}")

if __name__ == "__main__":
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language, in particular how to program computers 
    to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" 
    the contents of documents, including the contextual nuances of the language within them. The technology can then 
    accurately extract information and insights contained in the documents as well as categorize and organize the 
    documents themselves. Synergy and utilizing leveraging paradigms can streamline facilitating robust implementations.
    """
    
    # Example reference summary for evaluation demonstration
    sample_reference = "NLP is an AI field focused on computer-human language interaction and document understanding."
    
    run_pipeline(sample_text, sample_reference)
