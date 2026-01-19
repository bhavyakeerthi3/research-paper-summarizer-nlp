# Research Paper Summarizer (NLP)

This project focuses on simplifying and summarizing research paper text using
Natural Language Processing (NLP) and transformer-based models.
The aim is to make academic content easier to understand for students and
non-expert readers.

---

## Project Overview
Research papers often contain dense academic language and lengthy explanations.
This project applies text preprocessing, extractive summarization, and
abstractive summarization techniques to generate concise and readable summaries.

The core experimentation and implementation are performed using a Jupyter
Notebook, with supporting logic modularized into separate Python scripts.

---

## Technologies Used
- Python  
- NLTK  
- Scikit-learn  
- PyTorch  
- Hugging Face Transformers  
- ROUGE evaluation metrics  

---

## Features
- Text cleaning and preprocessing
- Extractive summarization using TF-IDF
- Abstractive summarization using transformer models (T5 and BART)
- ROUGE-based evaluation of summaries
- Notebook-based experimentation and analysis

---

## Project Structure
research-paper-summarizer-nlp/
├── notebooks/
│   └── research_paper_summarizer.ipynb
├── preprocessing/
│   └── text_preprocessing.py
├── summarization/
│   ├── extractive.py
│   ├── t5_abstractive.py
│   └── bart_abstractive.py
├── evaluation/
│   └── rouge_eval.py
├── requirements.txt

---

## Implementation Note
The primary experimentation and execution are done in the Jupyter notebook
located in the notebooks directory.
Core logic has been modularized into separate Python scripts for better
organization and future extensibility.

---

## Evaluation
The generated summaries are evaluated using ROUGE-1 and ROUGE-L metrics to
measure content overlap and summary quality.

---

## How to Run
1. Clone the repository  
2. Install dependencies using:  
   pip install -r requirements.txt  
3. Open the notebook:  
   notebooks/research_paper_summarizer.ipynb  
4. Run all cells to generate summaries  

---

## Future Improvements
- Support for full research paper summarization beyond abstracts
- Domain-specific summarization for technical fields
- Improved text simplification for non-technical audiences
- Integration with a lightweight web interface

---

## Author
Bhavya Keerthi  
B.Tech CSE   

This project was developed as part of hands-on learning in Natural Language
Processing, with a focus on combining research-oriented experimentation and
clean software structuring. The work reflects an iterative learning approach
and can be extended further for real-world academic and educational use cases.

