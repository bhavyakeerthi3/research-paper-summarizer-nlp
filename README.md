# 📄 Research Paper Summarizer (NLP)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/NLP-Transformers-green.svg)](https://huggingface.co/docs/transformers/index)

An advanced NLP-powered tool designed to bridge the gap between complex academic content and reader comprehension. This project leverages state-of-the-art Transformer models (BART and T5) to provide high-quality extractive and abstractive summaries of research papers.

---

## 🌟 Key Features

### 🔍 Intelligent Extraction
Automatically identifies and extracts the **Abstract** section from PDF documents using advanced regular expressions, saving you time from manual searching.

### 🧹 Smart Preprocessing
- **Jargon Simplification**: Automatically converts complex academic terminology into clear, accessible language.
- **Redundancy Removal**: Uses TF-IDF cosine similarity to eliminate repetitive information, ensuring a concise output.

### 📝 Dual-Mode Summarization
- **Extractive**: Highlights the most significant sentences directly from the source text using word-frequency scoring.
- **Abstractive**: Generates human-like, coherent summaries that simplify dense concepts using **BART-large-cnn** and **T5-small**.

### 💻 Interactive Web Interface
A modern, glassmorphism-inspired UI built with **Gradio** that supports both direct text input and PDF file uploads for seamless processing.

---

## 📂 Project Architecture

The project follows a modular and extensible structure:

```bash
├── app.py              # Main Entry: Gradio Web Interface
├── main.py             # CLI Entry: Pipeline Demonstration
├── preprocessing/      # Linguistic cleaning & PDF extraction
├── summarization/      # BART/T5 abstractive & TF-IDF extractive logic
├── evaluation/         # ROUGE metric calculations
├── notebooks/          # Research & technical experimentation
├── requirements.txt    # Project dependencies
└── LICENSE             # MIT License
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Usage

#### **Launch the Web Dashboard**
```bash
python app.py
```
*Access the interface at `http://127.0.0.1:7860` in your browser.*

#### **Run Command-Line Demo**
```bash
python main.py
```
*Processes a sample paragraph and displays ROUGE evaluation scores in the terminal.*

---

## 📊 Evaluation & Metrics
The system's performance is validated using **ROUGE-1** and **ROUGE-L** metrics, which measure the overlap between generated summaries and reference texts, ensuring high standards of accuracy and coherence.

---

## 🛠️ Built With
- **Transformers**: State-of-the-art NLP models.
- **NLTK**: Natural Language Toolkit for linguistic analysis.
- **Gradio**: High-performance UI framework for ML applications.
- **PyMuPDF**: High-speed PDF processing.
- **Scikit-learn**: For advanced vectorization and similarity calculations.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

**Bhavya Keerthi** 
For questions or suggestions, please open an issue on GitHub.



---

## 🙏 Acknowledgements
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK](https://www.nltk.org/)
- [Gradio](https://gradio.app/)
- [All the contributors and readers who find this useful!](https://github.com/bhavyakeerthi3/research-paper-summarizer-nlp)
