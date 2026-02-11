import gradio as gr
from preprocessing.text_preprocessing import preprocess_text, extract_abstract_from_pdf
from summarization.extractive import extractive_summary
from summarization.bart_abstractive import bart_summary

# Professional Theme Configuration
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_title_text_weight="600",
    block_label_text_weight="600",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

def simplify_text(text):
    """Processes text for Gradio UI."""
    if not text.strip():
        return "Please enter some text.", ""
    
    cleaned = preprocess_text(text)
    ext_summary = extractive_summary(cleaned)
    abs_summary = bart_summary(cleaned)
    
    return ext_summary, abs_summary

def process_pdf(pdf_file):
    """Processes PDF file for Gradio UI."""
    if pdf_file is None:
        return "Please upload a PDF file.", ""
    
    abstract_text = extract_abstract_from_pdf(pdf_file.name)
    if abstract_text == "Abstract not found.":
        return abstract_text, ""
    
    return simplify_text(abstract_text)

# Custom CSS for glassmorphism and spacing
custom_css = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
}
.header-text {
    text-align: center;
    margin-bottom: 2rem;
}
.output-box {
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1) !important;
}
"""

with gr.Blocks(theme=theme, css=custom_css, title="Research Paper Simplifier") as app:
    with gr.Column(elem_classes="header-text"):
        gr.Markdown("# 📝 Research Paper Simplifier")
        gr.Markdown("Transform complex academic jargon into clear, concise summaries using AI.")

    with gr.Tabs():
        with gr.TabItem("📄 Paste Text", id=0):
            with gr.Row():
                input_text = gr.Textbox(
                    label="Input Text", 
                    lines=10, 
                    placeholder="Paste your abstract or paper content here...",
                    info="Supports abstracts, introductions, or full paragraphs."
                )
            with gr.Row():
                with gr.Column():
                    extractive_output = gr.Textbox(
                        label="Extractive Summary", 
                        lines=5, 
                        placeholder="Key sentences will appear here...",
                        elem_classes="output-box"
                    )
                with gr.Column():
                    abstractive_output = gr.Textbox(
                        label="Abstractive Summary", 
                        lines=5, 
                        placeholder="Simplified explanation will appear here...",
                        elem_classes="output-box"
                    )
            run_button = gr.Button("🚀 Simplify Text", variant="primary", scale=1)
            run_button.click(
                fn=simplify_text, 
                inputs=input_text, 
                outputs=[extractive_output, abstractive_output]
            )

        with gr.TabItem("📂 Upload PDF", id=1):
            gr.Markdown("### Automatic Abstract Extraction")
            with gr.Row():
                pdf_input = gr.File(
                    label="Research Paper PDF", 
                    file_types=[".pdf"],
                    file_count="single"
                )
            with gr.Row():
                with gr.Column():
                    pdf_extractive = gr.Textbox(
                        label="Extractive Summary", 
                        lines=5, 
                        placeholder="Extracted key sentences...",
                        elem_classes="output-box"
                    )
                with gr.Column():
                    pdf_abstractive = gr.Textbox(
                        label="Abstractive Summary", 
                        lines=5, 
                        placeholder="Simplified abstract results...",
                        elem_classes="output-box"
                    )
            pdf_button = gr.Button("🔍 Process PDF", variant="primary")
            pdf_button.click(
                fn=process_pdf, 
                inputs=pdf_input, 
                outputs=[pdf_extractive, pdf_abstractive]
            )

    gr.Markdown("---")
    gr.Markdown(
        "Built with **Hugging Face Transformers**, **NLTK**, and **Gradio**. "
        "Models used: `BART-large-cnn` and `T5-small`."
    )

if __name__ == "__main__":
    app.launch()
