"""
src/app.py
----------
CrediTrust Financial — Intelligent Complaint Analysis Portal
Gradio-based interactive UI for the RAG chatbot.

Features:
- Real-time AI-generated answers (via rag_pipeline.py)
- Source chunk display with complaint metadata
- Product category filter
- Conversation history with clear button
- Premium dark-mode design
"""

import os
import sys
import gradio as gr

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import run_rag, load_vector_store, _get_llm_pipeline

# ─── CSS Styling ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: #0f172a !important;
}

body, .dark {
    background: #0f172a !important;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 60%, #2563eb 100%);
    border-radius: 18px;
    padding: 2.25rem 2.75rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(37, 99, 235, 0.35), 0 0 0 1px rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    margin: 0 0 0.4rem 0;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.2;
}
.hero-banner .subtitle {
    margin: 0;
    opacity: 0.85;
    font-size: 0.95rem;
    font-weight: 400;
    max-width: 600px;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    backdrop-filter: blur(8px);
}

/* Answer Box */
.answer-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.5rem;
    color: #e2e8f0;
    font-size: 0.92rem;
    line-height: 1.75;
    min-height: 120px;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
}
.answer-box .answer-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #60a5fa;
    margin-bottom: 0.6rem;
}
.answer-box p {
    margin: 0;
    white-space: pre-wrap;
}

/* Source Cards */
.sources-section {
    margin-top: 1.25rem;
}
.sources-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.75rem;
}
.source-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.65rem;
    font-size: 0.84rem;
    color: #cbd5e1;
}
.source-card .source-meta {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}
.source-chip {
    display: inline-block;
    background: rgba(37,99,235,0.2);
    color: #93c5fd;
    border: 1px solid rgba(37,99,235,0.3);
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 500;
}
.source-card .source-text {
    color: #94a3b8;
    font-style: italic;
    font-size: 0.81rem;
    line-height: 1.6;
}

/* Info Cards */
.info-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.25rem;
    height: 100%;
}
.info-card h3 {
    margin: 0 0 0.75rem 0;
    color: #60a5fa;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.info-card ul {
    margin: 0;
    padding-left: 1.1rem;
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.8;
}
.info-card li {
    margin-bottom: 0.2rem;
}
.stat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid #334155;
    font-size: 0.82rem;
    color: #94a3b8;
}
.stat-row:last-child { border-bottom: none; }
.stat-val {
    font-weight: 600;
    color: #e2e8f0;
}

/* Footer */
.footer-note {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    margin-top: 1.75rem;
    padding-top: 1rem;
    border-top: 1px solid #1e293b;
}

/* Gradio overrides for dark theme */
.gr-button-primary {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
}
.gr-button-secondary {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    border-radius: 10px !important;
}
"""

# ─── Gradio Theme ─────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#0f172a",
    body_text_color="#e2e8f0",
    block_background_fill="#1e293b",
    block_border_color="#334155",
    block_border_width="1px",
    block_radius="14px",
    block_label_text_color="#94a3b8",
    input_background_fill="#0f172a",
    input_border_color="#334155",
    input_border_color_focus="#2563eb",
    button_primary_background_fill="linear-gradient(135deg, #2563eb, #3b82f6)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8, #2563eb)",
    button_primary_text_color="white",
    button_secondary_background_fill="#1e293b",
    button_secondary_border_color="#334155",
    button_secondary_text_color="#94a3b8",
)

# ─── Product Categories ────────────────────────────────────────────────────────

PRODUCT_OPTIONS = [
    "All Products",
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer",
]

EXAMPLE_QUESTIONS = [
    "What are the most common complaints about credit card billing and fees?",
    "Why are customers frustrated with money transfers?",
    "What problems do people report with their savings accounts?",
    "What issues arise most frequently with personal loans?",
    "Are there patterns of fraud complaints across any products?",
    "What account access problems do customers face?",
    "How do customers describe issues with late payment processing?",
]

# ─── Core Processing Function ──────────────────────────────────────────────────


def format_sources_html(sources: list) -> str:
    """Format source chunks as styled HTML cards."""
    if not sources:
        return ""

    html = '<div class="sources-section">'
    html += '<div class="sources-label">📋 Retrieved Complaint Excerpts (Evidence)</div>'

    for i, src in enumerate(sources, 1):
        html += f'<div class="source-card">'
        html += '<div class="source-meta">'
        html += f'<span class="source-chip">#{i}</span>'
        html += f'<span class="source-chip">ID: {src["complaint_id"]}</span>'
        html += f'<span class="source-chip">{src["product"]}</span>'
        if src.get("issue"):
            html += f'<span class="source-chip">{src["issue"]}</span>'
        html += '</div>'
        html += f'<div class="source-text">"{src["excerpt"]}"</div>'
        html += '</div>'

    html += '</div>'
    return html


def process_query(question: str, product_filter: str, history: list):
    """
    Main RAG processing function called by Gradio.

    Args:
        question: User's input question.
        product_filter: Selected product category.
        history: Chat history (list of [user, assistant] pairs).

    Returns:
        Tuple: (updated_history, answer_html, sources_html, "")
    """
    if not question or not question.strip():
        return history, "<p style='color:#475569;'>Please enter a question.</p>", "", ""

    # Map "All Products" to None for the pipeline
    product_arg = None if product_filter == "All Products" else product_filter

    # Run RAG pipeline
    result = run_rag(question.strip(), k=5, product_filter=product_arg)
    answer = result["answer"]
    sources = result["sources"]

    # Format answer
    answer_html = f"""
    <div class="answer-box">
        <div class="answer-label">🤖 AI-Generated Analysis</div>
        <p>{answer.replace(chr(10), '<br>')}</p>
    </div>
    """

    # Format sources
    sources_html = format_sources_html(sources)

    # Update chat history
    history = history or []
    history.append([question, answer])

    return history, answer_html, sources_html, ""


def clear_all():
    """Reset the entire conversation and outputs."""
    return [], "", "", ""


# ─── Gradio Interface ─────────────────────────────────────────────────────────


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    with gr.Blocks(
        title="CrediTrust — Complaint Analysis Portal",
        theme=theme,
        css=CUSTOM_CSS,
    ) as demo:

        # Header
        gr.HTML("""
        <div class="hero-banner">
            <h1>🏦 CrediTrust Financial</h1>
            <h1 style="font-size:1.35rem; font-weight:500; margin-top:0.2rem;">
                Intelligent Complaint Analysis Portal
            </h1>
            <p class="subtitle">
                Ask plain-English questions about customer complaints across our product lines.
                Powered by semantic search over real CFPB complaint data with AI-generated insights.
            </p>
            <div class="hero-badges">
                <span class="badge">⚡ RAG Pipeline</span>
                <span class="badge">🔍 FAISS Vector Search</span>
                <span class="badge">🤖 AI Generation</span>
                <span class="badge">📊 CFPB Data</span>
            </div>
        </div>
        """)

        with gr.Row():
            # ── Left: Main chat panel ──────────────────────────────────────
            with gr.Column(scale=3):

                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="e.g. Why are people unhappy with credit cards?",
                        label="Your Question",
                        lines=2,
                        max_lines=4,
                        elem_id="question_input",
                    )

                with gr.Row():
                    product_filter = gr.Dropdown(
                        choices=PRODUCT_OPTIONS,
                        value="All Products",
                        label="Filter by Product",
                        elem_id="product_filter",
                        scale=1,
                    )
                    submit_btn = gr.Button(
                        "🔍 Analyze Complaints",
                        variant="primary",
                        scale=2,
                        elem_id="submit_btn",
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        scale=1,
                        elem_id="clear_btn",
                    )

                # Answer display
                answer_display = gr.HTML(
                    value='<div class="answer-box" style="color:#475569;">Your AI-generated analysis will appear here...</div>',
                    label="",
                    elem_id="answer_display",
                )

                # Sources display
                sources_display = gr.HTML(
                    value="",
                    label="",
                    elem_id="sources_display",
                )

                # Example questions
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=question_input,
                    label="💡 Example Questions — Click to Try",
                    elem_id="examples",
                )

            # ── Right: Info panel ─────────────────────────────────────────
            with gr.Column(scale=1):

                gr.HTML("""
                <div class="info-card" style="margin-bottom:1rem;">
                    <h3>📦 Coverage</h3>
                    <ul>
                        <li>Credit Cards</li>
                        <li>Personal Loans</li>
                        <li>Savings Accounts</li>
                        <li>Money Transfers</li>
                    </ul>
                </div>
                """)

                gr.HTML("""
                <div class="info-card" style="margin-bottom:1rem;">
                    <h3>⚙️ How It Works</h3>
                    <ul>
                        <li>Your question is embedded into a 384-dim vector</li>
                        <li>FAISS finds the 5 most similar complaint chunks</li>
                        <li>Retrieved context is injected into an analyst prompt</li>
                        <li>LLM generates a synthesized, grounded answer</li>
                        <li>Source excerpts are shown for full transparency</li>
                    </ul>
                </div>
                """)

                gr.HTML("""
                <div class="info-card">
                    <h3>🎯 Use Cases</h3>
                    <ul>
                        <li>Identify top complaint trends</li>
                        <li>Spot emerging product issues</li>
                        <li>Support compliance reviews</li>
                        <li>Inform product roadmap decisions</li>
                        <li>Reduce manual complaint triage time</li>
                    </ul>
                </div>
                """)

        # Hidden state for chat history
        chat_history = gr.State([])

        # Footer
        gr.HTML("""
        <div class="footer-note">
            CrediTrust Financial — Internal AI Governance Tool &nbsp;|&nbsp;
            Powered by FAISS · sentence-transformers/all-MiniLM-L6-v2 · Google Flan-T5 &nbsp;|&nbsp;
            Data: CFPB Consumer Complaint Database
        </div>
        """)

        # ── Event Handlers ────────────────────────────────────────────────
        submit_btn.click(
            fn=process_query,
            inputs=[question_input, product_filter, chat_history],
            outputs=[chat_history, answer_display, sources_display, question_input],
        )

        question_input.submit(
            fn=process_query,
            inputs=[question_input, product_filter, chat_history],
            outputs=[chat_history, answer_display, sources_display, question_input],
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[chat_history, answer_display, sources_display, question_input],
        )

    return demo


# ─── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-warm: load vector store and LLM at startup
    print("[App] Pre-loading RAG resources...")
    try:
        load_vector_store()
        _get_llm_pipeline()
        print("[App] Resources loaded. Launching Gradio interface...")
    except Exception as e:
        print(f"[App] Warning: Could not pre-load resources: {e}")
        print("[App] Resources will load on first query.")

    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
