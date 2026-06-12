"""
Task-01: Text Generation with GPT-2
Fine-tunes GPT-2 on a custom dataset and generates text via a Gradio UI.
"""

import os
import json
import random
import gradio as gr
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
import torch

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"
FINE_TUNED_DIR = "./fine_tuned_gpt2"
TRAIN_FILE = "./train_data.txt"

# ── Sample training data (tech-blog style) ─────────────────────────────────
SAMPLE_TRAINING_TEXT = """
Artificial intelligence is transforming every industry in the modern world.
Machine learning algorithms can now predict outcomes with remarkable accuracy.
Deep learning neural networks have revolutionized computer vision and natural language processing.
The future of AI lies in creating systems that can reason and learn like humans.
Generative models are capable of producing realistic images, text, and audio.
Transfer learning allows models trained on large datasets to be adapted for specific tasks.
The transformer architecture has become the foundation of modern NLP systems.
Attention mechanisms allow models to focus on relevant parts of input sequences.
Fine-tuning pre-trained models is a powerful technique for domain-specific tasks.
Reinforcement learning from human feedback improves the alignment of AI systems.
Large language models have demonstrated emergent capabilities at scale.
Prompt engineering is a new discipline focused on effectively communicating with AI.
AI ethics and responsible deployment are critical concerns for the industry.
Autonomous systems powered by AI are being deployed in transportation and logistics.
Natural language interfaces are making technology accessible to non-technical users.
The convergence of AI and robotics is opening new frontiers in automation.
Data quality and diversity are essential for building robust machine learning models.
Explainability and interpretability remain open challenges in deep learning research.
Edge AI enables intelligent processing directly on devices without cloud dependency.
Multimodal AI systems can process and generate text, images, and audio together.
""".strip()


def prepare_training_data():
    """Write sample training data to disk if not present."""
    if not os.path.exists(TRAIN_FILE):
        with open(TRAIN_FILE, "w") as f:
            # Repeat for a richer corpus
            f.write((SAMPLE_TRAINING_TEXT + "\n") * 20)
        print(f"[✓] Training data written to {TRAIN_FILE}")
    else:
        print(f"[✓] Training data already exists at {TRAIN_FILE}")


def fine_tune_gpt2(epochs: int = 1, batch_size: int = 2) -> str:
    """Fine-tune GPT-2 on the custom dataset."""
    prepare_training_data()

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=TRAIN_FILE,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=FINE_TUNED_DIR,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=1,
        logging_steps=50,
        prediction_loss_only=True,
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(FINE_TUNED_DIR)
    tokenizer.save_pretrained(FINE_TUNED_DIR)

    return f"✅ Fine-tuning complete! Model saved to `{FINE_TUNED_DIR}`"


def generate_text(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    use_fine_tuned: bool = False,
) -> str:
    """Generate text using GPT-2 (base or fine-tuned)."""
    model_path = FINE_TUNED_DIR if (use_fine_tuned and os.path.exists(FINE_TUNED_DIR)) else MODEL_NAME

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    result = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = result[0]["generated_text"]
    model_label = "Fine-tuned GPT-2" if use_fine_tuned and os.path.exists(FINE_TUNED_DIR) else "Base GPT-2"
    return f"[Model: {model_label}]\n\n{generated}"


# ── Gradio UI ──────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="Task-01 · GPT-2 Text Generation",
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="indigo",
            font=gr.themes.GoogleFont("JetBrains Mono"),
        ),
        css="""
        .gradio-container { max-width: 900px; margin: auto; }
        #title { text-align: center; padding: 20px 0 10px; }
        """,
    ) as demo:

        gr.Markdown(
            """
# 🤖 Task-01 — Text Generation with GPT-2
**Fine-tune GPT-2 on custom data, then generate contextually coherent text.**
            """,
            elem_id="title",
        )

        with gr.Tabs():
            # ── Tab 1: Generate ──────────────────────────────────────────
            with gr.TabItem("✍️ Generate Text"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here…",
                            lines=3,
                            value="Artificial intelligence is",
                        )
                        use_ft = gr.Checkbox(
                            label="Use fine-tuned model (run Fine-Tune tab first)",
                            value=False,
                        )
                        max_len = gr.Slider(50, 500, value=200, step=10, label="Max Length (tokens)")
                        temp = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
                        top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-P (nucleus sampling)")
                        gen_btn = gr.Button("🚀 Generate", variant="primary")

                    with gr.Column(scale=1):
                        output_box = gr.Textbox(label="Generated Text", lines=20, interactive=False)

                gen_btn.click(
                    fn=generate_text,
                    inputs=[prompt_input, max_len, temp, top_k, top_p, use_ft],
                    outputs=output_box,
                )

            # ── Tab 2: Fine-Tune ─────────────────────────────────────────
            with gr.TabItem("🏋️ Fine-Tune Model"):
                gr.Markdown(
                    """
### Fine-tune GPT-2 on custom tech-blog text
The training corpus covers AI/ML topics. After training, switch to
*Use fine-tuned model* in the Generate tab to see the difference.
                    """
                )
                with gr.Row():
                    epochs_slider = gr.Slider(1, 5, value=1, step=1, label="Epochs")
                    batch_slider = gr.Slider(1, 8, value=2, step=1, label="Batch Size")
                ft_btn = gr.Button("🏋️ Start Fine-Tuning", variant="primary")
                ft_status = gr.Textbox(label="Status", interactive=False)
                ft_btn.click(
                    fn=fine_tune_gpt2,
                    inputs=[epochs_slider, batch_slider],
                    outputs=ft_status,
                )

            # ── Tab 3: About ─────────────────────────────────────────────
            with gr.TabItem("📖 About"):
                gr.Markdown(
                    """
## How It Works

| Step | Description |
|------|-------------|
| 1 | Load pre-trained **GPT-2** weights from Hugging Face |
| 2 | Prepare a custom text corpus (AI/ML blog style) |
| 3 | Fine-tune with **Hugging Face Trainer** API |
| 4 | Generate text using **sampling strategies** (top-k, top-p, temperature) |

### Key Concepts
- **Temperature**: Controls randomness. Lower → more focused. Higher → more creative.
- **Top-K**: Restricts next-token sampling to top K candidates.
- **Top-P (nucleus)**: Dynamically selects tokens covering P% of probability mass.
- **Fine-tuning**: Adapts GPT-2's weights to match the style of your training corpus.

### Stack
- 🤗 Hugging Face `transformers` · PyTorch · Gradio
                    """
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
