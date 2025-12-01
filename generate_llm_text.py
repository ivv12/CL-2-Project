import os
from pathlib import Path

# -------------------------------
# 1. SETUP API CLIENTS
# -------------------------------

# OpenAI (GPT-4.1, GPT-5)
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic (Claude 3.5, Claude 4 etc.)
from anthropic import Anthropic
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Google Gemini (Gemini 2.0, 2.5, etc.)
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------------
# 2. THE FIXED PROMPT
# -------------------------------

PROMPT = """
Write an original long-form reflective essay about the idea that life is short.
Use a modern, conversational tone.
Blend personal observations, practical reflections, and emotional honesty.
Include small anecdotes or examples, but do not copy or imitate any existing author or essay.
Focus on how people spend time, how priorities change, and how modern life creates distractions.
Make it about 900–1200 words of continuous prose.
"""

# -------------------------------
# 3. OUTPUT FOLDER
# -------------------------------

BASE_DIR = Path("data/llm")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 4. MODEL LIST
# -------------------------------

MODELS = {
    "gpt-4.1": "openai",
    "gpt-5": "openai",                   # If not available, remove
    "claude-3-5-sonnet-latest": "anthropic",
    "gemini-2.0-pro": "google"
}

# -------------------------------
# 5. GENERATORS FOR EACH PROVIDER
# -------------------------------

def generate_openai(model, sample_id):
    """Generate text using OpenAI models (GPT-4.1 / GPT-5)."""
    response = openai_client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=1600,
        messages=[{"role": "user", "content": PROMPT}]
    )
    return response.choices[0].message["content"]


def generate_anthropic(model, sample_id):
    """Generate text from Claude models."""
    response = anthropic_client.messages.create(
        model=model,
        temperature=0.7,
        max_tokens=1600,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return response.content[0].text


def generate_google(model, sample_id):
    """Generate text using Google Gemini models."""
    gen_model = genai.GenerativeModel(model)
    response = gen_model.generate_content(
        PROMPT,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1600,
        }
    )
    return response.text


# -------------------------------
# 6. SAVE FUNCTION
# -------------------------------

def save_text(model, sample_id, text):
    model_dir = BASE_DIR / model.replace(".", "_")
    model_dir.mkdir(exist_ok=True)

    out_path = model_dir / f"sample_{sample_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved: {out_path}")


# -------------------------------
# 7. MAIN LOOP
# -------------------------------

if __name__ == "__main__":
    for model, provider in MODELS.items():
        print(f"\n=== Generating for {model} ({provider}) ===")
        for sample_id in range(1, 4):  # 3 samples
            print(f"  → Sample {sample_id}")

            if provider == "openai":
                text = generate_openai(model, sample_id)
            elif provider == "anthropic":
                text = generate_anthropic(model, sample_id)
            elif provider == "google":
                text = generate_google(model, sample_id)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            save_text(model, sample_id, text)

    print("\nDONE — All LLM samples generated.")
