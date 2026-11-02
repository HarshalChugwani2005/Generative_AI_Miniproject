# Deployment Guide

This guide covers two free hosting options:

1. **Streamlit Community Cloud**, the simplest option, deployed straight from GitHub
2. **Hugging Face Spaces**, using the included `Dockerfile`

Both use Google Gemini by default, so you need a Gemini API key from
<https://aistudio.google.com/app/apikey>. Ollama is not available on these hosts because
there is no local model server, so use **Gemini** or **Groq**.

> **Never commit API keys.** `.env` and `.streamlit/secrets.toml` are already in
> `.gitignore`. Use each platform's secrets manager instead.

---

## Pre-deployment checklist

| Check | How |
|---|---|
| App runs locally | `streamlit run app.py` → ask "How long does a refund take?" |
| Requirements are complete | `pip install -r requirements.txt` in a **fresh** virtual environment, then run the app |
| Python version | 3.12 or 3.13 (`numpy==2.5.4` needs ≥ 3.12) |
| No secrets in Git | `git status` must not show `.env` or `.streamlit/secrets.toml` |
| Data files committed | `data/faq.md` and `data/orders.json` are tracked by Git |
| Admin page protected | Set `ADMIN_PASSWORD` in the platform's secrets |

Optional fresh-environment check:

```bash
python -m venv /tmp/check-env            # Windows: python -m venv %TEMP%\check-env
/tmp/check-env/bin/pip install -r requirements.txt
/tmp/check-env/bin/python -c "import streamlit, langchain_google_genai, faiss, sentence_transformers; print('OK')"
```

---

## 1. Streamlit Community Cloud

### Step 1: Push the project to GitHub

From the repository root (the folder that contains `support-chatbot/`):

```bash
git add support-chatbot
git commit -m "ShopEasy support chatbot"
git push origin main
```

### Step 2: Create the app

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `your-username/Generative_AI_Miniproject`
   - **Branch:** `main`
   - **Main file path:** `support-chatbot/app.py`
4. Open **Advanced settings**:
   - **Python version:** 3.12 (or 3.13)
   - **Secrets:** paste the following (also in `.streamlit/secrets.toml.example`):

     ```toml
     LLM_PROVIDER = "gemini"
     GOOGLE_API_KEY = "your_real_gemini_key"
     GEMINI_MODEL = "gemini-3.5-flash"
     GROQ_API_KEY = ""
     ADMIN_PASSWORD = "choose-a-strong-password"
     ```
5. Click **Deploy**. The first build takes about 5–10 minutes because it installs PyTorch and
   downloads the embedding model.

Community Cloud finds `support-chatbot/requirements.txt` automatically because it sits next to
the main file. The app reads secrets through `config.py`, which checks environment variables
first and then `st.secrets`.

### Theme config on Streamlit Cloud

Streamlit reads `.streamlit/config.toml` from the directory it is started in. Community Cloud
starts from the **repository root**, so copy the config there as well:

```bash
mkdir -p .streamlit
cp support-chatbot/.streamlit/config.toml .streamlit/config.toml
```

Without this the app still works and looks almost the same, because all styling comes from the
CSS in `ui.py`. Only the slider and toggle accent colour fall back to Streamlit's default red.

### If the build fails or runs out of space

The default Linux PyTorch wheel includes large CUDA libraries. To install the smaller CPU-only
build, change the top of `requirements.txt` like this:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.15.0+cpu
```

(and remove the original `torch==2.15.0` line). Commit, push, and click **Reboot app**.

### Updating secrets later

App → **⋮ → Settings → Secrets** → edit → **Save**. The app restarts automatically.

---

## 2. Hugging Face Spaces (Docker)

### Step 1: Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Pick a name (e.g. `shopeasy-support`), choose **Docker** as the SDK and **Blank** as the
   template. **CPU basic (free)** hardware is enough.
3. Click **Create Space**.

### Step 2: Add secrets

Space → **Settings → Variables and secrets → New secret**:

| Name | Value |
|---|---|
| `GOOGLE_API_KEY` | your Gemini key |
| `ADMIN_PASSWORD` | a strong password |
| `LLM_PROVIDER` | `gemini` (optional, it is the default) |

Spaces exposes secrets to the container as environment variables, which `config.py` reads.

### Step 3: Push the code

A Space is its own Git repository, and its `README.md` must start with a YAML header. Copy the
project into the Space repo and add that header:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/shopeasy-support
cd shopeasy-support

# copy the app files (not .venv, .env or vectorstore)
#   Linux/macOS:
rsync -av --exclude .venv --exclude .env --exclude vectorstore ../Generative_AI_Miniproject/support-chatbot/ .
#   Windows (PowerShell):
#   robocopy ..\Generative_AI_Miniproject\support-chatbot . /E /XD .venv vectorstore __pycache__ /XF .env
```

Then put this block at the **very top** of the Space's `README.md`, above the existing title:

```yaml
---
title: ShopEasy Support Chatbot
emoji: 🛍️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
```

Commit and push:

```bash
git add .
git commit -m "Deploy ShopEasy support chatbot"
git push
```

Hugging Face asks for a password when you push. Use an **access token** with *write* permission
from <https://huggingface.co/settings/tokens>, not your account password.

### Step 4: Watch the build

Open the Space → **Logs**. The `Dockerfile`:

- installs CPU-only PyTorch, then the pinned requirements
- runs `python rag.py` to download the embedding model and build the FAISS index at build time
- starts Streamlit on port **7860**

The first build takes about 5–10 minutes. Later pushes reuse cached layers.

### Test the Docker image locally (optional)

```bash
docker build -t shopeasy-support .
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key shopeasy-support
# open http://localhost:7860
```

---

## Notes for both platforms

- **Storage is temporary.** `data/feedback.json`, `data/tickets.json` and
  `data/chat_log.jsonl` reset whenever the app restarts or redeploys. That is fine for a
  demo; a production system would use a database.
- **Free-tier limits.** Gemini's free tier is rate limited. When the limit is hit, users see a
  friendly "please wait a minute" message. Switching to `gemini-3.5-flash-lite` or Groq helps.
- **Sleeping apps.** Both platforms pause idle free apps, and the first visit after a pause
  takes about 30–60 seconds to wake up.
