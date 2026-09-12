# AI Assistant — A GenAI Flask App with watsonx.ai and LangChain

A small web application that lets you send a message to one of three foundation models hosted on IBM watsonx.ai and get back a structured, JSON-parsed response. Built with Flask on the backend and vanilla JavaScript on the frontend.

This project is a hands-on example from the [IBM RAG and Agentic AI Professional Certificate](https://www.coursera.org/professional-certificates/ibm-rag-and-agentic-ai) on Coursera — specifically the *Develop Generative AI Applications: Get Started* course, which covers building a GenAI web application with Flask and using JSON output parsing for structured AI responses.

---

## What it does

You type a customer-style inquiry into a chat interface, pick a model, and the backend runs your message through a LangChain prompt template tailored to that model's expected format. The model is asked to return structured JSON, which is parsed and validated before being sent back to the browser.

Each model gets its own prompt template because they use different chat markup:

| Model | ID | Template format |
|---|---|---|
| Llama | `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` | Llama header tokens (`<\|begin_of_text\|>`, `<\|start_header_id\|>`) |
| Granite | `ibm/granite-4-h-small` | Plain `System:` / `Human:` / `AI:` |
| Mistral | `mistralai/mistral-small-3-1-24b-instruct-2503` | `<s>[INST] ... [/INST]` |

### Structured output

Rather than returning free text, the model is instructed to produce JSON matching a Pydantic schema:

```python
class AIResponse(BaseModel):
    summary: str    # Summary of the user's message
    sentiment: int  # Sentiment score, 0 (negative) to 100 (positive)
    response: str   # Suggested response to the user
```

LangChain's `JsonOutputParser` generates the format instructions injected into the prompt and parses the model's reply back into a Python dict. The chain is a simple LCEL pipeline:

```python
chain = template | model | json_parser
```

The response time for each call is measured server-side and shown in the UI next to each message.

---

## Project structure

```
.
├── app.py              # Flask app — routes and request handling
├── model.py            # Model initialization, prompt templates, LangChain chains
├── config.py           # Model IDs, generation parameters, watsonx credentials
├── capital.py          # Standalone demo: direct watsonx.ai call without LangChain
├── llm_test.py         # CLI script to compare all three models on one prompt
├── requirements.txt
├── templates/
│   └── index.html      # Chat interface
└── static/
    ├── script.js       # Chat state, fetch calls, message rendering
    └── styles.css      # Styling (IBM Plex Sans)
```

### The backend files

**`app.py`** — Two routes. `GET /` serves the chat page. `POST /generate` takes `{message, model}` as JSON, dispatches to the right model function, times the call, and returns the parsed result with a `duration` field. Returns 400 on missing parameters and 500 on model errors.

**`model.py`** — Initializes three `ChatWatsonx` instances via `langchain_ibm`, defines the per-model prompt templates, and exposes `llama_response()`, `granite_response()`, and `mistral_response()`.

**`config.py`** — Generation parameters (greedy decoding, 256 max new tokens), the watsonx endpoint, and the model IDs.

**`capital.py`** — A minimal example using `ibm_watsonx_ai`'s `ModelInference` directly, no LangChain involved. Useful for seeing what the abstraction layer is doing underneath.

**`llm_test.py`** — Runs the same prompt through all three models and prints the results side by side.

---

## Setup

### Prerequisites

- Python 3.11+
- An IBM watsonx.ai project (see the credentials note below)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/genai_flask_app.git
cd genai_flask_app

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you only want the packages this app actually imports:

```bash
pip install flask langchain-ibm langchain-core ibm-watsonx-ai pydantic
```

### Credentials

This project was written to run inside the **IBM Skills Network Cloud IDE**, which injects watsonx credentials automatically. That's why `config.py` has a `project_id` of `"skills-network"` and no API key — the lab environment handles authentication transparently.

**To run it outside that environment**, you need your own watsonx.ai API key and project ID. Create a `.env` file:

```
WATSONX_APIKEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

And update `model.py` to read from it:

```python
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_model(model_id):
    return ChatWatsonx(
        model_id=model_id,
        url=os.getenv("WATSONX_URL"),
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params=PARAMETERS
    )
```

Never commit the `.env` file. It's already listed in `.gitignore`.

### Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

---

## API

### `POST /generate`

**Request**

```json
{
  "message": "My order arrived damaged and I want a refund.",
  "model": "llama"
}
```

`model` accepts `llama`, `granite`, or `mistral`.

**Response**

```json
{
  "summary": "Customer received a damaged order and is requesting a refund.",
  "sentiment": 15,
  "response": "I'm sorry your order arrived damaged. I can start a refund for you right away.",
  "duration": 1.84
}
```

**Errors**

| Status | Condition |
|---|---|
| 400 | Missing `message` or `model`, or an unrecognized model name |
| 500 | Model call or JSON parsing failed |

---

## Notes and known quirks

A few things worth knowing if you're picking this up or extending it:

- **The UI only displays `response`.** The `summary` and `sentiment` fields come back from the model but aren't rendered. Surfacing sentiment as a colored badge is a natural next exercise.
- **`config.py` defines `CREDENTIALS`, but `model.py` doesn't use it** — the URL and project ID are hardcoded in `initialize_model()`. Worth consolidating.
- **`script.js` sets the default model to `'llama3'`**, which isn't one of the `<select>` options, so the browser falls back to the first option (`llama`). Harmless, but it should say `'llama'`.
- **`llm_test.py` accesses `.content`** on the results, but `JsonOutputParser` returns plain dicts, so this will raise `AttributeError`. Print the dicts directly or index into them.
- **`requirements.txt` is a full `pip freeze` of the lab environment.** It includes a lot of unrelated packages (Airflow, watsonx-orchestrate) and is missing `langchain-ibm` and `ibm-watsonx-ai`, which the app genuinely needs. Use the short `pip install` command above if the full file gives you trouble.
- **Message content is inserted with `innerHTML`.** Fine for a local learning project, but a real deployment should use `textContent` or sanitize, since model output is rendered directly into the DOM.
- **Debug mode is on** (`app.run(debug=True)`). Turn it off before deploying anywhere public.

---

## Possible extensions

- Display the sentiment score and summary in the chat UI
- Keep conversation history and pass it to the model for multi-turn context
- Add streaming responses instead of waiting for the full generation
- Add retry logic for when a model returns malformed JSON
- Extend into a RAG pipeline with a vector store, following the later courses in the certificate

---

## Acknowledgements

Built as part of the [IBM RAG and Agentic AI Professional Certificate](https://www.coursera.org/professional-certificates/ibm-rag-and-agentic-ai), offered by IBM through Coursera. The certificate covers LangChain, LangGraph, RAG pipelines, vector databases, multimodal AI, and agentic frameworks such as CrewAI, AG2, BeeAI, and the Model Context Protocol.

Model access is provided by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai).

## License

MIT — see `LICENSE` if you add one.
