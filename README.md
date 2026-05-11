# SHL Assessment Recommendation System

> A production-ready conversational AI agent that helps recruiters and hiring managers discover the right SHL assessments through natural conversation.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup & Installation](#setup--installation)
4. [Running the System](#running-the-system)
5. [API Reference](#api-reference)
6. [Retrieval Pipeline](#retrieval-pipeline)
7. [Conversation Behavior](#conversation-behavior)
8. [Security](#security)
9. [Evaluation](#evaluation)
10. [Deployment](#deployment)
11. [Design Tradeoffs](#design-tradeoffs)
12. [Limitations](#limitations)
13. [Future Improvements](#future-improvements)

---

## Overview

This system implements a **stateless, conversational RAG agent** that:

- Understands vague hiring requirements and asks targeted clarifying questions
- Recommends 1–10 SHL assessments from the official catalog (no hallucinations)
- Refines recommendations dynamically based on mid-conversation changes
- Compares assessments using grounded catalog data
- Refuses off-topic queries and prompt injection attempts
- Returns a strict, non-negotiable JSON schema on every response

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI App                          │
│  GET /health          POST /chat                         │
└────────────────────────┬────────────────────────────────┘
                         │
                ┌────────▼────────┐
                │  Security Guard │  ← Injection detection
                │  Input Validator│    Off-topic refusal
                └────────┬────────┘    Query classification
                         │
                ┌────────▼────────┐
                │  Conversation   │  ← Stateless context
                │  Manager        │    extraction from
                └────────┬────────┘    full history
                         │
                ┌────────▼────────┐
                │  Retrieval      │  ← Semantic search
                │  Engine         │    Metadata filtering
                └────────┬────────┘    Hybrid re-ranking
                         │
              ┌──────────▼──────────┐
              │    Vector Store      │  ← FAISS / ChromaDB
              │ (SHL Catalog Index) │    Sentence Transformers
              └──────────▲──────────┘    all-MiniLM-L6-v2
                         │
              ┌──────────┴──────────┐
              │    Data Pipeline    │  ← Scraper (BS4/Playwright)
              │  Scrape→Clean→Index │    Cleaner + Normalizer
              └─────────────────────┘    Embedding Generator
                         │
                ┌────────▼────────┐
                │  Prompt Builder │  ← RAG prompt with
                │  (Templates)    │    catalog context
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │   LLM Client    │  ← Groq/OpenAI/Gemini
                │  (Multi-provider│    OpenRouter
                │   with retry)   │    JSON enforcement
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │  Response       │  ← Catalog grounding
                │  Validator      │    Hallucination check
                └─────────────────┘    Schema enforcement
```

### Module Breakdown

| Module | Purpose |
|--------|---------|
| `app/main.py` | FastAPI app, middleware, lifespan |
| `app/routes/chat.py` | `/health` and `/chat` endpoints |
| `app/services/agent.py` | Core orchestration engine |
| `app/services/conversation.py` | Stateless context extraction |
| `app/services/llm_client.py` | Multi-provider LLM client |
| `app/services/startup.py` | DI container, singleton management |
| `app/retrieval/embeddings.py` | Sentence Transformer embedding gen |
| `app/retrieval/vector_store.py` | FAISS + ChromaDB implementations |
| `app/retrieval/retrieval_engine.py` | Hybrid search + re-ranking |
| `app/prompts/templates.py` | RAG prompt engineering |
| `app/security/guards.py` | Injection detection, scope enforcement |
| `app/models/schemas.py` | Pydantic models, exact response schema |
| `app/models/config.py` | Settings from .env |
| `app/evaluation/evaluator.py` | Full evaluation suite |
| `scraper/shl_scraper.py` | SHL catalog scraper |
| `scraper/cleaner.py` | Data cleaning pipeline |
| `run_pipeline.py` | CLI pipeline runner |

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- At least one LLM API key (Groq recommended — free tier available)

### Step 1: Clone & Enter Directory

```bash
cd SHL
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Copy `.env` and fill in your API keys:

```bash
# Edit .env
LLM_PROVIDER=groq                    # groq | openai | google | openrouter
GROQ_API_KEY=your_groq_api_key_here  # Get free at console.groq.com
GOOGLE_API_KEY=your_google_key       # Optional
OPENAI_API_KEY=your_openai_key       # Optional
```

> **Recommended**: Use Groq (free tier, fast inference) with `llama-3.3-70b-versatile`

### Step 5: Run the Data Pipeline

```bash
# Full pipeline: scrape + clean + build vector index
python run_pipeline.py --all

# Or individual steps:
python run_pipeline.py --scrape   # Scrape SHL catalog
python run_pipeline.py --clean    # Clean scraped data
python run_pipeline.py --index    # Build FAISS index
```

> **Note**: Scraping may take 5–15 minutes depending on network speed.
> The vector index is built from `data/shl_catalog.json`.

---

## Running the System

### Development Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Docker

```bash
# Build
docker build -t shl-api .

# Run
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e LLM_PROVIDER=groq \
  shl-api

# Docker Compose
docker-compose up
```

---

## API Reference

### GET /health

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**
```json
{"status": "ok"}
```

---

### POST /chat

**CRITICAL: Send FULL conversation history on every request (stateless).**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a Java developer"}
    ]
  }'
```

**Response Schema (NON-NEGOTIABLE):**
```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "assessment name",
      "url": "official SHL URL",
      "test_type": "type"
    }
  ],
  "end_of_conversation": false
}
```

---

### Sample Conversations

#### 1. Vague Query → Clarification

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need an assessment"}
  ]
}
```

**Response:**
```json
{
  "reply": "I'd love to help you find the right SHL assessment! Could you tell me: 1) What role are you hiring for? 2) Is this a technical or non-technical position? 3) What seniority level?",
  "recommendations": [],
  "end_of_conversation": false
}
```

---

#### 2. Specific Query → Recommendations

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a senior Java developer. Need cognitive ability and personality tests. Remote testing is required."}
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements for a senior Java developer, here are my top recommendations...",
  "recommendations": [
    {
      "name": "Verify G+ (Global Skills Assessment)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus/",
      "test_type": "Cognitive Ability"
    },
    {
      "name": "OPQ32",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32/",
      "test_type": "Personality"
    }
  ],
  "end_of_conversation": false
}
```

---

#### 3. Multi-turn with Refinement

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a software engineer"},
    {"role": "assistant", "content": "{\"reply\": \"What seniority level and assessment types?\", \"recommendations\": [], \"end_of_conversation\": false}"},
    {"role": "user", "content": "Senior. Need coding tests. Actually, also add personality assessment."}
  ]
}
```

---

#### 4. Comparison Query

```json
{
  "messages": [
    {"role": "user", "content": "What is the difference between OPQ32 and a cognitive ability test for leadership roles?"}
  ]
}
```

---

#### 5. Off-Topic Refusal

```json
{
  "messages": [
    {"role": "user", "content": "What salary should I offer a Java developer?"}
  ]
}
```

**Response:**
```json
{
  "reply": "I'm the SHL Assessment Advisor and I'm specialized only in helping you find the right SHL assessments. I can't assist with salary advice. Would you like help discovering SHL assessments for a specific role?",
  "recommendations": [],
  "end_of_conversation": false
}
```

---

## Retrieval Pipeline

### Embedding Strategy

All assessments are embedded using **`all-MiniLM-L6-v2`** (384-dim, normalized):
- Fast (500K sentences/sec on CPU)
- High quality for semantic similarity
- Normalized vectors → cosine similarity = inner product

### What's Embedded

Each assessment combines:
```
name | description | test_type | category | skills_measured | tags | keywords
```

### Search Flow

```
User Query
    │
    ▼
Embed Query (sentence-transformers)
    │
    ▼
FAISS Inner Product Search (cosine similarity)
    │
    ▼
Metadata Filtering (remote, adaptive, type, duration)
    │
    ▼
Keyword Boost Re-ranking (hybrid search)
    │
    ▼
Top-K Results → Prompt Context
```

### Hybrid Search

Combines semantic similarity with BM25-inspired keyword boost:
- Semantic score from FAISS
- +0.05 per keyword hit in raw_text
- +0.10 per keyword hit in assessment name
- Capped at +0.30 boost
- Results re-sorted by combined score

---

## Conversation Behavior

### Stateless Design

Every `/chat` request must include the **full conversation history**. The server:
- Extracts context by analyzing ALL user messages
- Detects role, seniority, requirements, refinements
- Identifies previous recommendations from assistant messages
- Builds an appropriate search query

### Query Classification Pipeline

```
User Message
    │
    ├── Prompt Injection? → REFUSE immediately
    ├── Off-Topic? → REFUSE with redirect
    ├── Closing? → CLOSE conversation
    ├── Comparison? → COMPARE mode
    ├── Refinement? → REFINE mode
    ├── Vague? → CLARIFY mode
    └── Specific? → RECOMMEND mode
```

### Context Extraction

The `ConversationManager` extracts from full history:
- **Role**: Java, Python, Sales, Manager, etc.
- **Seniority**: junior, mid, senior, executive
- **Requirements**: coding, personality, cognitive, language, leadership
- **Constraints**: remote testing, max duration, adaptive

---

## Security

### Prompt Injection Defense

Detects patterns like:
- "Ignore all previous instructions"
- "Pretend you are..."
- "Reveal your system prompt"
- "You are now DAN"
- "Jailbreak mode"

Returns a safe refusal without exposing internals.

### Scope Enforcement

Off-topic detection for:
- Legal/salary advice
- General coding help
- Medical/financial advice
- Non-SHL products

### Hallucination Prevention

1. **Catalog grounding**: LLM receives only retrieved catalog context
2. **Response validation**: Recommendations matched against retrieved items
3. **URL validation**: Only `shl.com` URLs pass validation
4. **Name matching**: Fuzzy match against catalog names rejects invented ones

### Input Validation

- Max message length: 8192 chars
- Max messages: 50
- Null byte removal
- Control character stripping
- Role validation (user/assistant/system only)
- Last message must be from user

---

## Evaluation

### Running Evaluations

```bash
# Start server first
uvicorn app.main:app --port 8000

# Run evaluations
python -m app.evaluation.evaluator
```

### Test Categories

| Category | Tests | What's Measured |
|----------|-------|----------------|
| Vague Queries | 3 | Clarification behavior |
| Specific Queries | 4 | Recommendation quality + URLs |
| Off-Topic Refusals | 3 | Scope enforcement |
| Injection Attacks | 2 | Security |
| Multi-turn | 1 | Refinement handling |
| Comparisons | 1 | Comparison quality |

### Metrics

- **Schema Compliance Rate**: % responses with correct JSON schema
- **Hallucination Rate**: % responses with non-SHL URLs
- **Refusal Accuracy**: % off-topic queries properly refused
- **Recall@10**: Relevant assessments in top 10 results
- **Avg Latency**: Mean response time (target: <30s)

### Running Unit Tests

```bash
pytest tests/ -v
```

---

## Deployment

### Render

```bash
# Push to GitHub, then connect to Render
# render.yaml handles configuration automatically
```

Set environment variables in Render dashboard:
- `GROQ_API_KEY`
- `LLM_PROVIDER=groq`

### Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway up
```

Set vars: `GROQ_API_KEY`, `LLM_PROVIDER=groq`

### Docker

```bash
docker build -t shl-api .
docker run -p 8000:8000 -e GROQ_API_KEY=xxx shl-api
```

---

## Design Tradeoffs

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector Store | FAISS | Fastest for CPU-only deployment; no external service needed |
| Embedding Model | all-MiniLM-L6-v2 | Best speed/quality balance; 384-dim, runs on CPU |
| LLM | Groq (Llama 3.3 70B) | Free tier, sub-second inference, production quality |
| State Management | Stateless | Assignment requirement; enables horizontal scaling |
| Retrieval Strategy | Hybrid (semantic + keyword) | Better precision than pure semantic alone |
| Response Validation | Post-LLM catalog matching | Prevents any hallucinated recommendations |

---

## Limitations

1. **Scraping**: SHL may update their catalog structure, requiring scraper updates
2. **Context window**: Long conversations may need summarization (implemented)
3. **Cold start**: Vector index must be pre-built before serving requests
4. **Single instance**: FAISS index is in-memory; not shared across processes
5. **LLM dependency**: Quality depends on the chosen LLM provider

---

## Future Improvements

- [ ] **Reranker**: Cross-encoder reranker (ms-marco-MiniLM) for precision
- [ ] **Streaming**: SSE streaming responses for better UX
- [ ] **Redis cache**: Cache embeddings and frequent queries
- [ ] **Analytics dashboard**: Request logging + visualization
- [ ] **Confidence scores**: Explicit confidence in recommendations
- [ ] **Multi-language**: Support non-English hiring requirements
- [ ] **Async scraper**: Playwright async for faster catalog updates
- [ ] **Catalog versioning**: Track catalog changes over time
- [ ] **A/B testing**: Compare recommendation strategies

---

## Project Structure

```
SHL/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── routes/
│   │   └── chat.py             # /health and /chat endpoints
│   ├── services/
│   │   ├── agent.py            # Core orchestration
│   │   ├── conversation.py     # Stateless context extraction
│   │   ├── llm_client.py       # Multi-provider LLM client
│   │   └── startup.py          # DI container
│   ├── prompts/
│   │   └── templates.py        # RAG prompt engineering
│   ├── retrieval/
│   │   ├── embeddings.py       # Embedding generation
│   │   ├── vector_store.py     # FAISS + ChromaDB
│   │   └── retrieval_engine.py # Hybrid search + reranking
│   ├── models/
│   │   ├── schemas.py          # Pydantic models
│   │   └── config.py           # Settings
│   ├── security/
│   │   └── guards.py           # Security module
│   ├── evaluation/
│   │   └── evaluator.py        # Evaluation suite
│   └── utils/
│       └── logging_config.py   # Structured logging
├── scraper/
│   ├── shl_scraper.py          # SHL catalog scraper
│   └── cleaner.py              # Data cleaning pipeline
├── data/                       # Scraped + cleaned data
├── vectorstore/                # FAISS index files
├── tests/
│   ├── test_api.py             # API integration tests
│   └── test_security.py        # Security unit tests
├── run_pipeline.py             # Data pipeline CLI
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker config
├── docker-compose.yml          # Docker Compose
├── render.yaml                 # Render deployment
├── railway.toml                # Railway deployment
└── README.md                   # This file
```
