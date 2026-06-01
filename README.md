# Tally Prime AI-Powered Voucher Ingestion Pipeline

This service is an automated voucher automation and validation system for Tally Prime. It handles document OCR parsing, multi-agent classification, invoice data extraction, ledger rule matching, GST verification, and posting as XML directly to Tally Prime.

## Directory Structure

```
Tally-Prime/
├── alembic/              # Database migration configuration and versions
│   └── versions/
├── app/                  # FastAPI main application
│   ├── agents/           # Multi-agent graph (LangGraph)
│   ├── api/              # API router endpoints
│   ├── integrations/     # Integrations with OCR engines and Tally Prime XML
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic validation schemas
│   ├── services/         # Orchestrator and service runners
│   ├── config.py         # Application settings configuration
│   ├── database.py       # Async Database connections & helpers
│   └── main.py           # FastAPI Application entrypoint
├── storage/              # Local file storage for raw and processed documents
├── tests/                # Pipeline component testing suite
├── .env.example          # Template environment configurations
├── alembic.ini           # Alembic database configuration
├── pyproject.toml        # Dependencies and packages manifest
└── README.md             # Project documentation
```

## Setup Instructions

### 1. Prerequisites
- **Python**: 3.11+
- **PostgreSQL**: With asyncpg compatibility (e.g. standard PostgreSQL instance)
- **Tally Prime**: Running on the network with HTTP XML servers enabled

### 2. Local Installation

Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -e .
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and fill in the required keys:
```bash
cp .env.example .env
```

Review and adjust variables in `.env`:
*   `DATABASE_URL`: Asynchronous PostgreSQL URI (must start with `postgresql+asyncpg://`)
*   `OPENAI_API_KEY`: API token for OpenAI LLM agents
*   `OLLAMA_BASE_URL`: API URL for local LLM fallback (e.g., http://localhost:11434)
*   `TALLY_HOST` & `TALLY_PORT`: Details of the target Tally Prime server
*   `COMPANY_GSTIN` & `COMPANY_STATE_CODE`: Default company details for voucher generation

### 4. Database Migrations

Apply database migrations using Alembic:
```bash
alembic upgrade head
```

To create a new migration schema:
```bash
alembic revision --autogenerate -m "description_here"
```

### 5. Running the Application

Launch the FastAPI application:
```bash
uvicorn app.main:app --reload --port 8000
```

Once running, navigate to:
- API Documentation (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
- API Alternative Docs (Redoc): [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health check endpoint: [http://localhost:8000/health](http://localhost:8000/health)
