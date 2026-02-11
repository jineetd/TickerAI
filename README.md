# TickerAI 📊

AI-powered stock analysis using local LLM models. Analyze stock tickers with RAG (Retrieval Augmented Generation) - completely free and private.

## Features

- 🦙 **Local LLM** - Uses Llama via Ollama (no API keys needed)
- 🔒 **100% Private** - Everything runs on your machine
- 💰 **Zero Cost** - No API fees
- 🔌 **Extensible** - Easy to swap LLM providers
- 📚 **RAG-Powered** - ChromaDB + semantic search
- 🎯 **Multiple Formats** - Supports TXT, MD, PDF, JSON documents

## Quick Start

### One-Command Setup

```bash
./setup.sh
```

This automated script will:
1. ✅ Check Python 3.10+ installation
2. ✅ Install Ollama
3. ✅ Download Llama model
4. ✅ Create virtual environment
5. ✅ Install dependencies
6. ✅ Initialize knowledge base

### Manual Setup (if preferred)

```bash
# 1. Install Ollama
brew install ollama  # macOS
# OR
curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# 2. Start Ollama & download model
ollama serve  # Keep running
ollama pull llama3.2

# 3. Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Initialize knowledge base
python main.py setup
```

## Usage

### Interactive Mode

```bash
source venv/bin/activate
python main.py interactive
```

Example:
```
📊 Enter ticker and question: AAPL: What are their main products?
📊 Enter ticker and question: TSLA: What are the risks?
📊 Enter ticker and question: stats
📊 Enter ticker and question: quit
```

### Single Query

```bash
python main.py query AAPL "What is Apple's revenue?"
python main.py query TSLA "What are Tesla's competitive advantages?"
```

### Add Your Own Documents

1. Add documents to `knowledge/` directory (TXT, MD, PDF, JSON)
2. Refresh knowledge base:
   ```bash
   python main.py setup --force
   ```

## Configuration

Edit `config.py` or set environment variables:

```bash
# Change LLM model
export LLM_MODEL="llama3.1"  # or llama2, llama3.2:1b, etc.

# Change LLM provider (for future extensibility)
export LLM_PROVIDER="ollama"  # currently only ollama supported

# Adjust generation parameters
export LLM_TEMPERATURE="0.7"
export LLM_MAX_TOKENS="1000"
```

## Architecture

```
User Query → MCP Client → MCP Server → Vector Store (ChromaDB)
                              ↓
                         LLM Provider (Abstracted)
                              ↓
                         Ollama (Llama 3.2)
                              ↓
                          Response
```

### Extensible LLM Design

The application uses an abstraction layer (`llm_provider.py`) that makes it easy to swap LLM providers:

```python
# Currently using Ollama
from llm_provider import OllamaProvider
llm = OllamaProvider(model="llama3.2")

# Future: Switch to OpenAI
from llm_provider import OpenAIProvider
llm = OpenAIProvider(api_key="...", model="gpt-4")

# Or implement your own
class CustomProvider(BaseLLMProvider):
    def generate(self, prompt, ...):
        # Your implementation
        pass
```

## Available Models

| Model | Size | RAM | Best For |
|-------|------|-----|----------|
| llama3.2:1b | 1GB | 4GB | Low-end systems |
| llama3.2 | 2GB | 8GB | **Recommended** |
| llama3.1 | 5GB | 16GB | Better quality |
| llama2 | 4GB | 16GB | Stable option |

Change models:
```bash
ollama pull llama3.1
export LLM_MODEL="llama3.1"
python main.py interactive
```

## Project Structure

```
TickerAI/
├── setup.sh              # One-command setup script
├── README.md             # This file
├── config.py             # Configuration
├── llm_provider.py       # LLM abstraction layer
├── mcp_server.py         # MCP server
├── mcp_client.py         # MCP client
├── vector_store.py       # ChromaDB integration
├── document_processor.py # Document processing
├── main.py               # CLI interface
├── requirements.txt      # Dependencies
└── knowledge/            # Your documents
    ├── AAPL_info.md
    ├── TSLA_info.md
    └── general_stock_analysis.md
```

## Troubleshooting

**Ollama not running:**
```bash
ollama serve
```

**Model not found:**
```bash
ollama pull llama3.2
```

**Import errors:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Slow responses:**
- Use smaller model: `llama3.2:1b`
- Check system resources
- Close other applications

## Extending to Other LLMs

To add a new LLM provider:

1. Edit `llm_provider.py`:
```python
class YourProvider(BaseLLMProvider):
    def generate(self, prompt, system_prompt, temperature, max_tokens):
        # Implement your LLM API call
        return response_text
    
    def get_model_name(self):
        return self.model
```

2. Update factory function:
```python
def get_llm_provider(provider_type):
    if provider_type == "your_provider":
        return YourProvider()
```

3. Set in config:
```bash
export LLM_PROVIDER="your_provider"
```

## Benefits vs Cloud APIs

| Feature | TickerAI (Local) | Cloud APIs |
|---------|------------------|------------|
| Cost | Free | Pay per request |
| Privacy | 100% local | Data sent to server |
| Internet | Not required* | Required |
| Setup | 5 minutes | Instant |
| Quality | Very good | Excellent |

*After initial model download

## Requirements

- Python 3.10+
- 8GB+ RAM (recommended)
- 5GB disk space (for models)
- macOS, Linux, or Windows

## Support

- Check `config.py` for configuration options
- Run `./setup.sh` to reset environment
- Ollama docs: https://ollama.ai
- MCP docs: https://modelcontextprotocol.io

## License

MIT License - Free to use and modify

---

**Built with:** Python | Ollama | ChromaDB | MCP | sentence-transformers
