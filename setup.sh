#!/bin/bash
# Complete setup script for TickerAI

set -e

echo ""
echo "=========================================="
echo "     TickerAI Complete Setup Script"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}✗ Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    echo "  Install: brew install python@3.10"
    exit 1
else
    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
fi
echo ""

# Step 2: Check Ollama
echo "Step 2: Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}✗ Ollama not found${NC}"
    echo ""
    echo "Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  macOS: brew install ollama"
        brew install ollama
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  Linux: curl -fsSL https://ollama.ai/install.sh | sh"
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo -e "${RED}Please install Ollama manually: https://ollama.ai/download${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi
echo ""

# Step 3: Start Ollama service
echo "Step 3: Starting Ollama service..."
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}✓ Ollama service already running${NC}"
else
    echo "  Starting Ollama in background..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    echo -e "${GREEN}✓ Ollama service started${NC}"
fi
echo ""

# Step 4: Download Llama model
echo "Step 4: Downloading Llama model..."
MODEL_NAME="llama3.2"
if ollama list | grep -q "$MODEL_NAME"; then
    echo -e "${GREEN}✓ Model $MODEL_NAME already downloaded${NC}"
else
    echo "  Downloading $MODEL_NAME (this may take a few minutes)..."
    ollama pull $MODEL_NAME
    echo -e "${GREEN}✓ Model downloaded${NC}"
fi
echo ""

# Step 5: Setup Python virtual environment
echo "Step 5: Setting up Python virtual environment..."
if [ -d "venv" ]; then
    echo "  Removing old virtual environment..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment created${NC}"
echo ""

# Step 6: Install Python dependencies
echo "Step 6: Installing Python dependencies..."
echo "  Upgrading pip..."
pip install --upgrade pip --quiet

echo "  Installing packages (this may take a few minutes)..."
pip install ollama sentence-transformers chromadb --quiet
pip install git+https://github.com/modelcontextprotocol/python-sdk.git --quiet
pip install pypdf2 python-docx python-dotenv requests colorlog --quiet

echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# Step 7: Create .env file
echo "Step 7: Creating configuration file..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_HOST=http://localhost:11434

# To use a different model, change LLM_MODEL:
# LLM_MODEL=llama3.1
# LLM_MODEL=llama2
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi
echo ""

# Step 8: Verify installation
echo "Step 8: Verifying installation..."
python -c "import ollama; import chromadb; from sentence_transformers import SentenceTransformer; from mcp import ClientSession; print('✓ All imports successful')" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Installation verified${NC}"
else
    echo -e "${RED}✗ Verification failed${NC}"
    exit 1
fi
echo ""

# Step 9: Setup knowledge base
echo "Step 9: Initializing knowledge base..."
if [ -d "knowledge" ] && [ "$(ls -A knowledge)" ]; then
    echo "  Found documents in knowledge/ directory"
    python main.py setup
    echo -e "${GREEN}✓ Knowledge base initialized${NC}"
else
    echo -e "${YELLOW}⚠ No documents found in knowledge/ directory${NC}"
    echo "  Add your documents to knowledge/ and run: python main.py setup"
fi
echo ""

# Done!
echo "=========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To start using TickerAI:"
echo ""
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run interactive mode:"
echo "     python main.py interactive"
echo ""
echo "  3. Or run a single query:"
echo "     python main.py query AAPL 'What are their products?'"
echo ""
echo "Need help? Check README.md"
echo ""
