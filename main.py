"""Main application file for TickerAI.

Entry point for the application with CLI interface.
"""

import argparse
import asyncio
import logging
import sys

import config
from mcp_client import TickerAIClientInterface
from vector_store import VectorStore

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_knowledge_base(force_refresh: bool = False):
    """Initialize the knowledge base by ingesting documents.

    Args:
        force_refresh: Whether to force refresh of the knowledge base.

    Returns:
        True if setup was successful, False otherwise.
    """
    print("\n" + "=" * 60)
    print("TickerAI Knowledge Base Setup")
    print("=" * 60)

    # Check if knowledge directory has files
    if not config.KNOWLEDGE_DIR.exists():
        print(
            f"\nError: Knowledge directory does not exist: "
            f"{config.KNOWLEDGE_DIR}"
        )
        print("Please create the directory and add some documents.")
        return False

    files = list(config.KNOWLEDGE_DIR.rglob('*'))
    doc_files = [
        f for f in files
        if f.is_file() and f.suffix in config.SUPPORTED_FILE_TYPES
    ]

    if not doc_files:
        print(f"\nWarning: No documents found in {config.KNOWLEDGE_DIR}")
        supported = ', '.join(config.SUPPORTED_FILE_TYPES)
        print(f"Supported file types: {supported}")
        print("\nPlease add some documents to the knowledge directory.")
        return False

    print(f"\nFound {len(doc_files)} documents:")
    for file in doc_files:
        print(f"  - {file.relative_to(config.KNOWLEDGE_DIR)}")

    # Initialize vector store and ingest documents
    print("\nInitializing vector store...")
    vector_store = VectorStore()

    print("Ingesting documents into ChromaDB...")
    vector_store.ingest_documents(force_refresh=force_refresh)

    # Get stats
    stats = vector_store.get_collection_stats()
    print("\n" + "-" * 60)
    print("Knowledge Base Statistics:")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Total chunks: {stats['document_count']}")
    print(f"  Database path: {stats['db_path']}")
    print("-" * 60)

    print("\n✓ Knowledge base setup complete!")
    return True


async def interactive_mode():
    """Run interactive query mode."""
    print("\n" + "=" * 60)
    print("TickerAI Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  - Type your question about a stock")
    print("  - Type 'stats' to see vector store statistics")
    print("  - Type 'refresh' to refresh knowledge base")
    print("  - Type 'quit' or 'exit' to exit")
    print("=" * 60 + "\n")

    async with TickerAIClientInterface() as ticker_ai:
        while True:
            try:
                # Get user input
                prompt = (
                    "\n📊 Enter ticker and question "
                    "(e.g., 'AAPL: What is their revenue?'): "
                )
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'stats':
                    stats = await ticker_ai.get_stats()
                    print("\n📈 Vector Store Statistics:")
                    print(
                        f"  Collection: "
                        f"{stats.get('collection_name', 'N/A')}"
                    )
                    print(
                        f"  Documents: "
                        f"{stats.get('document_count', 0)}"
                    )
                    print(f"  DB Path: {stats.get('db_path', 'N/A')}")
                    continue

                if user_input.lower() == 'refresh':
                    print("\n🔄 Refreshing knowledge base...")
                    result = await ticker_ai.refresh()
                    print(f"  {result}")
                    continue

                # Parse ticker and question
                if ':' in user_input:
                    ticker, question = user_input.split(':', 1)
                    ticker = ticker.strip().upper()
                    question = question.strip()
                else:
                    print("⚠️  Please use format: TICKER: Your question")
                    print("   Example: AAPL: What is their revenue?")
                    continue

                if not ticker or not question:
                    print("⚠️  Both ticker and question are required")
                    continue

                # Query the stock
                print(f"\n🔍 Querying {ticker}...")
                response = await ticker_ai.ask(ticker, question)

                print("\n" + "-" * 60)
                print("📝 AI Response:")
                print("-" * 60)
                print(response)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ Error: {e}")


async def query_mode(ticker: str, query: str, context: str = None):
    """Single query mode.

    Args:
        ticker: Stock ticker symbol.
        query: Question about the stock.
        context: Optional additional context.
    """
    print("\n" + "=" * 60)
    print(f"Querying {ticker}")
    print("=" * 60)
    print(f"Question: {query}")
    if context:
        print(f"Context: {context}")
    print("=" * 60 + "\n")

    async with TickerAIClientInterface() as ticker_ai:
        print("🔍 Processing query...")
        response = await ticker_ai.ask(ticker, query, context)

        print("\n" + "-" * 60)
        print("📝 AI Response:")
        print("-" * 60)
        print(response)
        print("-" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TickerAI - AI-powered stock analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )

    # Setup command
    setup_parser = subparsers.add_parser(
        'setup',
        help='Setup knowledge base'
    )
    setup_parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh of knowledge base'
    )

    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Query a stock ticker'
    )
    query_parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol (e.g., AAPL)'
    )
    query_parser.add_argument(
        'question',
        type=str,
        help='Question about the stock'
    )
    query_parser.add_argument(
        '--context',
        type=str,
        help='Additional context'
    )

    # Interactive command
    subparsers.add_parser('interactive', help='Run in interactive mode')

    args = parser.parse_args()

    # Execute command
    if args.command == 'setup':
        success = setup_knowledge_base(force_refresh=args.force)
        sys.exit(0 if success else 1)

    elif args.command == 'query':
        asyncio.run(query_mode(args.ticker, args.question, args.context))

    elif args.command == 'interactive':
        asyncio.run(interactive_mode())

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py setup")
        print("  python main.py interactive")
        print("  python main.py query AAPL 'What is their revenue?'")


if __name__ == "__main__":
    main()
