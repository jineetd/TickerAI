"""MCP Server implementation.

Handles client requests, queries vector store, and generates AI responses.
"""

import asyncio
from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

import config
from llm_provider import get_llm_provider
from vector_store import VectorStore

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StockQuery:
    """Stock query data structure."""

    ticker: str
    query: str
    user_context: Optional[str] = None


class TickerAIMCPServer:
    """MCP Server for TickerAI application."""

    def __init__(self):
        self.server = Server("tickerai-server")
        self.vector_store = VectorStore()
        self.llm_provider = get_llm_provider()

        # Register handlers
        self._register_handlers()

        logger.info(
            f"TickerAI MCP Server initialized with "
            f"{config.LLM_PROVIDER} provider, "
            f"model: {self.llm_provider.get_model_name()}"
        )

    def _register_handlers(self):
        """Register MCP server handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools.

            Returns:
                List of available Tool objects.
            """
            return [
                Tool(
                    name="query_stock",
                    description=(
                        "Query information about a stock ticker with "
                        "AI-powered analysis"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": (
                                    "Stock ticker symbol "
                                    "(e.g., AAPL, TSLA)"
                                )
                            },
                            "query": {
                                "type": "string",
                                "description": (
                                    "User's question about the stock"
                                )
                            },
                            "context": {
                                "type": "string",
                                "description": (
                                    "Additional context for the query "
                                    "(optional)"
                                )
                            }
                        },
                        "required": ["ticker", "query"]
                    }
                ),
                Tool(
                    name="get_vector_store_stats",
                    description=(
                        "Get statistics about the vector store database"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="refresh_knowledge_base",
                    description=(
                        "Refresh the knowledge base by "
                        "re-ingesting documents"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(
                name: str,
                arguments: Any
        ) -> list[TextContent]:
            """Handle tool calls.

            Args:
                name: Name of the tool to call.
                arguments: Arguments for the tool.

            Returns:
                List of TextContent responses.
            """
            try:
                if name == "query_stock":
                    result = await self._handle_stock_query(arguments)
                    return [TextContent(type="text", text=result)]

                elif name == "get_vector_store_stats":
                    stats = self.vector_store.get_collection_stats()
                    stats_json = json.dumps(stats, indent=2)
                    return [TextContent(type="text", text=stats_json)]

                elif name == "refresh_knowledge_base":
                    self.vector_store.ingest_documents(
                        force_refresh=True
                    )
                    return [TextContent(
                        type="text",
                        text="Knowledge base refreshed successfully"
                    )]

                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]

            except Exception as e:
                logger.error(f"Error handling tool call {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

    async def _handle_stock_query(
            self,
            arguments: Dict[str, Any]
    ) -> str:
        """Handle stock query request.

        Args:
            arguments: Dictionary containing ticker, query, and context.

        Returns:
            AI-generated response string.
        """
        ticker = arguments.get("ticker", "").upper()
        query = arguments.get("query", "")
        user_context = arguments.get("context", "")

        if not ticker or not query:
            return "Error: Both ticker and query are required"

        logger.info(f"Processing query for {ticker}: {query}")

        # Get relevant context from vector store
        search_query = f"{ticker} {query}"
        context = self.vector_store.get_context_for_query(
            search_query,
            ticker
        )

        # Generate AI response using OpenAI
        response = self._generate_ai_response(
            ticker,
            query,
            context,
            user_context
        )

        return response

    def _generate_ai_response(
            self,
            ticker: str,
            query: str,
            context: str,
            user_context: Optional[str] = None
    ) -> str:
        """Generate AI response using configured LLM provider.

        Args:
            ticker: Stock ticker symbol.
            query: User's question.
            context: Retrieved context from vector store.
            user_context: Optional additional context from user.

        Returns:
            AI-generated response string.
        """

        # Build system prompt
        system_prompt = (
            f"You are a knowledgeable financial analyst AI assistant "
            f"specializing in stock market analysis.\n"
            f"You have access to a knowledge base containing information "
            f"about various stocks.\n\n"
            f"Your task is to answer questions about stock ticker "
            f"{ticker} based on the provided context.\n\n"
            f"Guidelines:\n"
            f"- Be factual and base your answers on the provided context\n"
            f"- If the context doesn't contain enough information, "
            f"acknowledge the limitation\n"
            f"- Provide clear, concise, and actionable insights\n"
            f"- Use professional financial terminology when appropriate\n"
            f"- If asked about predictions, provide balanced analysis "
            f"with appropriate disclaimers\n"
        )

        # Build user prompt
        user_prompt = (
            f"Stock Ticker: {ticker}\n"
            f"User Question: {query}\n\n"
            f"Relevant Context from Knowledge Base:\n"
            f"{context}\n"
        )

        if user_context:
            user_prompt += f"\n\nAdditional User Context: {user_context}"

        user_prompt += (
            "\n\nPlease provide a comprehensive answer to the user's "
            "question based on the context provided."
        )

        try:
            # Use LLM provider abstraction
            logger.info(
                f"Generating response using "
                f"{self.llm_provider.get_model_name()}..."
            )

            answer = self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS
            )

            logger.info(f"Generated AI response for {ticker}")
            return answer

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return f"Error generating response: {str(e)}"

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP Server running on stdio")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point for MCP server."""
    server = TickerAIMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
