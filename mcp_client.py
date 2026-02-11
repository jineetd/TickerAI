"""MCP Client implementation.

Communicates with MCP server to query stock information.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TickerAIMCPClient:
    """MCP Client for TickerAI application."""

    def __init__(self, server_script_path: str = "mcp_server.py"):
        self.server_script_path = server_script_path
        self.session: Optional[ClientSession] = None
        logger.info("TickerAI MCP Client initialized")

    async def connect(self):
        """Connect to the MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path],
            env=None
        )

        logger.info(
            f"Connecting to MCP server: {self.server_script_path}"
        )

        # Create stdio client context
        self.stdio_context = stdio_client(server_params)
        streams = await self.stdio_context.__aenter__()
        self.read_stream, self.write_stream = streams

        # Create client session
        self.session = ClientSession(
            self.read_stream,
            self.write_stream
        )
        await self.session.__aenter__()

        # Initialize the session
        await self.session.initialize()

        logger.info("Connected to MCP server successfully")

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'stdio_context'):
            await self.stdio_context.__aexit__(None, None, None)
        logger.info("Disconnected from MCP server")

    async def list_available_tools(self) -> list:
        """List all available tools from the server.

        Returns:
            List of available tools.

        Raises:
            RuntimeError: If not connected to server.
        """
        if not self.session:
            raise RuntimeError(
                "Not connected to server. Call connect() first."
            )

        result = await self.session.list_tools()
        return result.tools

    async def query_stock(
            self,
            ticker: str,
            query: str,
            context: Optional[str] = None
    ) -> str:
        """Query stock information through MCP server.

        Args:
            ticker: Stock ticker symbol.
            query: Question about the stock.
            context: Optional additional context.

        Returns:
            AI-generated response string.

        Raises:
            RuntimeError: If not connected to server.
        """
        if not self.session:
            raise RuntimeError(
                "Not connected to server. Call connect() first."
            )

        logger.info(f"Querying stock {ticker}: {query}")

        # Prepare arguments
        arguments = {
            "ticker": ticker,
            "query": query
        }

        if context:
            arguments["context"] = context

        # Call the tool
        result = await self.session.call_tool(
            "query_stock",
            arguments
        )

        # Extract text content from result
        if result.content and len(result.content) > 0:
            return result.content[0].text

        return "No response received from server"

    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.

        Returns:
            Dictionary with statistics.

        Raises:
            RuntimeError: If not connected to server.
        """
        if not self.session:
            raise RuntimeError(
                "Not connected to server. Call connect() first."
            )

        logger.info("Requesting vector store stats")

        result = await self.session.call_tool(
            "get_vector_store_stats",
            {}
        )

        if result.content and len(result.content) > 0:
            stats_str = result.content[0].text
            return json.loads(stats_str)

        return {}

    async def refresh_knowledge_base(self) -> str:
        """Request server to refresh the knowledge base.

        Returns:
            Status message string.

        Raises:
            RuntimeError: If not connected to server.
        """
        if not self.session:
            raise RuntimeError(
                "Not connected to server. Call connect() first."
            )

        logger.info("Requesting knowledge base refresh")

        result = await self.session.call_tool(
            "refresh_knowledge_base",
            {}
        )

        if result.content and len(result.content) > 0:
            return result.content[0].text

        return "No response received from server"


class TickerAIClientInterface:
    """High-level interface for interacting with TickerAI."""

    def __init__(self):
        self.client = TickerAIMCPClient()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.disconnect()

    async def ask(
            self,
            ticker: str,
            question: str,
            context: Optional[str] = None
    ) -> str:
        """Ask a question about a stock ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA').
            question: Question about the stock.
            context: Optional additional context.

        Returns:
            AI-generated response.
        """
        return await self.client.query_stock(
            ticker,
            question,
            context
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.

        Returns:
            Dictionary with statistics.
        """
        return await self.client.get_vector_store_stats()

    async def refresh(self) -> str:
        """Refresh the knowledge base.

        Returns:
            Status message string.
        """
        return await self.client.refresh_knowledge_base()

    async def list_tools(self) -> list:
        """List available tools.

        Returns:
            List of available tools.
        """
        return await self.client.list_available_tools()


async def example_usage():
    """Example usage of the MCP client."""
    async with TickerAIClientInterface() as ticker_ai:
        # List available tools
        print("\n=== Available Tools ===")
        tools = await ticker_ai.list_tools()
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

        # Get vector store stats
        print("\n=== Vector Store Statistics ===")
        stats = await ticker_ai.get_stats()
        print(json.dumps(stats, indent=2))

        # Query a stock
        print("\n=== Stock Query Example ===")
        response = await ticker_ai.ask(
            ticker="AAPL",
            question="What is the company's recent performance?"
        )
        print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(example_usage())
