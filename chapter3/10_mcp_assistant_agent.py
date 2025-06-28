"""
Chapter 3: Agent Communication Patterns
Example 10: MCP Assistant Agent

Description:
Demonstrates using the Model Context Protocol (MCP) with an assistant agent to enable
external tool integration. Shows how to create an agent that can fetch and process
web content using MCP server capabilities for enhanced information retrieval.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- MCP extensions installed (autogen-ext)
- mcp-server-fetch package available via uvx

Usage:
```bash
python -m chapter3.10_mcp_assistant_agent
```

Expected Output:
A comprehensive summary of the Wikipedia page for "Agentic AI" fetched and processed
through the MCP workbench. The agent will:
1. Connect to the MCP fetch server
2. Retrieve the Wikipedia page content
3. Analyze and summarize the information
4. Present structured insights about Agentic AI

Key Concepts:
- Model Context Protocol (MCP) integration
- External tool workbench setup
- Web content fetching and processing
- Streaming output with tool reflection
- Server parameter configuration
- Asynchronous tool execution

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config


async def main() -> None:
    """Main execution function demonstrating MCP assistant agent."""
    # Configuration and setup
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config)
    
    # Configure MCP server parameters for web fetching
    params = StdioServerParams(
        command="uvx", 
        args=["mcp-server-fetch"]
    )
    
    # Execute with MCP workbench context
    async with McpWorkbench(params) as wb:
        # Create assistant agent with MCP workbench integration
        agent = AssistantAgent(
            name="fetcher",
            model_client=client,
            workbench=wb,                    # Enable MCP tool access
            reflect_on_tool_use=True,        # Enhance tool usage reasoning
            model_client_stream=True         # Enable streaming output
        )
        
        # Execute web content summarization task
        print("=== MCP Agent Web Content Analysis ===")
        result = agent.run_stream(
            task="Summarize the Wikipedia page for Agentic AI"
        )
        
        # Display streamed output using Console UI
        await Console(result)
    
    # Cleanup
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())