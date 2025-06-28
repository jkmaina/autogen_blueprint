"""
Chapter 2: Advanced Agent Concepts
Example 2: MCP-enabled Assistant Agent

Description:
Demonstrates how to create an AutoGen assistant agent with web browsing and date
capabilities using the Model Context Protocol (MCP). Shows integration of external
tools and services through MCP for enhanced agent functionality.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed
- MCP packages: `pip install mcp universal-exchange`
- MCP server: `uvx install mcp-server-fetch`

Usage:
```bash
python -m chapter2.02_mcp_assistant_agent
```

Expected Output:
An MCP-enabled agent will:
1. Get the current date using MCP date capabilities
2. Fetch and summarize today's news from AP using web browsing tools
The agent demonstrates seamless integration of external services through MCP.

Key Concepts:
- Model Context Protocol (MCP) integration
- External tool and service connectivity
- Web browsing and data fetching capabilities
- Workbench configuration and management
- Tool reflection and usage optimization

AutoGen Version: 0.5+
"""
# Standard library imports
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Configure MCP servers for different capabilities
# 1. Web fetching server for retrieving news content
fetch_params = StdioServerParams(command="uvx", args=["mcp-server-fetch"])
# 2. Date server that returns current date in YYYY-MM-DD format
date_params = StdioServerParams(command="python", args=["-c", "from datetime import datetime; print(datetime.now().strftime('%Y-%m-%d'))"])

async def main():
    # Initialize OpenAI client with configuration
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config)
    
    # Create MCP workbench with both fetch and date capabilities
    async with McpWorkbench([fetch_params, date_params]) as wb:
        # Initialize the assistant agent with the workbench
        agent = AssistantAgent(
            name="fetcher",
            model_client=client,
            workbench=wb,
            reflect_on_tool_use=True,  # Enable reflection on tool usage for better decision making
        )
        # Run the agent with a task that combines date and news fetching
        result = await agent.run(task="What's today's date and summarize today's news from AP")
        print(result.messages[-1].content)
    
    # Clean up by closing the OpenAI client
    await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 
