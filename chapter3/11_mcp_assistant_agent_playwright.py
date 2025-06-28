"""
Chapter 3: Agent Communication Patterns
Example 11: MCP Assistant Agent with Playwright

Description:
Demonstrates advanced web automation using an assistant agent integrated with
Playwright through the Model Context Protocol (MCP). Shows how to create an agent
capable of complex web interactions including navigation, link clicking, and
content analysis through browser automation.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- MCP extensions installed (autogen-ext)
- Node.js and npx available
- Playwright MCP server: npm install -g @playwright/mcp

Usage:
```bash
python -m chapter3.11_mcp_assistant_agent_playwright
```

Expected Output:
A comprehensive web automation demonstration where the agent will:
1. Navigate to the specified website (https://ekzhu.com/)
2. Identify and click the first link on the page
3. Analyze the content of the linked page
4. Provide detailed insights about the navigation and content
The agent performs complex browser interactions through Playwright automation.

Key Concepts:
- Playwright browser automation via MCP
- Advanced web navigation and interaction
- Link clicking and page analysis
- Round-robin team coordination
- MCP server session management
- Complex web scraping workflows
- Browser tool integration

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, create_mcp_server_session, mcp_server_tools

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config


async def main() -> None:
    """Main execution function demonstrating MCP Playwright agent automation."""
    # Configuration and setup
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(
        **config,
        parallel_tool_calls=False  # Disable parallel calls for browser automation
    )
    
    # Configure Playwright MCP server parameters
    params = StdioServerParams(
        command="npx",
        args=["@playwright/mcp@latest"],
        read_timeout_seconds=60,      # Extended timeout for browser operations
    )
    
    # Execute with MCP server session context
    async with create_mcp_server_session(params) as session:
        await session.initialize()
        
        # Retrieve available browser automation tools
        tools = await mcp_server_tools(server_params=params, session=session)
        print(f"Available Playwright Tools: {[tool.name for tool in tools]}")

        # Create assistant agent with browser automation capabilities
        agent = AssistantAgent(
            name="WebAutomationAssistant",
            model_client=model_client,
            tools=tools,                      # Enable Playwright automation tools
        )

        # Configure termination condition for the automation task
        termination = TextMentionTermination("TERMINATE")
        
        # Create round-robin team for coordinated web automation
        team = RoundRobinGroupChat(
            participants=[agent], 
            termination_condition=termination
        )
        
        # Execute complex web automation task
        print("=== MCP Playwright Web Automation ===")
        await Console(team.run_stream(
            task="Go to https://ekzhu.com/, visit the first link in the page, then tell me about the linked page."
        ))

    # Cleanup handled by context manager
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
