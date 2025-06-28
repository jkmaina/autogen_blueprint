"""
Chapter 3: Agent Communication Patterns
Example 9: MagenticOne Group Chat

Description:
Demonstrates the MagenticOne framework for autonomous web research and task completion.
Shows how to create a group chat team that can autonomously browse and summarize web content
using the MagenticOne architecture for coordinated agent collaboration.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- MagenticOne extensions installed
- Internet connectivity for web browsing

Usage:
```bash
python -m chapter3.09_magentic_one
```

Expected Output:
A comprehensive news summary from APnews.com generated through autonomous web browsing.
The MagenticOne team will:
1. Navigate to the specified website
2. Extract current news articles
3. Analyze and summarize the content
4. Present a structured summary of today's news

Key Concepts:
- MagenticOne autonomous agent framework
- Web browsing and content extraction
- Autonomous task completion
- Group chat coordination
- Real-time web research
- Structured information synthesis

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config


async def main() -> None:
    """Main execution function demonstrating MagenticOne group chat."""
    # Configuration and setup
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)

    # Create assistant agent for the MagenticOne team
    assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
    )
    
    # Create MagenticOne group chat team for autonomous web research
    team = MagenticOneGroupChat(
        participants=[assistant], 
        model_client=model_client
    )
    
    # Execute autonomous news summarization task
    print("=== MagenticOne Autonomous News Research ===")
    await Console(team.run_stream(
        task="Summarise todays news from APnews.com."
    ))
    
    # Cleanup
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
