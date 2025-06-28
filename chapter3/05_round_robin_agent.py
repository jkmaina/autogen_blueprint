"""
Chapter 3: Agent Communication Patterns
Example 5: Round Robin Group Chat

Description:
Demonstrates round-robin group chat where agents take turns speaking in a predefined order.
Shows collaborative workflow between a writer, editor, and critic working together on
creative content with structured turn-based communication.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter3.05_round_robin_agent
```

Expected Output:
A collaborative writing session where:
1. Writer creates initial content
2. Editor reviews and refines the writing
3. Critic provides constructive feedback
The cycle continues for 6 messages total, demonstrating structured team collaboration.

Key Concepts:
- Round-robin group chat mechanics
- Turn-based agent communication
- Collaborative content creation
- Agent role specialization (writer/editor/critic)
- Termination conditions and message limits
- Streaming group chat responses

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def round_robin_example():
    """Demonstrates round-robin group chat with specialized agents."""
    print("🔄 ROUND ROBIN: Agents take turns")
    print("=" * 40)
    
    # Configuration and setup
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config)
    
    # Create specialized agents with distinct roles
    writer = AssistantAgent(
        name="writer",
        model_client=client,
        description="Creative writer",
        system_message="You are a creative writer who crafts engaging content.",
        model_client_stream=True
    )
    
    editor = AssistantAgent(
        name="editor", 
        model_client=client,
        description="Critical editor",
        system_message="You are a critical editor. Review and refine the writer's work.",
        model_client_stream=True
    )
    
    critic = AssistantAgent(
        name="critic",
        model_client=client, 
        description="Constructive critic",
        system_message="You are a constructive critic. Provide thoughtful feedback to improve the work.",
        model_client_stream=True
    )
    
    # Create round robin team with ordered communication
    team = RoundRobinGroupChat(
        participants=[writer, editor, critic],           # Speaking order
        termination_condition=MaxMessageTermination(6)   # Stop after 6 messages
    )
    
    # Execute collaborative writing task
    await Console(team.run_stream(
        task="Write a single paragraph short story about AutoGen AI"
    ))
    
    # Cleanup
    await client.close()


async def main():
    """Main execution function."""
    await round_robin_example()

if __name__ == "__main__":
    asyncio.run(main())