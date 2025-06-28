"""
Chapter 12: Human Interaction Patterns
Example 1: Basic User Proxy Agent

Description:
Demonstrates the fundamental human-in-the-loop integration using UserProxyAgent.
Shows how to create interactive conversations where humans can provide input,
feedback, and control the conversation flow through direct interaction.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Interactive terminal environment

Usage:
```bash
python -m chapter12.01_basic_user_proxy
```

Expected Output:
Basic user proxy demonstration:
1. Assistant agent creates poetry content
2. User proxy collects human input and feedback
3. Interactive conversation flow with human participation
4. Round-robin chat coordination
5. Termination based on user approval
6. Real-time human-AI collaboration

Key Concepts:
- UserProxyAgent for human integration
- Interactive input collection
- Human feedback loops
- Conversation termination control
- Round-robin group chat patterns
- Human-AI collaboration workflows
- Real-time interaction patterns
- Input function customization

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def main() -> None:
    """Main function to demonstrate basic UserProxyAgent integration patterns."""
    print("\n=== Basic UserProxyAgent Example ===\n")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent
    assistant = AssistantAgent(
        name="assistant",
        system_message="""You are a creative assistant that writes poetry.
        When asked to write a poem, create a short 4-line poem on the requested topic.
        After writing the poem, ask the user if they like it or want any changes.""",
        model_client=model_client,
    )
    
    # Create a UserProxyAgent with human input
    user_proxy = UserProxyAgent(
        name="user_proxy",
        description="A user who requests and reviews poems",
        input_func=input,  # Use standard input function for human input
    )
    
    # Create a termination condition
    termination = TextMentionTermination("APPROVED")
    
    # Create a group chat
    group_chat = RoundRobinGroupChat(
        [assistant, user_proxy],
        termination_condition=termination,
    )
    
    # Run the group chat with an initial task
    initial_task = "Write a poem about the mountains."
    print("\nStarting conversation. Type 'APPROVED' when you're satisfied with the poem.")
    await Console(group_chat.run_stream(task=initial_task))
    
    # Close the model client connection
    await model_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBasic user proxy demo interrupted by user")
    except Exception as e:
        print(f"Error running basic user proxy demo: {e}")
        raise
