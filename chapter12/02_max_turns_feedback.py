"""
Chapter 12: Human Interaction Patterns
Example 2: Max Turns Feedback

Description:
Demonstrates iterative feedback loops using max_turns parameter control.
Shows how to create bounded conversation sessions where humans can provide
feedback between runs, enabling controlled iterative improvement workflows.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Interactive terminal environment

Usage:
```bash
python -m chapter12.02_max_turns_feedback
```

Expected Output:
Max turns feedback demonstration:
1. Assistant creates initial content with max_turns limit
2. Conversation stops after defined turn count
3. Human provides feedback for next iteration
4. New conversation starts with feedback incorporated
5. Iterative improvement through feedback cycles
6. Controlled conversation session management

Key Concepts:
- Max turns conversation control
- Iterative feedback loops
- Bounded conversation sessions
- Human feedback integration
- Conversation restart patterns
- Turn-based interaction control
- Session-based improvement cycles
- Interactive refinement workflows

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def main() -> None:
    """Main function to demonstrate max turns feedback control patterns."""
    print("\n=== Max Turns Feedback Example ===\n")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent
    assistant = AssistantAgent(
        name="assistant",
        system_message="""You are a creative assistant that writes poetry.
        When asked to write a poem, create a short 4-line poem on the requested topic.
        Incorporate any feedback provided by the user in your next poem.""",
        model_client=model_client,
    )
    
    # Create a group chat with max_turns=1 (stop after assistant responds once)
    group_chat = RoundRobinGroupChat(
        [assistant],
        max_turns=1,  # Stop after 1 turn (assistant's response)
    )
    
    # Initial task
    task = "Write a poem about the mountains."
    
    # Interactive loop for feedback
    while True:
        print(f"\nTask: {task}")
        
        # Run the group chat with the current task
        await Console(group_chat.run_stream(task=task))
        
        # Get feedback from the user
        feedback = input("\nEnter your feedback (or type 'exit' to quit): ")
        
        # Check if the user wants to exit
        if feedback.lower() == "exit":
            break
        
        # Set the next task with the feedback
        task = feedback
    
    # Close the model client connection
    await model_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMax turns feedback demo interrupted by user")
    except Exception as e:
        print(f"Error running max turns feedback demo: {e}")
        raise
