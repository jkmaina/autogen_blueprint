"""
Chapter 7: Advanced Patterns and Error Handling
Example 14: Minimal Agent

Description:
Demonstrates the simplest possible agent setup with basic configuration.
Shows minimal code requirements for creating and running a research assistant
agent with OpenAI integration.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter7.14_minimal_agent
```

Expected Output:
Minimal agent demonstration:
1. Simple agent creation with basic configuration
2. Research task execution
3. Direct response output
4. Essential agent functionality showcase

Key Concepts:
- Minimal agent configuration
- Basic model client setup
- Simple system message definition
- Essential agent operations
- Streamlined agent creation
- Core functionality demonstration

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent 
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config


async def main():
    """Main execution function demonstrating minimal agent setup."""
    try:
        print("=== Minimal Agent Demonstration ===")
        
        # Create a model client 
        config = get_openai_config()
        model_client = OpenAIChatCompletionClient(**config)

        # Create a research assistant agent 
        research_assistant = AssistantAgent( 
            name="research_assistant", 
            system_message="""You are a research assistant specializing in finding and summarizing academic papers.
Your goal is to help users find relevant information on their research topics.""",
            model_client=model_client,    
        )
        
        # Simple task execution
        response = await research_assistant.run(
            task="Briefly explain what makes a good research methodology."
        )
        
        print("Agent Response:")
        for message in response.messages:
            if hasattr(message, 'content'):
                print(message.content)
        
        # Cleanup
        await model_client.close()
        print("\n✅ Minimal agent demonstration complete!")
        
    except Exception as e:
        print(f"Error in minimal agent demo: {e}")


if __name__ == "__main__":
    asyncio.run(main())

