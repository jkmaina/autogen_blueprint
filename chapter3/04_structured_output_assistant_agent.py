"""
Chapter 3: Agent Communication Patterns
Example 4: Structured Output Assistant Agent

Description:
Demonstrates how to create an AutoGen assistant agent that returns structured JSON
responses using Pydantic models instead of free-form text. Shows sentiment analysis
with consistent, typed output format for reliable data processing.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed
- pydantic package for structured output

Usage:
```bash
python -m chapter3.04_structured_output_assistant_agent
```

Expected Output:
The agent will analyze sentiment and return structured JSON:
{
  "thoughts": "The user seems positive.",
  "response": "happy"
}
Demonstrates typed, predictable agent responses for integration scenarios.

Key Concepts:
- Structured output with Pydantic models
- Type-safe agent responses
- Sentiment analysis implementation
- JSON schema enforcement
- Reliable data format for downstream processing
- Agent output content type configuration

AutoGen Version: 0.5+
"""
# Standard library imports
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]

async def main():
    """Main execution function demonstrating structured output."""
    # Configuration and setup
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config)
    
    # Create agent with structured output type
    agent = AssistantAgent(
        name="classifier",                    # Agent identifier
        model_client=client,                  # OpenAI client instance
        output_content_type=AgentResponse     # Pydantic model for structured output
    )
    
    # Execute sentiment analysis task
    result = await agent.run(task="Analyze: 'I love this!'")
    response = result.messages[-1].content
    
    # Display structured JSON output
    output = {
        "thoughts": response.thoughts,
        "response": response.response
    }
    print(json.dumps(output, indent=2))
    
    # Cleanup
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())