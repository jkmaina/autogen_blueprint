"""
Chapter 3: Agent Communication Patterns
Example 7: Basic Swarm Agent

Description:
Demonstrates the Swarm pattern with agent handoffs in a sequential chain.
Shows how agents can pass control to each other in a predetermined sequence,
creating a multilingual greeting workflow with translation capabilities.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter3.07_basic_swarm_agent
```

Expected Output:
A sequential agent handoff chain demonstrating multilingual translations:
1. Agent A: Spanish greeting (¡Hola, mundo!)
2. Agent B: English translation
3. Agent C: French translation
4. Agent D: German translation
5. Agent E: Summary of all translations
Each agent hands off control to the next in the predefined sequence.

Key Concepts:
- Swarm pattern implementation
- Agent handoff mechanisms
- Sequential workflow control
- Multilingual processing chain
- Termination condition handling
- Fixed handoff sequences

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config


async def main() -> None:
    """Main execution function demonstrating swarm agent handoffs."""
    # Configuration and setup
    config = get_openai_config()
    model = OpenAIChatCompletionClient(**config)

    # Create sequential handoff chain of translation agents
    agent_a = AssistantAgent(
        name="agent_a",
        model_client=model,
        handoffs=["agent_b"],                    # Next agent in chain
        system_message=(
            "You are Agent A, a Spanish greeter. "
            "Greet with '¡Hola, mundo!' and hand off to agent_b."
        ),
    )

    agent_b = AssistantAgent(
        name="agent_b",
        model_client=model,
        handoffs=["agent_c"],
        system_message=(
            "You are Agent B, an English translator. "
            "Translate the last greeting to English, then hand off to agent_c."
        ),
    )

    agent_c = AssistantAgent(
        name="agent_c",
        model_client=model,
        handoffs=["agent_d"],
        system_message=(
            "You are Agent C, a French translator. "
            "Translate the latest greeting to French, then hand off to agent_d."
        ),
    )

    agent_d = AssistantAgent(
        name="agent_d",
        model_client=model,
        handoffs=["agent_e"],
        system_message=(
            "You are Agent D, a German translator. "
            "Translate the latest greeting to German, then hand off to agent_e."
        ),
    )

    agent_e = AssistantAgent(
        name="agent_e",
        model_client=model,
        handoffs=[],                             # End of chain
        system_message=(
            "You are Agent E, the summariser. "
            "Gather all previous translations and print them on separate lines. "
            "End with the word TERMINATE."
        ),
    )

    # Create swarm team with sequential handoff pattern
    team = Swarm(
        participants=[agent_a, agent_b, agent_c, agent_d, agent_e],
        termination_condition=TextMentionTermination("TERMINATE"),
    )

    # Execute multilingual greeting workflow
    await Console(team.run_stream(
        task="Start the multilingual greeting sequence."
    ))

    # Cleanup
    await model.close()


if __name__ == "__main__":
    asyncio.run(main())
