"""
Chapter 8: Advanced Orchestration Patterns
Example 3: Model-Based Selector

Description:
Demonstrates intelligent agent selection using model-based decision making.
Shows how to create specialized expert agents and use AI-driven selection
logic to route tasks to the most appropriate agent based on content analysis.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter8.03_model_based_selector
```

Expected Output:
Model-based selection demonstration:
1. Multiple specialized expert agents available
2. AI selector analyzes incoming tasks
3. Intelligent routing to most appropriate expert
4. Specialized responses based on expertise
5. Adaptive agent selection patterns
6. Expertise-driven task distribution

Key Concepts:
- AI-driven agent selection
- Specialized expert agents
- Content-based routing
- Intelligent task distribution
- Expertise matching algorithms
- Dynamic agent selection
- Model-based decision making

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Create a model client for the selector
selector_model = OpenAIChatCompletionClient(model="gpt-4o")

# Define four agents
data_expert = AssistantAgent(
    name="data_expert",
    description="Expert in data analysis and statistics.",
    model_client=selector_model,
    system_message="You are a data expert.You specialize in data analysis and statistics."
)
code_expert = AssistantAgent(
    name="code_expert",
    description="Expert in programming and algorithms.",
    model_client=selector_model,
    system_message="You are a code expert. You specialize in programming and algorithms."
)
research_expert = AssistantAgent(
    name="research_expert",
    description="Expert in research and literature review.",
    model_client=selector_model,
    system_message="You are a research expert. You specialize in research and literature review."
)
writing_expert = AssistantAgent(
    name="writing_expert",
    description="Expert in writing and content creation.",
    model_client=selector_model,
    system_message="You are a writing expert.You specialize in writing and content creation."
)
planner = AssistantAgent(
    name="planner",
    description="Expert in task planning and organization.",
    model_client=selector_model,
    system_message="You are a planner. You specialize in task planning and organization. Start by listing all available agaents and assigning tasks."
)

# Termination condition: end after 8 messages
termination = MaxMessageTermination(8)

# Improved selector prompt using variables
# {participants}: list of agent names
# {roles}: newline-separated agent name and description
# {history}: conversation history
selector_prompt = """Select an agent to perform the appropriate task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent. Respond with ONLY the agent's name (e.g., data_expert).
If the task is complete, respond with 'NONE'.

"""

# Create a SelectorGroupChat with a model-based selector
team = SelectorGroupChat(
    [data_expert, code_expert, research_expert, writing_expert, planner],
    model_client=selector_model,
    selector_prompt=selector_prompt,
    termination_condition=termination,
)

async def main():
    stream = team.run_stream(task="Write a short report on the impact of AI in healthcare with analytics and sample code.")
    await Console(stream)
    await selector_model.close()

if __name__ == "__main__":
    asyncio.run(main())
