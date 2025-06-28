"""
Chapter 8: Advanced Orchestration Patterns
Example 4: Multiple Selectors

Description:
Demonstrates advanced selector patterns with multiple selection layers
and hierarchical agent coordination. Shows how to create complex selection
systems with specialized coordinators and expert agents.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter8.04_multiple_selectors
```

Expected Output:
Multiple selector coordination demonstration:
1. Hierarchical selector system activation
2. Coordinator manages overall task flow
3. Specialized experts handle domain-specific tasks
4. Multi-layer selection and routing
5. Complex coordination patterns
6. Hierarchical task distribution

Key Concepts:
- Multi-layer selector systems
- Hierarchical agent coordination
- Specialized coordinator roles
- Expert agent specialization
- Complex selection hierarchies
- Advanced routing patterns
- Multi-tier orchestration

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Define agents
model_client = OpenAIChatCompletionClient(model="gpt-4o")

coordinator = AssistantAgent(
    name="coordinator",
    model_client=model_client,
    system_message="You are the coordinator. Assign tasks and keep the team on track."
)
data_expert = AssistantAgent(
    name="data_expert",
    model_client=model_client,
    system_message="You are a data expert. Specialize in data analysis and statistics."
)
code_expert = AssistantAgent(
    name="code_expert",
    model_client=model_client,
    system_message="You are a code expert. Specialize in programming and algorithms."
)
research_expert = AssistantAgent(
    name="research_expert",
    model_client=model_client,
    system_message="You are a research expert. Specialize in research and literature review."
)
writing_expert = AssistantAgent(
    name="writing_expert",
    model_client=model_client,
    system_message="You are a writing expert. Specialize in writing and content creation."
)
emergency_handler = AssistantAgent(
    name="emergency_handler",
    model_client=model_client,
    system_message="You handle emergencies and urgent situations."
)
user_proxy = UserProxyAgent(
    name="user_proxy",    
)


def combined_selector(messages):
    """
    Combined selector that uses both rule-based and priority-based selection.
    """
    # First, check for special conditions using the rule-based selector
    rule_result = rule_based_selector(messages)
    if rule_result:
        return rule_result
    
    # If no special conditions, use priority-based selection
    return priority_based_selector(messages)

def rule_based_selector(messages):
    """Rule-based selector for special conditions."""
    if not messages:
        return "coordinator"
    
    # Check for termination
    if "TASK COMPLETE" in messages[-1].content:
        return None
    
    # Check for specific triggers
    content = messages[-1].content.lower()
    if "emergency" in content or "urgent" in content:
        return "emergency_handler"
    
    if "user input needed" in content:
        return "user_proxy"
    
    return None

def priority_based_selector(messages):
    """Priority-based selector for normal operation."""
    if not messages:
        return "coordinator"
    
    # Define agent priorities (lower number = higher priority)
    priorities = {
        "coordinator": 1,
        "data_expert": 2,
        "code_expert": 2,
        "research_expert": 2,
        "writing_expert": 2,
        "user_proxy": 0,  # Highest priority
        "emergency_handler": 0
    }
    
    # Get agents who haven't spoken recently (last 3 messages)
    recent_speakers = [msg.source for msg in messages[-3:] if hasattr(msg, "source")]
    available_agents = [agent for agent in priorities.keys() if agent not in recent_speakers]
    
    # If all agents have spoken recently, allow the coordinator to speak
    if not available_agents:
        return "coordinator"
    
    # Return the highest priority available agent
    return min(available_agents, key=lambda a: priorities.get(a, 999))

# --- End selectors ---

termination = MaxMessageTermination(12)

team = SelectorGroupChat(
    [coordinator, data_expert, code_expert, research_expert, writing_expert, emergency_handler, user_proxy],
    selector_func=combined_selector,
    model_client=model_client,
    termination_condition=termination,
)

async def main():
    stream = team.run_stream(task="Write a collaborative report on AI safety. If an emergency arises, handle it. If user input is needed, request it. End with 'TASK COMPLETE'.")
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
