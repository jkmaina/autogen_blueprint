"""
Chapter 8: Advanced Orchestration Patterns
Example 1: Advanced Graph Flow

Description:
Demonstrates sophisticated graph-based workflow orchestration with conditional
routing, multi-stage processing, and complex agent coordination patterns.
Shows advanced graph construction with multiple decision points and paths.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter8.01_advanced_graph_flow
```

Expected Output:
Advanced graph workflow demonstration:
1. Planner evaluates task and determines research necessity
2. Conditional routing to researcher or direct to analyst
3. Researcher gathers information if needed
4. Analyst processes data and findings
5. Reporter generates final summary
6. Complex multi-path workflow execution

Key Concepts:
- Multi-stage graph workflows
- Conditional routing logic
- Complex decision points
- Advanced graph construction
- Multi-path execution flows
- Sophisticated agent coordination
- Workflow optimization patterns

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Define all agents
# NOTE: The planner must include the condition string (e.g., 'NEEDS_RESEARCH' or 'SKIP_RESEARCH') in its output to advance the graph.
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message=(
        "You are the planner. Organize the workflow and assign tasks. "
        "To send the researcher to work, include 'NEEDS_RESEARCH' in your message. "
        "If research is not needed, include 'SKIP_RESEARCH'. "
        "Always include the exact condition string for the next step in your output."
    )
)
researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    system_message="You are the researcher. Gather and summarize information."
)
analyst = AssistantAgent(
    name="analyst",
    model_client=model_client,
    system_message="You are the analyst. Analyze data and findings."
)
writer = AssistantAgent(
    name="writer",
    model_client=model_client,
    system_message="You are the writer. Draft and edit the report."
)
reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="You are the reviewer. Review and approve or request revisions."
)
reviser = AssistantAgent(
    name="reviser",
    model_client=model_client,
    system_message="You are the reviser. Address reviewer feedback and revise the report."
)
publisher = AssistantAgent(
    name="publisher",
    model_client=model_client,
    system_message="You are the publisher. Finalize and publish the report."
)

# Build a complex graph with conditional branching
builder = DiGraphBuilder()

# Add nodes
builder.add_node(planner)
builder.add_node(researcher)
builder.add_node(analyst)
builder.add_node(writer)
builder.add_node(reviewer)
builder.add_node(reviser)
builder.add_node(publisher)

# Set entry point
builder.set_entry_point(planner)

# Define the workflow with conditions
builder.add_edge(planner, researcher, condition="NEEDS_RESEARCH")
builder.add_edge(planner, writer, condition="SKIP_RESEARCH")

builder.add_edge(researcher, analyst, condition="DATA_AVAILABLE")
builder.add_edge(researcher, writer, condition="NO_DATA")

builder.add_edge(analyst, writer)

builder.add_edge(writer, reviewer)

builder.add_edge(reviewer, reviser, condition="NEEDS_REVISION")
builder.add_edge(reviewer, publisher, condition="APPROVED")

builder.add_edge(reviser, reviewer)

# Build the graph
graph = builder.build()

# Create the flow
flow = GraphFlow(
    participants=[planner, researcher, analyst, writer, reviewer, reviser, publisher],
    graph=graph,
)

async def main():
    stream = flow.run_stream(task="Produce a comprehensive research report on AI in medicine. If revisions are needed, loop until approved, then publish.")
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
