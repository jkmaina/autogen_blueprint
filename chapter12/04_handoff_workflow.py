"""
Chapter 12: Human Interaction Patterns
Example 4: Handoff Workflow

Description:
Demonstrates structured human-in-the-loop workflows using agent handoffs.
Shows how to create collaborative workflows where agents pass control to
each other, including humans, in organized review and approval processes.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Interactive terminal environment

Usage:
```bash
python -m chapter12.04_handoff_workflow
```

Expected Output:
Handoff workflow demonstration:
1. Content creator generates initial material
2. Creator hands off to human reviewer
3. Human reviewer provides feedback and suggestions
4. Reviewer can hand back to creator for revisions
5. Iterative improvement through handoff cycles
6. Final approval workflow termination

Key Concepts:
- Agent handoff mechanisms
- Structured workflow orchestration
- Human review integration
- Collaborative content creation
- Handoff-based termination
- Multi-stage approval processes
- Interactive workflow control
- Human-AI collaboration patterns

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
from autogen_agentchat.conditions import HandoffTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def main() -> None:
    """Main function to demonstrate structured handoff workflow patterns."""
    print("\n=== Handoff Workflow Example ===\n")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create a content creator agent
    creator = AssistantAgent(
        name="content_creator",
        system_message="""You are a creative content writer.
        Create engaging content based on the given topic.
        After creating content, hand off to the reviewer by saying "HANDOFF TO reviewer".""",
        model_client=model_client,
        handoffs=["reviewer"],  # Specify that this agent can hand off to reviewer
    )
    
    # Create a reviewer agent (human)
    reviewer = UserProxyAgent(
        name="reviewer",
        description="A human reviewer who approves or suggests changes to content",
        input_func=input,  # Use standard input function for human input
        handoffs=["content_creator"],  # Specify that this agent can hand off back to creator
    )
    
    # Create a termination condition based on handoff to a non-existent agent
    # This will terminate when someone tries to hand off to "final_approver"
    termination = HandoffTermination("final_approver")
    
    # Create a group chat
    workflow = RoundRobinGroupChat(
        [creator, reviewer],
        termination_condition=termination,
    )
    
    # Run the workflow with an initial task
    initial_task = """
    Create a short blog post introduction about the benefits of AI in healthcare.
    The reviewer will provide feedback, and you should revise based on that feedback.
    When the reviewer is satisfied, they should hand off to "final_approver" to end the process.
    """
    
    print("\nStarting workflow. The reviewer can hand off to 'final_approver' to end the process.")
    await Console(workflow.run_stream(task=initial_task))
    
    # Close the model client connection
    await model_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nHandoff workflow demo interrupted by user")
    except Exception as e:
        print(f"Error running handoff workflow demo: {e}")
        raise
