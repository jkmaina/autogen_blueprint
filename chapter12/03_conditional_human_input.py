"""
Chapter 12: Human Interaction Patterns
Example 3: Conditional Human Input

Description:
Demonstrates selective human input based on specific trigger conditions.
Shows how to create intelligent human-in-the-loop systems that only request
human input when needed, using conditional logic and trigger phrases.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Interactive terminal environment

Usage:
```bash
python -m chapter12.03_conditional_human_input
```

Expected Output:
Conditional human input demonstration:
1. AI researcher works autonomously on simple tasks
2. Automatic acknowledgment for routine updates
3. Human input requested only for complex decisions
4. Trigger phrase detection and conditional routing
5. Expert consultation integration
6. Research workflow completion

Key Concepts:
- Conditional human input patterns
- Trigger phrase detection
- Selective human engagement
- Custom UserProxyAgent behavior
- Autonomous vs human-guided workflows
- Expert consultation patterns
- Conditional message routing
- Intelligent escalation systems

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
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

class ConditionalHumanExpert(UserProxyAgent):
    """A user proxy that only asks for human input when specifically requested."""
    
    async def on_messages(self, messages, cancellation_token=None):
        """Override to conditionally request human input."""
        last_message = messages[-1]
        
        # Check if the last message contains the trigger phrase
        if "NEED_HUMAN_INPUT" in last_message.content:
            print("\n[System] Human input requested. Please provide your expertise.\n")
            return await super().on_messages(messages, cancellation_token)
        else:
            # Auto-reply with acknowledgment if no human input needed
            print("\n[System] No human input needed. Auto-acknowledging...\n")
            return self.create_response("I acknowledge your update. Continue with your work.")

async def main() -> None:
    """Main function to demonstrate conditional human input patterns."""
    print("\n=== Conditional Human Input Example ===\n")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AI researcher agent
    researcher = AssistantAgent(
        name="researcher",
        system_message="""You are an AI researcher assistant.
        You help with research tasks but consult the human expert for complex decisions.
        When you need human input, explicitly ask for it by saying "NEED_HUMAN_INPUT".
        Otherwise, proceed with your analysis independently.
        
        For this example:
        1. First, provide a brief introduction to your research topic
        2. Then, ask a simple question that doesn't require human input
        3. Next, pose a complex question and request human input with "NEED_HUMAN_INPUT"
        4. Finally, incorporate the human's response and conclude your analysis
        
        End your final message with "RESEARCH_COMPLETE" when you're done.""",
        model_client=model_client,
    )
    
    # Create a human expert agent with conditional input
    human_expert = ConditionalHumanExpert(
        name="human_expert",
        description="A human expert who provides input on complex research questions",
        input_func=input,  # Use standard input function for human input
    )
    
    # Create a termination condition
    termination = TextMentionTermination("RESEARCH_COMPLETE")
    
    # Create a group chat
    research_team = RoundRobinGroupChat(
        [researcher, human_expert],
        termination_condition=termination,
    )
    
    # Run the workflow with a research task
    research_task = """
    Conduct a brief analysis of quantum computing applications in cryptography.
    Start with basic concepts, then move to more complex topics where you'll need expert input.
    When you've completed the research, end with RESEARCH_COMPLETE.
    """
    await Console(research_team.run_stream(task=research_task))
    
    # Close the model client connection
    await model_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nConditional human input demo interrupted by user")
    except Exception as e:
        print(f"Error running conditional human input demo: {e}")
        raise
