"""
Chapter 12: Human Interaction Patterns
Example 5: Comprehensive Human Interaction

Description:
Demonstrates comprehensive human-in-the-loop integration patterns across
multiple workflow scenarios including basic interaction, approval workflows,
selective human input, and feedback learning loops.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Interactive terminal environment

Usage:
```bash
python -m chapter12.05_human_interaction
```

Expected Output:
Comprehensive human interaction demonstration:
1. Basic human-AI conversation patterns
2. Approval workflow with content review
3. Selective human input based on triggers
4. Human feedback learning loops
5. Multiple interaction paradigms
6. Adaptive human engagement strategies

Key Concepts:
- Multiple human interaction paradigms
- Basic conversational interfaces
- Approval and review workflows
- Conditional human engagement
- Feedback learning systems
- Human-AI collaboration patterns
- Interactive workflow orchestration
- Adaptive interaction strategies

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Third-party imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import Handoff
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def basic_human_interaction() -> None:
    """Demonstrate basic human interaction with UserProxyAgent."""
    print("\n=== Basic Human Interaction ===\n")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent
    assistant = AssistantAgent(
        name="assistant",
        system_message="""You are a helpful assistant that helps users plan trips.
        Ask questions to understand their preferences and provide tailored recommendations.""",
        model_client=model_client,
    )
    
    # Create a UserProxyAgent with human input
    user_proxy = UserProxyAgent(
        name="user",
        description="A user planning a trip",
        input_func=input,  # Use standard input function for human input
    )
    
    # Create a termination condition
    termination = TextMentionTermination("TERMINATE")
    
    # Create a group chat
    group_chat = RoundRobinGroupChat(
        [assistant, user_proxy],
        termination_condition=termination,
    )
    
    # Run the group chat with an initial task
    initial_task = "I want to plan a vacation for next month."
    await Console(group_chat.run_stream(task=initial_task))
    
    # Close the model client connection
    await model_client.close()

async def approval_workflow() -> None:
    """Demonstrate an approval workflow with human feedback."""
    print("\n=== Approval Workflow ===\n")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create a content creator agent
    creator = AssistantAgent(
        name="content_creator",
        system_message="""You are a creative content writer.
        Create engaging content based on the given topic.
        After creating content, ask for approval from the reviewer.""",
        model_client=model_client,
        handoffs=["reviewer"],  # Handoff to reviewer
    )
    
    # Create a reviewer agent (human)
    reviewer = UserProxyAgent(
        name="reviewer",
        description="A human reviewer who approves or suggests changes to content",
        input_func=input,  # Use standard input function for human input
        handoffs=["content_creator"],  # Handoff back to content creator
    )
    
    # Create a termination condition
    termination = TextMentionTermination("APPROVED")
    
    # Create a group chat
    approval_workflow = RoundRobinGroupChat(
        [creator, reviewer],
        termination_condition=termination,
    )
    
    # Run the workflow with an initial task
    initial_task = "Create a short blog post about the benefits of AI in healthcare."
    await Console(approval_workflow.run_stream(task=initial_task))
    
    # Close the model client connection
    await model_client.close()

async def selective_human_input() -> None:
    """Demonstrate selective human input based on specific conditions."""
    print("\n=== Selective Human Input ===\n")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AI researcher agent
    researcher = AssistantAgent(
        name="researcher",
        system_message="""You are an AI researcher assistant.
        You help with research tasks but consult the human expert for complex decisions.
        When you need human input, explicitly ask for it by saying "NEED_HUMAN_INPUT".""",
        model_client=model_client,
    )
    
    # Create a human expert agent with conditional input
    class ConditionalHumanExpert(UserProxyAgent):
        """A user proxy that only asks for human input when specifically requested."""
        
        async def on_messages(self, messages, cancellation_token):
            """Override to conditionally request human input."""
            last_message = messages[-1]
            
            # Check if the last message contains the trigger phrase
            if "NEED_HUMAN_INPUT" in last_message.content:
                print("\n[System] Human input requested. Please provide your expertise.\n")
                return await super().on_messages(messages, cancellation_token)
            else:
                # Auto-reply with acknowledgment if no human input needed
                return self.create_response("I acknowledge your update. Continue with your work.")
    
    # Create the conditional human expert
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

async def human_feedback_loop() -> None:
    """Demonstrate a feedback loop where AI learns from human feedback."""
    print("\n=== Human Feedback Loop ===\n")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create a learning assistant agent
    learning_assistant = AssistantAgent(
        name="learning_assistant",
        system_message="""You are an assistant that learns from human feedback.
        After each response, ask if the human is satisfied with your answer.
        If they provide feedback, acknowledge it and incorporate it into your future responses.
        Remember the feedback you receive to improve over time.""",
        model_client=model_client,
    )
    
    # Create a human teacher agent
    human_teacher = UserProxyAgent(
        name="human_teacher",
        description="A human providing feedback to help the assistant learn",
        input_func=input,  # Use standard input function for human input
    )
    
    # Create a termination condition
    termination = TextMentionTermination("END_LEARNING_SESSION")
    
    # Create a group chat
    feedback_loop = RoundRobinGroupChat(
        [learning_assistant, human_teacher],
        termination_condition=termination,
    )
    
    # Run the feedback loop with an initial task
    initial_task = """
    Let's have a learning session about climate change.
    I'll provide feedback on your responses to help you improve.
    When we're done, I'll say END_LEARNING_SESSION.
    """
    await Console(feedback_loop.run_stream(task=initial_task))
    
    # Close the model client connection
    await model_client.close()

async def main() -> None:
    """Main function to demonstrate comprehensive human-in-the-loop integration."""
    # Run the examples
    await basic_human_interaction()
    await approval_workflow()
    await selective_human_input()
    await human_feedback_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nHuman interaction demo interrupted by user")
    except Exception as e:
        print(f"Error running human interaction demo: {e}")
        raise
