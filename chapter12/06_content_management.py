"""
Chapter 12: Human Interaction Patterns
Example 6: Content Management and Moderation

Description:
Demonstrates advanced content management workflows with automated analysis,
human moderation, and selective routing. Shows how to create sophisticated
content moderation systems with AI-human collaboration patterns.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Interactive terminal environment

Usage:
```bash
python -m chapter12.06_content_management
```

Expected Output:
Content management demonstration:
1. Automated content analysis and classification
2. Confidence-based routing to auto-approval/rejection
3. Human moderation for borderline cases
4. Selector group chat coordination
5. Workflow-based content processing
6. Final approval/rejection decisions

Key Concepts:
- Content moderation workflows
- Automated content analysis
- Confidence-based routing
- Human moderation integration
- Selector group chat patterns
- Multi-stage approval processes
- Workflow coordination systems
- AI-human decision making

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Create a model client
config = get_openai_config()
model_client = OpenAIChatCompletionClient(**config)

# Create a content analyzer agent
analyzer = AssistantAgent(
    name="content_analyzer",
    system_message="""You are a content moderation assistant.
    Your job is to analyze content for potential policy violations related to:
    1. Harmful content
    2. Hate speech
    3. Adult content
    4. Violence
    5. Self-harm
    
    For each piece of content, provide an analysis with:
    - Content summary
    - Potential policy violations (if any)
    - Confidence score (0-100%)
    
    Based on your analysis:
    - If your confidence is above 90% and no violations are found, say "AUTO_APPROVE" and hand off to "auto_approver"
    - If your confidence is above 90% and violations are found, say "AUTO_REJECT" and hand off to "auto_rejector"  
    - If your confidence is below 90%, say "NEEDS_HUMAN_REVIEW" and hand off to "human_moderator"
    
    Always end with requesting a handoff to the appropriate agent.""",
    model_client=model_client,
    handoffs=["auto_approver", "auto_rejector", "human_moderator"],
)

# Create an auto-approver agent
auto_approver = AssistantAgent(
    name="auto_approver",
    system_message="""You are an automatic content approver.
    When content is handed off to you, it means it has been automatically approved.
    Respond with "CONTENT APPROVED" and a brief explanation of why it was approved.
    End your response with "WORKFLOW_COMPLETE" to terminate the workflow.""",
    model_client=model_client,
)

# Create an auto-rejector agent
auto_rejector = AssistantAgent(
    name="auto_rejector",
    system_message="""You are an automatic content rejector.
    When content is handed off to you, it means it has been automatically rejected.
    Respond with "CONTENT REJECTED" and a brief explanation of why it was rejected.
    End your response with "WORKFLOW_COMPLETE" to terminate the workflow.""",
    model_client=model_client,
)

# Create a workflow coordinator for human review cases
human_coordinator = AssistantAgent(
    name="human_coordinator",
    system_message="""You are a workflow coordinator for human-reviewed content.
    When content needs human review, you coordinate the process:
    1. Present the content analysis to the human moderator
    2. Wait for human decision (APPROVE or REJECT)
    3. Based on the human decision, hand off to final_approver or final_rejector
    
    Always ask the human moderator to review the content and provide their decision.""",
    model_client=model_client,
    handoffs=["final_approver", "final_rejector"],
)

# Create a human moderator agent
human_moderator = UserProxyAgent(
    name="human_moderator",
    description="A human moderator who reviews content that needs human judgment. Review the content and the analysis provided. Respond with 'APPROVE' to approve the content, or 'REJECT' to reject it. You can also provide additional comments with your decision."
)

# Create a final approver agent
final_approver = AssistantAgent(
    name="final_approver",
    system_message="""You are the final content approver.
    When content is handed off to you, it means it has been approved by a human moderator.
    Respond with "FINAL APPROVAL: CONTENT APPROVED" and log the decision.
    End your response with "WORKFLOW_COMPLETE" to terminate the workflow.""",
    model_client=model_client,
)

# Create a final rejector agent
final_rejector = AssistantAgent(
    name="final_rejector",
    system_message="""You are the final content rejector.
    When content is handed off to you, it means it has been rejected by a human moderator.
    Respond with "FINAL DECISION: CONTENT REJECTED" and log the decision.
    End your response with "WORKFLOW_COMPLETE" to terminate the workflow.""",
    model_client=model_client,
)

# Create a selector function that determines which agent should act next
def select_speaker(messages):
    """Select the next speaker based on the conversation context."""
    if not messages:
        return analyzer.name

    # Terminate if workflow is complete
    if "WORKFLOW_COMPLETE" in messages[-1].content:
        return None

    last_message = messages[-1].content
    last_speaker = messages[-1].source
    
    # If analyzer just made a decision, route accordingly
    if last_speaker == "content_analyzer":
        if "AUTO_APPROVE" in last_message:
            return auto_approver.name
        elif "AUTO_REJECT" in last_message:
            return auto_rejector.name
        elif "NEEDS_HUMAN_REVIEW" in last_message:
            return human_coordinator.name
    
    # If human coordinator is asking for human input
    if last_speaker == "human_coordinator" and "human moderator" in last_message.lower():
        return human_moderator.name
    
    # If human moderator just responded, coordinator handles the routing
    if last_speaker == "human_moderator":
        return human_coordinator.name
    
    # If coordinator needs to route based on human decision
    if last_speaker == "human_coordinator":
        for msg in reversed(messages):
            if msg.source == "human_moderator":
                if "APPROVE" in msg.content.upper():
                    return final_approver.name
                elif "REJECT" in msg.content.upper():
                    return final_rejector.name
                break
    
    return analyzer.name

# Create a termination condition
termination = TextMentionTermination("WORKFLOW_COMPLETE")

# Create a selector group chat for the moderation workflow
moderation_workflow = SelectorGroupChat(
    [analyzer, auto_approver, auto_rejector, human_coordinator, human_moderator, final_approver, final_rejector],
    model_client=model_client,
    termination_condition=termination,
    selector_func=select_speaker,
)

async def moderate_content(content):
    """Moderate the given content using the moderation workflow."""
    task = f"Please moderate the following content:\n\n{content}"
    stream = moderation_workflow.run_stream(task=task)
    await Console(stream)

async def main():
    """Main function to demonstrate content management workflows."""
    # Example content to moderate
    content1 = "I love puppies and kittens. They bring joy to everyone's life!"
    content2 = "I hate everyone and wish they would all disappear."
    content3 = "The weather today is partly cloudy with a chance of rain."
    
    print("=== Content Moderation Example ===\n")
    
    print("Moderating content 1:")
    await moderate_content(content1)
    
    print("\nModerating content 2:")
    await moderate_content(content2)
    
    print("\nModerating content 3:")
    await moderate_content(content3)
    
    # Close the model client
    await model_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nContent management demo interrupted by user")
    except Exception as e:
        print(f"Error running content management demo: {e}")
        raise