"""
Chapter 1: AutoGen Fundamentals
Example 4: Multi-Agent Conversation

Description:
Demonstrates a simple conversation between two agents — a teacher and a student.
Shows how different agents with distinct personas can interact with each other in
a structured dialogue format.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter1.04_multi_agent_conversation
```

Expected Output:
A conversation between a teacher and student agent discussing supervised and unsupervised learning,
with the student asking questions and the teacher providing explanations. Demonstrates
basic agent-to-agent communication patterns.

Key Concepts:
- Multi-agent conversation patterns
- Agent persona definition and role-playing
- Sequential message exchange
- Agent response generation
- Conversation flow management

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

async def main() -> None:
    # Load config
    config = get_openai_config()

    # Create shared model client
    model_client = OpenAIChatCompletionClient(
        model=config["model"],
        api_key=config["api_key"]
    )

    # Create teacher and student agents with different roles
    teacher = AssistantAgent(
        name="teacher",
        system_message="You are a knowledgeable teacher who explains concepts clearly and concisely.",
        model_client=model_client
    )

    student = AssistantAgent(
        name="student",
        system_message="You are a curious student who asks thoughtful questions to understand concepts better.",
        model_client=model_client,
        model_client_stream=True
    )

    print("🧑‍🏫 Starting conversation between teacher and student...\n")

    # Student starts by asking a question
    student_msg = await student.run(
        task="What is the difference between supervised and unsupervised learning?"
    )
    print(f"🧑‍🎓 Student: {student_msg}\n")

    # Teacher responds
    teacher_msg = await teacher.run(
        task=f"Respond to this question: {student_msg}"
    )
    print(f"🧑‍🏫 Teacher: {teacher_msg}\n")

    # Student asks follow-up question
    student_followup = await student.run(
        task=f"Based on your explanation: {teacher_msg}\nCan you give me an example of when I might use each approach?"
    )
    print(f"🧑‍🎓 Student: {student_followup}\n")

    # Teacher responds to follow-up
    teacher_response = await teacher.run(
        task=f"Respond to this follow-up question: {student_followup}"
    )
    print(f"🧑‍🏫 Teacher: {teacher_response}\n")

    # Cleanup
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
