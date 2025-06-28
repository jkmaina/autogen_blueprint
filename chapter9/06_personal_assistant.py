import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def personal_assistant_example():
    # Create memory for different types of preferences
    personal_memory = ListMemory(name="personal_assistant_memory")
    
    # Add various user preferences
    preferences = [
        "User prefers concise, bullet-point answers",
        "User works in software development",
        "User is based in San Francisco timezone (PST)",
        "User prefers Python over JavaScript",
        "User has meetings typically scheduled for mornings",
        "User drinks coffee, not tea"
    ]
    
    for pref in preferences:
        await personal_memory.add(MemoryContent(
            content=pref,
            mime_type=MemoryMimeType.TEXT
        ))
    
    # Create the assistant
    assistant = AssistantAgent(
        name="personal_assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        memory=[personal_memory]
    )
    
    # Test different types of questions
    questions = [
        "What programming language should I use for my new project?",
        "What time should I schedule my next team meeting?", 
        "Can you explain API design? Keep it brief.",
        "Should I have coffee or tea while coding?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        stream = assistant.run_stream(task=question)
        await Console(stream)

asyncio.run(personal_assistant_example())