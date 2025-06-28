"""
Chapter 9: Memory and Context Management
Example 5: Context Injection

Description:
Demonstrates context injection techniques for agent memory systems.

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

async def context_example():
    memory = ListMemory()
    await memory.add(MemoryContent(
        content="User prefers responses in French.",
        mime_type=MemoryMimeType.TEXT
    ))

    class DummyAgent:
        def __init__(self):
            self.model_context = []
            self.name = "dummy_agent"

    dummy = DummyAgent()
    await memory.update_context(agent=dummy)
    print("Context injected:", dummy.model_context)

asyncio.run(context_example())

async def clear_memory():
    memory = ListMemory()
    await memory.add(MemoryContent(content="Temporary data.", mime_type=MemoryMimeType.TEXT))
    await memory.clear()
    assert len(await memory.query()) == 0

asyncio.run(clear_memory())

from autogen_core.memory import Memory, MemoryContent, MemoryMimeType

class AlwaysHelloMemory(Memory):
    def __init__(self):
        self.hello = MemoryContent(content="Hello from memory!", mime_type=MemoryMimeType.TEXT)

    async def add(self, content):
        pass  # no-op

    async def query(self, *args, **kwargs):
        return [self.hello]

    async def update_context(self, agent, *args, **kwargs):
        agent.model_context = [self.hello.content]

    async def clear(self):
        pass

    async def close(self):
        pass