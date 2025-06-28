"""
Chapter 9: Memory and Context Management
Example 1: Basic List Memory

Description:
Demonstrates the simplest memory implementation using AutoGen's ListMemory.
Shows how to store and retrieve memory content for maintaining context
across agent interactions.

Prerequisites:
- AutoGen v0.5+ with memory extensions
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter9.01_basic_list_memory
```

Expected Output:
Basic memory operations demonstration:
1. Creates ListMemory instance
2. Adds user preference to memory
3. Queries and retrieves stored content
4. Displays memory items with numbering
5. Shows fundamental memory storage patterns

Key Concepts:
- Basic memory implementation
- Memory content creation
- MIME type specification
- Memory querying operations
- Content persistence basics
- Simple memory patterns
- Context storage fundamentals

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType


async def basic_list_memory():
    """Demonstrate basic list memory operations."""
    try:
        print("=== Basic List Memory Demonstration ===")
        
        # Create memory instance
        memory = ListMemory()
        print("Created ListMemory instance")
        
        # Add content to memory
        await memory.add(MemoryContent(
            content="Please always use metric units in your responses.",
            mime_type=MemoryMimeType.TEXT
        ))
        print("Added user preference to memory")
        
        # Query memory contents
        items = await memory.query()
        print(f"\nRetrieved {len(items)} memory items:")
        
        for idx, item in enumerate(items):
            print(f"{idx+1}. {item}")
            
        print("\n✅ Basic list memory demonstration complete!")
        
    except Exception as e:
        print(f"Error in basic list memory demo: {e}")


if __name__ == "__main__":
    asyncio.run(basic_list_memory())