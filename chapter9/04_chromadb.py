"""
Chapter 9: Memory and Context Management
Example 4: ChromaDB Vector Memory

Description:
Demonstrates advanced vector-based memory using ChromaDB for persistent
storage and similarity search. Shows how to build knowledge bases with
semantic search capabilities for agent memory systems.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ with ChromaDB extensions
- ChromaDB package installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter9.04_chromadb
```

Expected Output:
ChromaDB vector memory demonstration:
1. Creates persistent ChromaDB vector memory
2. Populates knowledge base with AI/ML information
3. Demonstrates semantic similarity search
4. Shows agent integration with vector memory
5. Retrieves contextually relevant information
6. Persistent storage across sessions

Key Concepts:
- Vector-based memory storage
- Semantic similarity search
- Persistent ChromaDB integration
- Knowledge base construction
- Context retrieval systems
- Agent memory integration
- Similarity threshold configuration

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import os
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def chromadb_memory_example():
    # Create ChromaDB memory with persistence
    vector_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="knowledge_base",
            persistence_path=os.path.join(str(Path.home()), ".autogen_chromadb"),
            k=3,  # Return top 3 most relevant results
            score_threshold=0.4  # Minimum similarity score
        )
    )
    
    # Clear any existing data for this example
    await vector_memory.clear()
    
    # Add knowledge to the vector database
    knowledge_items = [
        {
            "content": "Python is excellent for data science, machine learning, and rapid prototyping. It has libraries like pandas, numpy, and scikit-learn.",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "content": "JavaScript is the language of the web. It's used for frontend development with React, Vue, Angular, and backend with Node.js.",
            "metadata": {"topic": "programming", "language": "javascript"}
        },
        {
            "content": "REST APIs should use proper HTTP methods: GET for reading, POST for creating, PUT for updating, DELETE for removing resources.",
            "metadata": {"topic": "api_design", "type": "best_practice"}
        },
        {
            "content": "Microservices architecture breaks applications into small, independent services that communicate over well-defined APIs.",
            "metadata": {"topic": "architecture", "type": "pattern"}
        },
        {
            "content": "Git best practices: use descriptive commit messages, create feature branches, and always review code before merging to main.",
            "metadata": {"topic": "version_control", "tool": "git"}
        }
    ]
    
    # Add all knowledge items to vector memory
    for item in knowledge_items:
        await vector_memory.add(MemoryContent(
            content=item["content"],
            mime_type=MemoryMimeType.TEXT,
            metadata=item["metadata"]
        ))
    
    print(f"Added {len(knowledge_items)} items to vector memory")
    
    # Create agent with vector memory
    knowledge_assistant = AssistantAgent(
        name="knowledge_assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        memory=[vector_memory]
    )
    
    # Test questions that should retrieve relevant knowledge
    questions = [
        "What programming language is good for machine learning?",
        "How should I structure my API endpoints?",
        "Tell me about microservices",
        "What are some Git best practices?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        stream = knowledge_assistant.run_stream(task=question)
        await Console(stream)
    
    # Clean up
    await vector_memory.close()

asyncio.run(chromadb_memory_example())