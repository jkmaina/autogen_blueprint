"""
Chapter 9: Memory and Context Management
Example 7: Simple RAG System

Description:
Demonstrates a complete Retrieval-Augmented Generation (RAG) system
using ChromaDB for document indexing and retrieval. Shows how to build
agents that can answer questions based on indexed documentation.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ with ChromaDB extensions
- ChromaDB package installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter9.07_simple_rag
```

Expected Output:
RAG system demonstration:
1. Creates ChromaDB vector memory for documents
2. Indexes AutoGen documentation chunks
3. Builds RAG-enabled assistant agent
4. Demonstrates question-answering with retrieval
5. Shows context-aware responses
6. Retrieval-augmented generation workflow

Key Concepts:
- Retrieval-Augmented Generation (RAG)
- Document indexing and chunking
- Vector similarity search
- Context-aware response generation
- Knowledge base integration
- RAG agent architecture
- Document retrieval systems

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
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def build_rag_system():
    # Step 1: Create vector memory for documents
    rag_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="autogen_docs",
            persistence_path=os.path.join(str(Path.home()), ".autogen_rag"),
            k=3,  # Return top 3 relevant chunks
            score_threshold=0.3  # Lower threshold for more results
        )
    )
    
    # Clear existing data for fresh start
    await rag_memory.clear()
    
    # Step 2: Index AutoGen documentation
    indexer = SimpleDocumentIndexer(memory=rag_memory, chunk_size=800)
    
    documentation_sources = [
        "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        # Add more documentation URLs as needed
    ]
    
    print("Starting document indexing...")
    total_chunks = await indexer.index_documents(documentation_sources)
    print(f"✅ Indexed {total_chunks} chunks from {len(documentation_sources)} documents")
    
    # Step 3: Create RAG assistant
    rag_assistant = AssistantAgent(
        name="autogen_expert",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        memory=[rag_memory]
    )
    
    # Step 4: Test the RAG system
    questions = [
        "What is AutoGen?",
        "How do I create an agent in AutoGen?",
        "What models does AutoGen support?",
        "How do I install AutoGen?"
    ]
    
    for question in questions:
        print(f"\n{'🤖 ' + '='*58}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        stream = rag_assistant.run_stream(task=question)
        await Console(stream)
    
    # Clean up
    await rag_memory.close()

# Run the RAG system
asyncio.run(build_rag_system())