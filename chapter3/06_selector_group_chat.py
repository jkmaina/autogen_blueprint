"""
Chapter 3: Agent Communication Patterns
Example 6: Selector Group Chat

Description:
Demonstrates intelligent agent selection based on expertise and context using SelectorGroupChat.
Shows how to create a team of domain experts where the most appropriate agent is automatically
selected to handle specific questions based on their expertise and the query content.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter3.06_selector_group_chat
```

Expected Output:
The system will process multi-domain questions by intelligently selecting the most
appropriate expert agents:
- Math expert for calculations and mathematical concepts
- Science expert for physics, chemistry, and biology questions
- History expert for historical events and analysis
- Tech expert for programming and computer science
- General assistant for coordination and general queries

Key Concepts:
- Intelligent agent selection mechanisms
- Domain expertise specialization
- Multi-domain query handling
- Context-aware agent routing
- Expert consultation workflows
- Preventing repeated speakers

AutoGen Version: 0.5+
"""
# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def expert_consultation_team():
    """Demonstrate expert consultation with intelligent selection"""
    print("🎯 EXPERT CONSULTATION TEAM")
    print("="*50)
    
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config)
    
    try:
        # Create domain experts
        math_expert = AssistantAgent(
            name="math_expert",
            model_client=client,
            description="Mathematics and statistics expert specializing in calculations, probability, and mathematical analysis",
            system_message="""You are a mathematics expert. Handle:
            - Mathematical calculations and proofs
            - Statistical analysis
            - Probability and combinatorics
            - Mathematical modeling
            
            Only respond to mathematics-related questions."""
        )
        
        science_expert = AssistantAgent(
            name="science_expert",
            model_client=client,
            description="Science expert covering physics, chemistry, biology, and scientific principles",
            system_message="""You are a science expert. Handle:
            - Physics principles and phenomena
            - Chemistry concepts and reactions
            - Biology and life sciences
            - Scientific method and research
            
            Only respond to science-related questions."""
        )
        
        history_expert = AssistantAgent(
            name="history_expert",
            model_client=client,
            description="History expert covering world history, historical events, and historical analysis",
            system_message="""You are a history expert. Handle:
            - Historical events and timelines
            - Historical figures and biographies
            - Cultural and social history
            - Historical analysis and interpretation
            
            Only respond to history-related questions."""
        )
        
        tech_expert = AssistantAgent(
            name="tech_expert",
            model_client=client,
            description="Technology expert covering programming, software development, and computer science",
            system_message="""You are a technology expert. Handle:
            - Programming and software development
            - Computer science concepts
            - Technology trends and innovations
            - Technical problem-solving
            
            Only respond to technology-related questions."""
        )
        
        general_assistant = AssistantAgent(
            name="general_assistant",
            model_client=client,
            description="General purpose assistant for questions that don't fit other expert domains",
            system_message="""You are a general assistant. Handle:
            - General knowledge questions
            - Coordination between experts
            - Questions that don't fit specific domains
            - General conversation and guidance"""
        )
        
        # Create selector team
        expert_team = SelectorGroupChat(
            participants=[math_expert, science_expert, history_expert, tech_expert, general_assistant],
            model_client=client,
            termination_condition=MaxMessageTermination(8),
            allow_repeated_speaker=False  # Prevent same agent from speaking consecutively
        )
        
        # Test with multi-domain questions
        complex_questions = [
            "I need help with three things: Calculate the probability of rolling two sixes with dice, explain the physics of how dice work when thrown, and tell me about the history of gambling in ancient Rome.",
            
            "Can you help me understand: What's the mathematical formula for compound interest, how does radioactive decay work scientifically, and what programming language should I use for data analysis?",
            
            "I'm curious about: The mathematical concept of infinity, the historical development of calculus, and how modern computers handle floating-point arithmetic."
        ]
        
        for i, question in enumerate(complex_questions, 1):
            print(f"\n{'='*80}")
            print(f"Multi-Domain Question {i}")
            print('='*80)
            print(f"Question: {question}")
            print('='*80)
            
            await Console(expert_team.run_stream(task=question))
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

async def main():
    """Main execution function."""
    await expert_consultation_team()


if __name__ == "__main__":
    asyncio.run(main())