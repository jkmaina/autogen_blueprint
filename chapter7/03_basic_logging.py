"""
Chapter 7: Advanced Patterns and Error Handling
Example 3: Basic Logging

Description:
Demonstrates comprehensive logging integration with AutoGen agents.
Shows how to implement structured logging for research workflows,
tracking agent activities, and monitoring task progress.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter7.03_basic_logging
```

Expected Output:
Logged research workflow demonstration:
1. Structured logging initialization with timestamps
2. Research task start logging
3. Agent streaming output with research summary
4. Task completion logging
5. Comprehensive activity tracking

Key Concepts:
- Structured logging configuration
- Agent activity monitoring
- Research workflow tracking
- Task progress logging
- Timestamp and level formatting
- Logger namespace management
- Debug and info level usage

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import logging
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("research_assistant")

async def main():
    """Main execution function demonstrating comprehensive logging."""
    topic = "ai agents"
    
    try:
        print("=== Research Assistant with Structured Logging ===")
        
        # Initialize components with logging
        logger.info("Initializing research assistant")
        config = get_openai_config()
        model_client = OpenAIChatCompletionClient(**config)
        
        agent = AssistantAgent(
            name="research_assistant",
            model_client=model_client,
            system_message="You are a research assistant specializing in AI and machine learning.",
            model_client_stream=True
        )
        
        # Log the start of the research task
        logger.info(f"Starting research on topic: {topic}")
        logger.debug(f"Agent configuration: streaming enabled, model: {config.get('model', 'default')}")
        
        # Run the agent with streaming output
        stream = agent.run_stream(
            task=f"Provide a comprehensive summary of the latest research on {topic}, including key developments and trends."
        )
        await Console(stream)
        
        # Log the completion of the research task
        logger.info(f"Research on '{topic}' completed successfully")
        
        # Cleanup with logging
        await model_client.close()
        logger.info("Resources cleaned up")
        
        print("\n✅ Logged research workflow complete!")
        
    except Exception as e:
        logger.error(f"Error during research workflow: {e}")
        print(f"Research error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
