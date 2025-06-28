"""
Chapter 4: AutoGen Core Fundamentals
Example 5: Runtime Basics

Description:
Demonstrates the fundamental runtime operations in AutoGen Core including agent
registration, message sending, and runtime lifecycle management. Shows the
simplest possible agent-message interaction pattern.

Prerequisites:
- AutoGen Core v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter4.05_runtime_example
```

Expected Output:
A simple message handling demonstration:
1. Agent registration with runtime
2. Runtime startup
3. Direct message sending to specific agent
4. Message reception and processing
Shows "Received message: Hello, world!" output.

Key Concepts:
- Runtime initialization and management
- Agent registration patterns
- Direct message sending with AgentId
- Basic message handling workflows
- Runtime lifecycle (start/stop)
- Simple agent-runtime interaction
- Message routing fundamentals

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler


@dataclass
class MyMessage:
    """Simple message containing string content."""
    content: str


class MyAgent(RoutedAgent):
    """Basic agent demonstrating message handling."""
    
    @message_handler
    async def handle_my_message(self, message: MyMessage, ctx: MessageContext) -> None:
        """Handle incoming MyMessage instances."""
        print(f"Received message: {message.content}")


async def main() -> None:
    """Main execution function demonstrating runtime basics."""
    # Create a runtime and register the agent
    runtime = SingleThreadedAgentRuntime()
    
    try:
        await MyAgent.register(
            runtime, 
            "my_agent", 
            lambda: MyAgent("My agent")
        )

        # Start the runtime, send a message and stop the runtime
        runtime.start()
        print("=== Runtime Message Handling ===")
        await runtime.send_message(
            MyMessage("Hello, world!"), 
            recipient=AgentId("my_agent", "default")
        )
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Proper cleanup
        try:
            await runtime.stop()
        except RuntimeError:
            # Runtime may already be stopped
            pass


if __name__ == "__main__":
    asyncio.run(main())
