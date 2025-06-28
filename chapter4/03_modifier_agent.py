"""
Chapter 4: AutoGen Core Fundamentals
Example 3: Modifier Agent

Description:
Demonstrates a simple modifier agent that transforms incoming messages using a
configurable transformation function. Shows basic message handling, runtime
management, and agent-to-topic communication patterns.

Prerequisites:
- AutoGen Core v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter4.03_modifier_agent
```

Expected Output:
Shows a modifier agent receiving a message with content=5, applying a doubling
transformation (x * 2), and publishing the modified result (10) back to the
default topic. Demonstrates basic message transformation workflows.

Key Concepts:
- Single agent message transformation
- Configurable transformation functions
- Basic runtime message publishing
- Default topic subscription patterns
- Message handling and republishing
- Lambda function configuration
- Simple agent workflow patterns

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Third-party imports
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)


# Define a message type
@dataclass
class Message:
    """Simple message containing integer content."""
    content: int


@default_subscription
class ModifierAgent(RoutedAgent):
    """Agent that modifies incoming messages using a transformation function."""
    
    def __init__(self, modify_val: Callable[[int], int]) -> None:
        super().__init__("A modifier agent.")
        self._modify_val = modify_val

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        """Handle incoming message and apply transformation."""
        val = self._modify_val(message.content)
        print(f"{'-'*80}\nModifier:\nModified {message.content} to {val}")
        await self.publish_message(Message(content=val), DefaultTopicId())


async def main() -> None:
    """Main execution function demonstrating modifier agent."""
    # Create a runtime and register the agent
    runtime = SingleThreadedAgentRuntime()
    
    try:
        # Create a function that doubles the input value
        await ModifierAgent.register(
            runtime, 
            "modifier_agent", 
            lambda: ModifierAgent(lambda x: x * 2)
        )

        # Start the runtime
        runtime.start()
        
        # Publish a message to the default topic that the agent is subscribed to
        print("=== Modifier Agent Message Processing ===")
        await runtime.publish_message(Message(content=5), DefaultTopicId())
        
        # Wait for the message to be processed and then stop the runtime
        await runtime.stop_when_idle()
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Proper cleanup
        try:
            await runtime.stop()
        except RuntimeError:
            # Runtime is not started, nothing to do
            pass


if __name__ == "__main__":
    asyncio.run(main())