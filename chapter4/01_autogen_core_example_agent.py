"""
Chapter 4: AutoGen Core Fundamentals
Example 1: Basic Core Application

Description:
Demonstrates the fundamental concepts of AutoGen Core including agent registration,
message handling, and runtime management. Shows a producer-modifier-consumer pattern
with three agents communicating through message passing in a single-threaded runtime.

Prerequisites:
- AutoGen Core v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter4.01_autogen_core_example_agent
```

Expected Output:
A sequence showing producer-modifier-consumer message flow:
1. Producer generates numbers 1-5 sequentially
2. Modifier receives each number and multiplies by 2
3. Consumer receives modified numbers and maintains running sum
Final output shows the complete processing chain with sum calculation.

Key Concepts:
- AutoGen Core agent registration
- Message handling with @message_handler
- RoutedAgent and SingleThreadedAgentRuntime
- Custom message types with dataclasses
- Agent-to-agent communication patterns
- Runtime lifecycle management
- Default subscription patterns

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
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)

# Define custom message types
@dataclass
class NumberMessage:
    content: int

@dataclass
class StartMessage:
    """Message to trigger the producer to start producing."""
    pass

# Define a producer agent that generates numbers
@default_subscription
class ProducerAgent(RoutedAgent):
    def __init__(self, start_val: int, max_val: int) -> None:
        super().__init__("A producer agent.")
        self._current_val = start_val
        self._max_val = max_val
    @message_handler
    async def start_producing(self, message: StartMessage, ctx: MessageContext) -> None:
        """Start producing numbers."""
        print(f"{'-'*80}\nProducer:\nStarting production...")
        while self._current_val <= self._max_val:
            print(f"{'-'*80}\nProducer:\nProducing {self._current_val}")
            await self.publish_message(
                NumberMessage(content=self._current_val),
                DefaultTopicId()
            )
            self._current_val += 1
            await asyncio.sleep(1)  # Wait for 1 second between messages

# Define a modifier agent that transforms numbers
@default_subscription
class ModifierAgent(RoutedAgent):
    def __init__(self, modify_val: Callable[[int], int]) -> None:
        super().__init__("A modifier agent.")
        self._modify_val = modify_val

    @message_handler
    async def handle_message(self, message: NumberMessage, ctx: MessageContext) -> None:
        val = self._modify_val(message.content)
        print(f"{'-'*80}\nModifier:\nModified {message.content} to {val}")
        await self.publish_message(
            NumberMessage(content=val),
            DefaultTopicId()
        )

# Define a consumer agent that receives the final numbers
@default_subscription
class ConsumerAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A consumer agent.")
        self._sum = 0

    @message_handler
    async def handle_message(self, message: NumberMessage, ctx: MessageContext) -> None:
        self._sum += message.content
        print(f"{'-'*80}\nConsumer:\nReceived {message.content}, sum is now {self._sum}")

async def main() -> None:
    # Create a runtime
    runtime = SingleThreadedAgentRuntime()
    
    try:
        # Register agents with the runtime
        producer = await ProducerAgent.register(
            runtime,
            "producer",
            lambda: ProducerAgent(start_val=1, max_val=5),
        )
        
        await ModifierAgent.register(
            runtime,
            "modifier",
            # Modify the value by multiplying by 2
            lambda: ModifierAgent(modify_val=lambda x: x * 2),
        )
        
        await ConsumerAgent.register(
            runtime,
            "consumer",
            lambda: ConsumerAgent(),
        )
        
        # Start the runtime
        runtime.start()
        
        # Start the producer by sending a StartMessage
        await runtime.send_message(
            message=StartMessage(),
            recipient=AgentId("producer", "default")
        )
        
        # Wait for all messages to be processed
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