"""
Chapter 4: AutoGen Core Fundamentals
Example 4: Producer Agent

Description:
Demonstrates a standalone producer agent that generates sequential numbers and
publishes them to the default topic. Shows how to implement message-triggered
production workflows with timing controls and sequential data generation.

Prerequisites:
- AutoGen Core v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter4.04_producer_agent
```

Expected Output:
A producer agent generating sequential numbers from 1 to 5:
1. Receives StartMessage trigger
2. Begins sequential number production with 1-second intervals
3. Publishes NumberMessage for each value (1, 2, 3, 4, 5)
4. Demonstrates timed message publishing workflows

Key Concepts:
- Producer agent patterns
- Sequential data generation
- Message-triggered workflows
- Timed message publishing
- Default topic communication
- Agent lifecycle management
- Asynchronous production loops

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

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
class Producer(RoutedAgent):
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

async def main() -> None:
    # Create a runtime
    runtime = SingleThreadedAgentRuntime()
    
    try:
        # Register the producer agent with the runtime
        await Producer.register(
            runtime,
            "producer",
            lambda: Producer(start_val=1, max_val=5),
        )
        
        # Start the runtime
        runtime.start()
        
        # Trigger the producer's start_producing method
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
