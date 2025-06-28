"""
Chapter 11: Concurrent Agents and Distributed Workflows
Example 1: Concurrent Agents

Description:
Demonstrates concurrent agent execution patterns using AutoGen Core with
SingleThreadedAgentRuntime. Shows how multiple agents can process tasks
simultaneously through topic-based messaging and result aggregation.

Prerequisites:
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Understanding of AutoGen Core concepts

Usage:
```bash
python -m chapter11.01_concurrent_agents
```

Expected Output:
Concurrent agents demonstration:
1. Multiple processor agents registered in runtime
2. Tasks published to shared topic
3. All processors receive and process tasks concurrently
4. Result collector aggregates outputs
5. Task distribution statistics displayed

Key Concepts:
- SingleThreadedAgentRuntime setup
- Topic-based message publishing
- Concurrent task processing
- Message handlers and type subscriptions
- Result aggregation patterns
- Agent lifecycle management
- Shared state coordination

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

# Third-party imports
from autogen_core import (
    AgentId,
    MessageContext,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    RoutedAgent,
    message_handler,
    type_subscription,
    DefaultTopicId,
)

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Define message types
@dataclass
class Task:
    """A task message."""
    task_id: str
    description: str

@dataclass
class TaskResult:
    """A task result message."""
    task_id: str
    result: str
    processor: str

# Shared results storage
shared_results: Dict[str, List[TaskResult]] = {}

# Define the results topic
RESULTS_TOPIC_TYPE = "results"
results_topic_id = TopicId(type=RESULTS_TOPIC_TYPE, source="default")

# Define agent classes using type_subscription decorator
@type_subscription(topic_type="tasks")
class ProcessorAgent(RoutedAgent):
    """An agent that processes tasks."""
    
    def __init__(self, description: str) -> None:
        """Initialize the processor agent."""
        super().__init__(description)
    
    @message_handler
    async def handle_task(self, message: Task, ctx: MessageContext) -> None:
        """Process a task."""
        print(f"{self._description} starting task {message.task_id}: {message.description}")
        
        # Simulate processing time with different durations based on agent name
        processing_time = 1.0 + (hash(self._description) % 3) / 2.0
        await asyncio.sleep(processing_time)
        
        # Generate a result
        result = TaskResult(
            task_id=message.task_id,
            result=f"Processed by {self._description} in {processing_time:.1f}s",
            processor=self._description
        )
        
        # Publish the result to the results topic
        await self.publish_message(result, topic_id=results_topic_id)
        
        print(f"{self._description} finished task {message.task_id} in {processing_time:.1f}s")

@type_subscription(topic_type=RESULTS_TOPIC_TYPE)
class ResultCollectorAgent(RoutedAgent):
    """An agent that collects task results."""
    
    def __init__(self, description: str) -> None:
        """Initialize the result collector agent."""
        super().__init__(description)
    
    @message_handler
    async def handle_result(self, message: TaskResult, ctx: MessageContext) -> None:
        """Collect a task result."""
        print(f"Collector received result for task {message.task_id} from {message.processor}")
        
        # Initialize the list for this task if it doesn't exist
        if message.task_id not in shared_results:
            shared_results[message.task_id] = []
        
        # Add the result to the list
        shared_results[message.task_id].append(message)
        
        # Count total results across all tasks
        total_results = sum(len(results) for results in shared_results.values())
        print(f"Current results: {len(shared_results)} unique tasks, {total_results} total results")

async def main() -> None:
    """Main function to demonstrate concurrent agents execution patterns."""
    print("\n=== Simplified Concurrent Agents Example ===\n")
    
    # Create a runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Register processor agents
    await ProcessorAgent.register(
        runtime, 
        "processor_1", 
        lambda: ProcessorAgent("Processor 1")
    )
    await ProcessorAgent.register(
        runtime, 
        "processor_2", 
        lambda: ProcessorAgent("Processor 2")
    )
    await ProcessorAgent.register(
        runtime, 
        "processor_3", 
        lambda: ProcessorAgent("Processor 3")
    )
    
    # Register the result collector
    await ResultCollectorAgent.register(
        runtime, 
        "collector", 
        lambda: ResultCollectorAgent("Result Collector")
    )
    
    # Start the runtime
    runtime.start()
    
    # Create tasks
    tasks = [
        Task(task_id="task-1", description="Process data concurrently"),
        Task(task_id="task-2", description="Another concurrent task"),
        Task(task_id="task-3", description="Third concurrent task"),
        Task(task_id="task-4", description="Fourth concurrent task"),
        Task(task_id="task-5", description="Fifth concurrent task")
    ]
    
    # Define the tasks topic
    tasks_topic = TopicId(type="tasks", source="default")
    
    # Publish tasks to the tasks topic (all processors will receive them)
    print("Publishing tasks to all processors...")
    for task in tasks:
        print(f"Publishing task {task.task_id}")
        await runtime.publish_message(task, topic_id=tasks_topic)
    
    # Wait for all processing to complete
    print("\nWaiting for tasks to complete...")
    await runtime.stop_when_idle()
    
    # Print final results
    print(f"\n=== Final Results ===")
    print(f"Tasks processed: {len(shared_results)} unique tasks")
    print(f"Total results: {sum(len(results) for results in shared_results.values())}")
    
    for task_id, results in sorted(shared_results.items()):
        print(f"\n{task_id}:")
        for result in results:
            print(f"  - {result.result}")
    
    # Show task distribution (each task is processed by all processors)
    print("\n=== Task Distribution ===")
    print("Since all processors subscribe to the same topic, each task is processed by all processors:")
    processor_counts = {}
    for task_id, results in shared_results.items():
        for result in results:
            if result.processor not in processor_counts:
                processor_counts[result.processor] = 0
            processor_counts[result.processor] += 1
    
    for processor, count in sorted(processor_counts.items()):
        print(f"{processor}: {count} tasks")
    
    print("\n=== Example completed ===")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Error running example: {e}")
        raise