"""
Chapter 11: Concurrent Agents and Distributed Workflows
Example 5: Sequential Workflow

Description:
Demonstrates sequential workflow patterns where agents process tasks in
a defined order with clear handoffs between stages. Shows a complete
end-to-end workflow with intake, processing, review, approval, and delivery.

Prerequisites:
- AutoGen v0.5+ installed  
- Python 3.9+ with asyncio support
- OpenAI API key (optional for enhanced processing)
- Understanding of AutoGen Core messaging

Usage:
```bash
python -m chapter11.05_sequential_workflow
```

Expected Output:
Sequential workflow demonstration:
1. Task submission to intake stage
2. Progressive processing through defined stages
3. Stage-by-stage result monitoring
4. Workflow state tracking and metadata
5. Complete task lifecycle visibility
6. Final workflow completion summary

Key Concepts:
- Sequential workflow orchestration
- Stage-based task processing
- Agent handoff mechanisms
- Workflow state management
- Task monitoring and tracking
- Metadata propagation
- Multi-stage result aggregation
- Workflow completion patterns

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

# Third-party imports
from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
    type_subscription,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Define workflow message types
@dataclass
class WorkflowTask:
    """A task to be processed in the workflow."""
    task_id: str
    content: str
    metadata: Dict[str, Any]

@dataclass
class WorkflowResult:
    """The result of a workflow stage."""
    task_id: str
    stage: str
    content: str
    status: str
    metadata: Dict[str, Any]

# Define topic types for each workflow stage
INTAKE_TOPIC_TYPE = "workflow.intake"
PROCESSING_TOPIC_TYPE = "workflow.processing"
REVIEW_TOPIC_TYPE = "workflow.review"
APPROVAL_TOPIC_TYPE = "workflow.approval"
DELIVERY_TOPIC_TYPE = "workflow.delivery"
RESULTS_TOPIC_TYPE = "workflow.results"

# Define workflow stages
STAGES = ["intake", "processing", "review", "approval", "delivery"]

# Base class for workflow stage agents
class WorkflowStageAgent(RoutedAgent):
    """Base class for workflow stage agents."""
    
    def __init__(
        self,
        description: str,
        stage: str,
        next_stage_topic: Optional[str] = None
    ) -> None:
        """Initialize the workflow stage agent."""
        super().__init__(description)
        self.stage = stage
        self.next_stage_topic = next_stage_topic
    
    async def process_task(self, task: WorkflowTask, ctx: MessageContext) -> WorkflowResult:
        """Process a task (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement process_task")
    
    async def handle_workflow_task(self, message: WorkflowTask, ctx: MessageContext) -> None:
        """Handle a workflow task."""
        print(f"{self._description} ({self.stage}) processing task {message.task_id}")
        
        # Process the task
        result = await self.process_task(message, ctx)
        
        # Publish the result
        await self.publish_message(
            result, 
            topic_id=TopicId(RESULTS_TOPIC_TYPE, source=self.stage)
        )
        
        # Forward to the next stage if applicable
        if self.next_stage_topic:
            await self.publish_message(
                WorkflowTask(
                    task_id=message.task_id,
                    content=result.content,
                    metadata={
                        **message.metadata, 
                        **result.metadata, 
                        f"{self.stage}_status": result.status
                    }
                ),
                topic_id=TopicId(self.next_stage_topic, source=self.stage)
            )

# Intake Agent
@type_subscription(topic_type=INTAKE_TOPIC_TYPE)
class IntakeAgent(WorkflowStageAgent):
    """Agent for the intake stage of the workflow."""
    
    def __init__(self) -> None:
        """Initialize the intake agent."""
        super().__init__(
            description="Intake Processor",
            stage="intake",
            next_stage_topic=PROCESSING_TOPIC_TYPE
        )
    
    @message_handler
    async def handle_task(self, message: WorkflowTask, ctx: MessageContext) -> None:
        """Handle intake task."""
        await self.handle_workflow_task(message, ctx)
    
    async def process_task(self, task: WorkflowTask, ctx: MessageContext) -> WorkflowResult:
        """Process an intake task."""
        # Validate the task
        is_valid = len(task.content) > 0
        
        # Add intake-specific metadata
        metadata = {
            "intake_timestamp": "now",
            "intake_validation": "passed" if is_valid else "failed"
        }
        
        # Return the result
        return WorkflowResult(
            task_id=task.task_id,
            stage=self.stage,
            content=task.content,
            status="accepted" if is_valid else "rejected",
            metadata=metadata
        )

# Processing Agent
@type_subscription(topic_type=PROCESSING_TOPIC_TYPE)
class ProcessingAgent(WorkflowStageAgent):
    """Agent for the processing stage of the workflow."""
    
    def __init__(self, model_client: Optional[OpenAIChatCompletionClient] = None) -> None:
        """Initialize the processing agent."""
        super().__init__(
            description="Content Processor",
            stage="processing",
            next_stage_topic=REVIEW_TOPIC_TYPE
        )
        self.model_client = model_client
    
    @message_handler
    async def handle_task(self, message: WorkflowTask, ctx: MessageContext) -> None:
        """Handle processing task."""
        await self.handle_workflow_task(message, ctx)
    
    async def process_task(self, task: WorkflowTask, ctx: MessageContext) -> WorkflowResult:
        """Process a task using AI if available."""
        processed_content = task.content
        
        # Process with AI if available
        if self.model_client:
            try:
                response = await self.model_client.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes text."},
                        {"role": "user", "content": f"Process the following text: {task.content}"}
                    ]
                )
                processed_content = response.content or processed_content
            except Exception as e:
                print(f"Error processing with AI: {e}")
        else:
            # Simple processing without AI
            processed_content = f"Processed: {task.content}"
        
        # Add processing-specific metadata
        metadata = {
            "processing_timestamp": "now",
            "processing_method": "ai" if self.model_client else "simple"
        }
        
        # Return the result
        return WorkflowResult(
            task_id=task.task_id,
            stage=self.stage,
            content=processed_content,
            status="completed",
            metadata=metadata
        )

# Review Agent
@type_subscription(topic_type=REVIEW_TOPIC_TYPE)
class ReviewAgent(WorkflowStageAgent):
    """Agent for the review stage of the workflow."""
    
    def __init__(self) -> None:
        """Initialize the review agent."""
        super().__init__(
            description="Content Reviewer",
            stage="review",
            next_stage_topic=APPROVAL_TOPIC_TYPE
        )
    
    @message_handler
    async def handle_task(self, message: WorkflowTask, ctx: MessageContext) -> None:
        """Handle review task."""
        await self.handle_workflow_task(message, ctx)
    
    async def process_task(self, task: WorkflowTask, ctx: MessageContext) -> WorkflowResult:
        """Review a processed task."""
        # Simulate review process
        review_passed = True
        review_comments = "Looks good!"
        
        # Add review-specific metadata
        metadata = {
            "review_timestamp": "now",
            "review_comments": review_comments
        }
        
        # Return the result
        return WorkflowResult(
            task_id=task.task_id,
            stage=self.stage,
            content=task.content,
            status="approved" if review_passed else "rejected",
            metadata=metadata
        )

# Approval Agent
@type_subscription(topic_type=APPROVAL_TOPIC_TYPE)
class ApprovalAgent(WorkflowStageAgent):
    """Agent for the approval stage of the workflow."""
    
    def __init__(self) -> None:
        """Initialize the approval agent."""
        super().__init__(
            description="Final Approver",
            stage="approval",
            next_stage_topic=DELIVERY_TOPIC_TYPE
        )
    
    @message_handler
    async def handle_task(self, message: WorkflowTask, ctx: MessageContext) -> None:
        """Handle approval task."""
        await self.handle_workflow_task(message, ctx)
    
    async def process_task(self, task: WorkflowTask, ctx: MessageContext) -> WorkflowResult:
        """Approve a reviewed task."""
        # Simulate approval process
        is_approved = True
        approval_notes = "Final approval granted."
        
        # Add approval-specific metadata
        metadata = {
            "approval_timestamp": "now",
            "approval_notes": approval_notes
        }
        
        # Return the result
        return WorkflowResult(
            task_id=task.task_id,
            stage=self.stage,
            content=task.content,
            status="approved" if is_approved else "rejected",
            metadata=metadata
        )

# Delivery Agent
@type_subscription(topic_type=DELIVERY_TOPIC_TYPE)
class DeliveryAgent(WorkflowStageAgent):
    """Agent for the delivery stage of the workflow."""
    
    def __init__(self) -> None:
        """Initialize the delivery agent."""
        super().__init__(
            description="Delivery Manager",
            stage="delivery",
            next_stage_topic=None  # No next stage
        )
    
    @message_handler
    async def handle_task(self, message: WorkflowTask, ctx: MessageContext) -> None:
        """Handle delivery task."""
        await self.handle_workflow_task(message, ctx)
    
    async def process_task(self, task: WorkflowTask, ctx: MessageContext) -> WorkflowResult:
        """Deliver the final task."""
        # Simulate delivery process
        delivery_method = "email"
        
        # Add delivery-specific metadata
        metadata = {
            "delivery_timestamp": "now",
            "delivery_method": delivery_method
        }
        
        # Return the result
        return WorkflowResult(
            task_id=task.task_id,
            stage=self.stage,
            content=task.content,
            status="delivered",
            metadata=metadata
        )

# Workflow Monitor
@type_subscription(topic_type=RESULTS_TOPIC_TYPE)
class WorkflowMonitor(RoutedAgent):
    """Agent that monitors the workflow and collects results."""
    
    def __init__(self) -> None:
        """Initialize the workflow monitor."""
        super().__init__("Workflow Monitor")
        self.results: Dict[str, Dict[str, WorkflowResult]] = {}
    
    @message_handler
    async def handle_result(self, message: WorkflowResult, ctx: MessageContext) -> None:
        """Handle a workflow result."""
        task_id = message.task_id
        stage = message.stage
        
        # Initialize task results if needed
        if task_id not in self.results:
            self.results[task_id] = {}
        
        # Store the result
        self.results[task_id][stage] = message
        
        # Print progress
        print(f"Monitor: Task {task_id} completed stage '{stage}' with status '{message.status}'")
        
        # Check if the task has completed all stages
        if stage == STAGES[-1]:
            print(f"\nTask {task_id} has completed the entire workflow!")
            self.print_task_summary(task_id)
    
    def print_task_summary(self, task_id: str) -> None:
        """Print a summary of a task's journey through the workflow."""
        if task_id not in self.results:
            print(f"No results found for task {task_id}")
            return
        
        task_results = self.results[task_id]
        
        print(f"\n{'='*50}")
        print(f"Task {task_id} Summary")
        print(f"{'='*50}")
        for stage in STAGES:
            if stage in task_results:
                result = task_results[stage]
                print(f"\nStage: {stage.upper()}")
                print(f"  Status: {result.status}")
                print(f"  Content: {result.content[:50]}..." if len(result.content) > 50 else f"  Content: {result.content}")
                print(f"  Metadata: {result.metadata}")
            else:
                print(f"\nStage: {stage.upper()} - Not completed")
        print(f"{'='*50}\n")

async def main() -> None:
    """Main function to demonstrate sequential workflow orchestration patterns."""
    print("\n=== Sequential Workflow Example ===\n")
    
    # Create a model client (optional)
    use_ai = False
    model_client = None
    if use_ai:
        config = get_openai_config()
        model_client = OpenAIChatCompletionClient(**config)
    
    # Create a runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Register workflow agents using the class register method
    await IntakeAgent.register(
        runtime, 
        type="intake_agent",
        factory=lambda: IntakeAgent()
    )
    
    await ProcessingAgent.register(
        runtime,
        type="processing_agent", 
        factory=lambda: ProcessingAgent(model_client)
    )
    
    await ReviewAgent.register(
        runtime,
        type="review_agent",
        factory=lambda: ReviewAgent()
    )
    
    await ApprovalAgent.register(
        runtime,
        type="approval_agent", 
        factory=lambda: ApprovalAgent()
    )
    
    await DeliveryAgent.register(
        runtime,
        type="delivery_agent",
        factory=lambda: DeliveryAgent()
    )
    
    await WorkflowMonitor.register(
        runtime,
        type="monitor",
        factory=lambda: WorkflowMonitor()
    )
    
    # Start the runtime
    runtime.start()
    
    # Create and submit tasks
    tasks = [
        WorkflowTask(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            content="This is a sample document that needs to be processed through the workflow.",
            metadata={"priority": "high", "source": "email"}
        ),
        WorkflowTask(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            content="Another document requiring processing with different metadata.",
            metadata={"priority": "medium", "source": "web"}
        )
    ]
    
    # Submit tasks to the intake stage
    for task in tasks:
        print(f"Submitting task {task.task_id} to workflow")
        await runtime.publish_message(
            task, 
            topic_id=TopicId(INTAKE_TOPIC_TYPE, source="main")
        )
    
    # Wait for tasks to complete the workflow
    await runtime.stop_when_idle()
    
    # Close the model client if used
    if model_client:
        await model_client.close()
    
    print("\n=== Workflow completed ===")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSequential workflow demo interrupted by user")
    except Exception as e:
        print(f"Error running sequential workflow demo: {e}")
        raise