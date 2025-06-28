"""
Chapter 13: Performance Optimization and Deployment
Example 5: Workflow Optimization

Description:
Demonstrates comprehensive workflow optimization techniques including agent
specialization, optimized conversation flows, selective context management,
and performance monitoring with comparative analytics.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- OAI_CONFIG_LIST configuration file

Usage:
```bash
python -m chapter13.05_workflow_optimization
```

Expected Output:
Workflow optimization demonstration:
1. Specialized agent task distribution
2. Optimized conversation flow patterns
3. Selective context management
4. Performance tracking and metrics
5. Inefficient workflow comparison
6. Optimization benefit analysis

Key Concepts:
- Agent specialization strategies
- Optimized conversation flows
- Selective context management
- Performance tracking systems
- Workflow efficiency metrics
- Multi-agent coordination
- Resource optimization patterns
- Comparative performance analysis

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Third-party imports
import autogen
from autogen import AssistantAgent, UserProxyAgent

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Performance metrics tracker
class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            "agent_calls": {},
            "conversation_turns": {},
            "response_times": {},
            "token_usage": {},  # Simulated
        }
    
    def start_timer(self, task_id):
        return datetime.now()
    
    def end_timer(self, task_id, start_time, agent_name):
        end_time = datetime.now()
        elapsed_seconds = (end_time - start_time).total_seconds()
        
        # Record response time
        if agent_name not in self.metrics["response_times"]:
            self.metrics["response_times"][agent_name] = []
        self.metrics["response_times"][agent_name].append(elapsed_seconds)
        
        # Record agent call
        if agent_name not in self.metrics["agent_calls"]:
            self.metrics["agent_calls"][agent_name] = 0
        self.metrics["agent_calls"][agent_name] += 1
        
        return elapsed_seconds
    
    def record_conversation_turns(self, conversation_id, turns):
        self.metrics["conversation_turns"][conversation_id] = turns
    
    def simulate_token_usage(self, agent_name, prompt_length, response_length):
        # Simulate token counting (rough approximation)
        prompt_tokens = prompt_length // 4
        response_tokens = response_length // 4
        
        if agent_name not in self.metrics["token_usage"]:
            self.metrics["token_usage"][agent_name] = {"prompt_tokens": 0, "response_tokens": 0}
        
        self.metrics["token_usage"][agent_name]["prompt_tokens"] += prompt_tokens
        self.metrics["token_usage"][agent_name]["response_tokens"] += response_tokens
    
    def get_summary(self):
        summary = {
            "total_agent_calls": sum(self.metrics["agent_calls"].values()),
            "calls_by_agent": self.metrics["agent_calls"],
            "avg_response_time": {},
            "total_conversation_turns": sum(self.metrics["conversation_turns"].values()),
            "avg_turns_per_conversation": sum(self.metrics["conversation_turns"].values()) / len(self.metrics["conversation_turns"]) if self.metrics["conversation_turns"] else 0,
            "estimated_token_usage": {}
        }
        
        # Calculate average response times
        for agent, times in self.metrics["response_times"].items():
            summary["avg_response_time"][agent] = sum(times) / len(times) if times else 0
        
        # Summarize token usage
        for agent, usage in self.metrics["token_usage"].items():
            summary["estimated_token_usage"][agent] = usage
        
        return summary

# Context manager for selective context
class SelectiveContextManager:
    def __init__(self):
        self.contexts = {}
    
    def add_to_context(self, context_id, content, metadata=None):
        if context_id not in self.contexts:
            self.contexts[context_id] = []
        
        self.contexts[context_id].append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context(self, context_id, max_items=None, filter_func=None):
        if context_id not in self.contexts:
            return []
        
        context = self.contexts[context_id]
        
        # Apply filter if provided
        if filter_func:
            context = [item for item in context if filter_func(item)]
        
        # Limit number of items if specified
        if max_items and max_items > 0:
            context = context[-max_items:]
        
        return context
    
    def get_formatted_context(self, context_id, max_items=None, filter_func=None):
        context = self.get_context(context_id, max_items, filter_func)
        return "\n\n".join([item["content"] for item in context])

async def main():
    """Main function to demonstrate comprehensive workflow optimization techniques."""
    # Initialize performance tracker
    tracker = PerformanceTracker()
    
    # Initialize context manager
    context_manager = SelectiveContextManager()
    
    # Load model configurations
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-3.5-turbo"],
        },
    )
    
    # 1. Demonstrate agent specialization
    print("\n=== Agent Specialization ===")
    
    # Create specialized agents with focused roles
    planner = AssistantAgent(
        name="planner",
        llm_config={"config_list": config_list},
        system_message="You are a planning expert who breaks down complex tasks into clear, actionable steps."
    )
    
    coder = AssistantAgent(
        name="coder",
        llm_config={"config_list": config_list},
        system_message="You are a Python expert who implements code based on specifications."
    )
    
    reviewer = AssistantAgent(
        name="reviewer",
        llm_config={"config_list": config_list},
        system_message="You are a code reviewer who checks for bugs and suggests improvements."
    )
    
    # Create user proxy agent
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )
    
    # 2. Demonstrate optimized conversation flow
    print("\n=== Optimized Conversation Flow ===")
    
    # Function to run an optimized workflow
    async def run_optimized_workflow(task_description):
        conversation_id = f"task_{int(time.time())}"
        total_turns = 0
        
        # Step 1: Planning phase
        plan_start = tracker.start_timer("planning")
        await user_proxy.a_initiate_chat(
            planner,
            message=f"Create a step-by-step plan for: {task_description}. Be concise and clear."
        )
        plan_time = tracker.end_timer("planning", plan_start, "planner")
        
        # Extract the plan
        plan = user_proxy.chat_messages[planner][-1]["content"]
        total_turns += len(user_proxy.chat_messages[planner])
        
        # Add to context
        context_manager.add_to_context(conversation_id, plan, {"type": "plan"})
        
        # Simulate token usage
        tracker.simulate_token_usage("planner", len(task_description), len(plan))
        
        print(f"Planning completed in {plan_time:.2f} seconds")
        
        # Step 2: Implementation phase with selective context
        relevant_context = context_manager.get_formatted_context(conversation_id)
        
        code_start = tracker.start_timer("coding")
        await user_proxy.a_initiate_chat(
            coder,
            message=f"Implement Python code for this plan:\n\n{plan}\n\nProvide complete, working code."
        )
        code_time = tracker.end_timer("coding", code_start, "coder")
        
        # Extract the code
        implementation = user_proxy.chat_messages[coder][-1]["content"]
        total_turns += len(user_proxy.chat_messages[coder])
        
        # Add to context
        context_manager.add_to_context(conversation_id, implementation, {"type": "code"})
        
        # Simulate token usage
        tracker.simulate_token_usage("coder", len(plan), len(implementation))
        
        print(f"Implementation completed in {code_time:.2f} seconds")
        
        # Step 3: Review phase with selective context
        # Only include the code, not the plan
        code_context = context_manager.get_formatted_context(
            conversation_id,
            filter_func=lambda item: item["metadata"].get("type") == "code"
        )
        
        review_start = tracker.start_timer("review")
        await user_proxy.a_initiate_chat(
            reviewer,
            message=f"Review this code and suggest improvements:\n\n{implementation}"
        )
        review_time = tracker.end_timer("review", review_start, "reviewer")
        
        # Extract the review
        review = user_proxy.chat_messages[reviewer][-1]["content"]
        total_turns += len(user_proxy.chat_messages[reviewer])
        
        # Simulate token usage
        tracker.simulate_token_usage("reviewer", len(implementation), len(review))
        
        print(f"Review completed in {review_time:.2f} seconds")
        
        # Record conversation turns
        tracker.record_conversation_turns(conversation_id, total_turns)
        
        return {
            "plan": plan,
            "implementation": implementation,
            "review": review,
            "total_time": plan_time + code_time + review_time,
            "total_turns": total_turns
        }
    
    # Run the optimized workflow
    task = "Create a Python function that calculates the Fibonacci sequence up to n terms"
    result = await run_optimized_workflow(task)
    
    print(f"\nTask completed in {result['total_time']:.2f} seconds with {result['total_turns']} conversation turns")
    
    # 3. Demonstrate inefficient workflow for comparison
    print("\n=== Inefficient Workflow (for comparison) ===")
    
    # Create a general-purpose agent
    generalist = AssistantAgent(
        name="generalist",
        llm_config={"config_list": config_list},
        system_message="You are a helpful assistant who can plan, code, and review."
    )
    
    # Function to run an inefficient workflow
    async def run_inefficient_workflow(task_description):
        conversation_id = f"inefficient_{int(time.time())}"
        
        # Single agent handles everything with multiple turns
        start_time = tracker.start_timer("generalist_task")
        
        # Step 1: Planning
        await user_proxy.a_initiate_chat(
            generalist,
            message=f"Create a plan for: {task_description}"
        )
        
        # Step 2: Ask for implementation
        await user_proxy.a_initiate_chat(
            generalist,
            message="Now implement the code for this plan"
        )
        
        # Step 3: Ask for review
        await user_proxy.a_initiate_chat(
            generalist,
            message="Review your code and suggest improvements"
        )
        
        total_time = tracker.end_timer("generalist_task", start_time, "generalist")
        total_turns = len(user_proxy.chat_messages[generalist])
        
        # Record conversation turns
        tracker.record_conversation_turns(conversation_id, total_turns)
        
        # Simulate token usage (rough estimate)
        combined_length = sum(len(msg["content"]) for msg in user_proxy.chat_messages[generalist])
        tracker.simulate_token_usage("generalist", len(task_description) * 3, combined_length)
        
        return {
            "total_time": total_time,
            "total_turns": total_turns
        }
    
    # Run the inefficient workflow
    inefficient_result = await run_inefficient_workflow(task)
    
    print(f"\nInefficient task completed in {inefficient_result['total_time']:.2f} seconds with {inefficient_result['total_turns']} conversation turns")
    
    # 4. Performance comparison and analytics
    print("\n=== Performance Comparison and Analytics ===")
    
    # Get performance summary
    performance_summary = tracker.get_summary()
    
    # Calculate improvement percentages
    time_improvement = (inefficient_result['total_time'] - result['total_time']) / inefficient_result['total_time'] * 100
    turns_improvement = (inefficient_result['total_turns'] - result['total_turns']) / inefficient_result['total_turns'] * 100
    
    print(f"Time improvement: {time_improvement:.1f}%")
    print(f"Conversation turns reduction: {turns_improvement:.1f}%")
    
    print("\nPerformance metrics:")
    print(json.dumps(performance_summary, indent=2))
    
    # Summary of workflow optimization benefits
    print("\n=== Workflow Optimization Benefits ===")
    print("1. Agent Specialization: Each agent focuses on its expertise")
    print("2. Optimized Conversation Flow: Reduced back-and-forth communication")
    print("3. Selective Context Management: Only relevant information is shared")
    print("4. Performance Monitoring: Identify bottlenecks and optimization opportunities")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWorkflow optimization demo interrupted by user")
    except Exception as e:
        print(f"Error running workflow optimization demo: {e}")
        raise
