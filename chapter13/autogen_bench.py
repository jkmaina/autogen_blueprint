"""
AutoGenBench example for AutoGen v0.5

This demonstrates how to use AutoGenBench to benchmark and evaluate agent performance.
"""

import asyncio
import sys
import os
import time
import json
import logging
from typing import Dict, Any, List, Optional

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Import AutoGenBench components (simulated for this example)
class AutoGenBench:
    """Simulated AutoGenBench class for demonstration purposes."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.tasks = []
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def add_task(self, task_id, task_description, expected_output=None):
        """Add a task to the benchmark."""
        self.tasks.append({
            "id": task_id,
            "description": task_description,
            "expected_output": expected_output
        })
        
    async def run_benchmark(self, agent, verbose=True):
        """Run the benchmark on the provided agent."""
        self.start_time = time.time()
        self.results = {}
        
        for task in self.tasks:
            task_id = task["id"]
            task_description = task["description"]
            
            if verbose:
                print(f"\nRunning task {task_id}: {task_description}")
            
            # Measure task execution time
            task_start = time.time()
            response = await agent.run(task=task_description)
            task_end = time.time()
            
            # Store results
            self.results[task_id] = {
                "response": str(response),
                "execution_time": task_end - task_start,
                "expected_output": task.get("expected_output")
            }
            
            if verbose:
                print(f"Task completed in {task_end - task_start:.2f} seconds")
        
        self.end_time = time.time()
        return self.results
    
    def generate_report(self):
        """Generate a benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        total_time = self.end_time - self.start_time
        avg_time = total_time / len(self.tasks)
        
        report = f"=== AutoGenBench Report: {self.name} ===\n"
        report += f"Description: {self.description}\n"
        report += f"Total tasks: {len(self.tasks)}\n"
        report += f"Total execution time: {total_time:.2f} seconds\n"
        report += f"Average task time: {avg_time:.2f} seconds\n\n"
        
        report += "Task Results:\n"
        for task in self.tasks:
            task_id = task["id"]
            result = self.results.get(task_id, {})
            report += f"- Task {task_id}: {task['description']}\n"
            report += f"  Execution time: {result.get('execution_time', 'N/A'):.2f} seconds\n"
            
            # If expected output is provided, we could add evaluation metrics here
            if result.get("expected_output"):
                report += f"  Expected output: {result['expected_output']}\n"
            
        return report
    
    def save_results(self, filename):
        """Save benchmark results to a file."""
        data = {
            "benchmark_name": self.name,
            "description": self.description,
            "tasks": self.tasks,
            "results": {k: {"response": v["response"], "execution_time": v["execution_time"], 
                           "expected_output": v.get("expected_output")} 
                       for k, v in self.results.items()},
            "total_time": self.end_time - self.start_time if self.end_time else None,
            "timestamp": time.time()
        }
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")

from utils.config import get_openai_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_autogenbench_example():
    """Run an example with AutoGenBench to benchmark agent performance."""
    logger.info("Starting AutoGenBench example")
    
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent to benchmark
    assistant = AssistantAgent(
        name="benchmarked_assistant",
        system_message="You are a helpful assistant focused on providing accurate and concise responses.",
        model_client=model_client,
    )
    
    # Create a benchmark
    benchmark = AutoGenBench(
        name="General Knowledge Benchmark",
        description="A benchmark to evaluate the agent's performance on general knowledge questions."
    )
    
    # Add tasks to the benchmark
    benchmark.add_task(
        task_id="math_1",
        task_description="What is the square root of 144?",
        expected_output="12"
    )
    
    benchmark.add_task(
        task_id="history_1",
        task_description="Who was the first president of the United States?",
        expected_output="George Washington"
    )
    
    benchmark.add_task(
        task_id="science_1",
        task_description="What is the chemical symbol for gold?",
        expected_output="Au"
    )
    
    benchmark.add_task(
        task_id="geography_1",
        task_description="What is the capital of Japan?",
        expected_output="Tokyo"
    )
    
    benchmark.add_task(
        task_id="literature_1",
        task_description="Who wrote 'Pride and Prejudice'?",
        expected_output="Jane Austen"
    )
    
    # Run the benchmark
    logger.info("Running benchmark...")
    await benchmark.run_benchmark(assistant)
    
    # Generate and display the report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save the results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    try:
        benchmark.save_results(os.path.join(results_dir, "general_knowledge_benchmark.json"))
    except TypeError as e:
        print(f"Error saving results: {e}")
        print("This is expected in our simulated implementation.")
    
    # Close the model client
    await model_client.close()

async def main():
    """Main function to run the AutoGenBench example."""
    await run_autogenbench_example()

if __name__ == "__main__":
    asyncio.run(main())
