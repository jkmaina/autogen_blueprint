"""
Chapter 15: Testing Frameworks
Example 3: Evaluation Framework

Description:
Demonstrates comprehensive evaluation framework for AutoGen v0.5 agents
including performance metrics tracking, quality assessment, benchmarking
systems, and result visualization for systematic agent evaluation.

Prerequisites:
- Python 3.9+ with matplotlib for visualization
- AutoGen v0.5+ installed
- unittest.mock for test mocking
- Chapter 15 Example 1 (unit testing) for MockLLM
- Understanding of evaluation methodologies

Usage:
```bash
python -m chapter15.03_evaluation_framework
```

Expected Output:
Evaluation framework demonstration:
1. Performance metrics tracking
2. Quality evaluation scoring
3. Agent benchmarking comparison
4. Result visualization generation
5. Data persistence and loading
6. Comprehensive evaluation reports

Key Concepts:
- Performance metrics collection
- Quality assessment frameworks
- Benchmarking methodologies
- Statistical analysis and comparison
- Data visualization techniques
- Result persistence and reporting
- Automated evaluation pipelines
- Agent performance optimization

AutoGen Version: 0.5+
"""

# Standard library imports
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

# Third-party imports
# matplotlib imported conditionally in visualization methods

# AutoGen imports
import autogen
from autogen import AssistantAgent, UserProxyAgent

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the MockLLM from the unit testing example
from chapter15.01_unit_testing import MockLLM

class PerformanceTracker:
    """Track performance metrics for agent evaluations"""
    
    def __init__(self):
        self.metrics = {
            "response_time": [],
            "token_usage": {"input": [], "output": []},
            "conversation_turns": [],
            "api_calls": []
        }
    
    def track_run(self, func):
        """Decorator to track performance metrics of a function"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            self.metrics["response_time"].append(elapsed)
            
            # Additional metric collection would happen here
            # In a real implementation, you would track actual token usage
            
            return result
        return wrapper
    
    def record_conversation(self, chat_history):
        """Record metrics from a conversation"""
        self.metrics["conversation_turns"].append(len(chat_history))
        
        # Simulate token counting
        input_tokens = sum(len(msg["content"]) // 4 for msg in chat_history if msg["role"] == "user")
        output_tokens = sum(len(msg["content"]) // 4 for msg in chat_history if msg["role"] != "user")
        
        self.metrics["token_usage"]["input"].append(input_tokens)
        self.metrics["token_usage"]["output"].append(output_tokens)
    
    def record_api_calls(self, count):
        """Record number of API calls made"""
        self.metrics["api_calls"].append(count)
    
    def get_summary(self):
        """Generate a summary of collected metrics"""
        if not self.metrics["response_time"]:
            return {"error": "No metrics collected"}
        
        return {
            "avg_response_time": sum(self.metrics["response_time"]) / len(self.metrics["response_time"]),
            "total_tokens": {
                "input": sum(self.metrics["token_usage"]["input"]),
                "output": sum(self.metrics["token_usage"]["output"])
            },
            "avg_conversation_turns": sum(self.metrics["conversation_turns"]) / len(self.metrics["conversation_turns"]) if self.metrics["conversation_turns"] else 0,
            "total_api_calls": sum(self.metrics["api_calls"]) if self.metrics["api_calls"] else 0
        }

class QualityEvaluator:
    """Evaluate the quality of agent responses"""
    
    def __init__(self, evaluation_config=None):
        self.evaluation_config = evaluation_config or {}
        self.results = []
    
    def evaluate_response(self, query, response, criteria=None):
        """Evaluate a response based on specified criteria"""
        criteria = criteria or ["relevance", "correctness", "completeness"]
        
        # In a real implementation, this would use an LLM to evaluate
        # For this example, we'll use a simple heuristic
        scores = {}
        
        for criterion in criteria:
            # Simple heuristic scoring (would be replaced with LLM evaluation)
            if criterion == "relevance":
                # Check if response contains keywords from query
                query_words = set(query.lower().split())
                response_words = set(response.lower().split())
                overlap = len(query_words.intersection(response_words))
                scores[criterion] = min(10, overlap * 2)
            
            elif criterion == "correctness":
                # Placeholder for correctness evaluation
                # In a real implementation, this would check factual accuracy
                scores[criterion] = 8  # Placeholder score
            
            elif criterion == "completeness":
                # Simple length-based heuristic
                scores[criterion] = min(10, len(response) / 100)
            
            else:
                scores[criterion] = 5  # Default score for unknown criteria
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores)
        
        result = {
            "query": query,
            "response": response,
            "scores": scores,
            "overall_score": overall_score
        }
        
        self.results.append(result)
        return result
    
    def get_summary(self):
        """Get a summary of all evaluations"""
        if not self.results:
            return {"error": "No evaluations performed"}
        
        # Calculate average scores
        all_criteria = set()
        for result in self.results:
            all_criteria.update(result["scores"].keys())
        
        avg_scores = {criterion: 0 for criterion in all_criteria}
        for result in self.results:
            for criterion in all_criteria:
                if criterion in result["scores"]:
                    avg_scores[criterion] += result["scores"][criterion]
        
        for criterion in avg_scores:
            avg_scores[criterion] /= len(self.results)
        
        return {
            "num_evaluations": len(self.results),
            "avg_scores": avg_scores,
            "avg_overall_score": sum(r["overall_score"] for r in self.results) / len(self.results)
        }

class AgentBenchmark:
    """Benchmark different agent configurations"""
    
    def __init__(self, test_cases, metrics=None):
        self.test_cases = test_cases
        self.metrics = metrics or ["success_rate", "response_time", "token_usage"]
        self.results = {}
    
    def run(self, agent_system, name=None):
        """Run benchmark on an agent system"""
        system_name = name or f"System_{len(self.results) + 1}"
        self.results[system_name] = {metric: [] for metric in self.metrics}
        
        performance_tracker = PerformanceTracker()
        quality_evaluator = QualityEvaluator()
        
        for i, test_case in enumerate(self.test_cases):
            print(f"Running test case {i+1}/{len(self.test_cases)} for {system_name}...")
            
            # Run the test case
            start_time = time.time()
            result = agent_system.process(test_case["input"])
            elapsed = time.time() - start_time
            
            # Evaluate success
            success = self._evaluate_success(result, test_case)
            
            # Record metrics
            self.results[system_name]["success_rate"].append(1 if success else 0)
            self.results[system_name]["response_time"].append(elapsed)
            
            # Record token usage (simulated)
            input_tokens = len(test_case["input"]) // 4
            output_tokens = len(result.get("output", "")) // 4
            self.results[system_name]["token_usage"].append({
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            })
            
            # Evaluate quality if expected output is provided
            if "expected_output" in test_case:
                quality_result = quality_evaluator.evaluate_response(
                    test_case["input"],
                    result.get("output", ""),
                    criteria=["relevance", "correctness", "completeness"]
                )
                
                # Store quality metrics
                if "quality" not in self.results[system_name]:
                    self.results[system_name]["quality"] = []
                self.results[system_name]["quality"].append(quality_result["overall_score"])
        
        return self._summarize_results(system_name)
    
    def _evaluate_success(self, result, test_case):
        """Determine if the test case was successful"""
        if "expected_output" in test_case:
            return test_case["expected_output"] in result.get("output", "")
        elif "validation_func" in test_case:
            return test_case["validation_func"](result)
        return True  # No validation criteria specified
    
    def _summarize_results(self, system_name):
        """Summarize benchmark results for a system"""
        summary = {}
        for metric, values in self.results[system_name].items():
            if metric == "success_rate":
                summary[metric] = sum(values) / len(values) * 100
            elif metric == "response_time":
                summary[metric] = sum(values) / len(values)
            elif metric == "token_usage":
                summary[metric] = {
                    "avg_input": sum(v["input"] for v in values) / len(values),
                    "avg_output": sum(v["output"] for v in values) / len(values),
                    "avg_total": sum(v["total"] for v in values) / len(values),
                    "total_input": sum(v["input"] for v in values),
                    "total_output": sum(v["output"] for v in values),
                    "total": sum(v["total"] for v in values)
                }
            elif metric == "quality":
                summary[metric] = sum(values) / len(values)
        return summary
    
    def compare(self):
        """Compare results across all benchmarked systems"""
        comparison = {}
        for metric in self.metrics:
            comparison[metric] = {}
            for system in self.results:
                summary = self._summarize_results(system)
                if metric in summary:
                    comparison[metric][system] = summary[metric]
        return comparison
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to a file"""
        with open(filename, "w") as f:
            json.dump({
                "results": self.results,
                "summary": {
                    system: self._summarize_results(system)
                    for system in self.results
                }
            }, f, indent=2)
    
    def load_results(self, filename="benchmark_results.json"):
        """Load benchmark results from a file"""
        with open(filename, "r") as f:
            data = json.load(f)
            self.results = data["results"]
    
    def visualize(self, metric=None):
        """Visualize benchmark results"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        metrics_to_plot = [metric] if metric else self.metrics
        
        for metric in metrics_to_plot:
            if metric not in self.metrics:
                print(f"Metric '{metric}' not found in results")
                continue
            
            plt.figure(figsize=(10, 6))
            
            systems = list(self.results.keys())
            
            if metric == "success_rate":
                values = [self._summarize_results(system)[metric] for system in systems]
                plt.bar(systems, values)
                plt.ylabel("Success Rate (%)")
                plt.title("Success Rate Comparison")
            
            elif metric == "response_time":
                values = [self._summarize_results(system)[metric] for system in systems]
                plt.bar(systems, values)
                plt.ylabel("Average Response Time (s)")
                plt.title("Response Time Comparison")
            
            elif metric == "token_usage":
                input_values = [self._summarize_results(system)[metric]["avg_input"] for system in systems]
                output_values = [self._summarize_results(system)[metric]["avg_output"] for system in systems]
                
                x = np.arange(len(systems))
                width = 0.35
                
                plt.bar(x - width/2, input_values, width, label='Input Tokens')
                plt.bar(x + width/2, output_values, width, label='Output Tokens')
                
                plt.xlabel('Systems')
                plt.ylabel('Average Token Usage')
                plt.title('Token Usage Comparison')
                plt.xticks(x, systems)
                plt.legend()
            
            elif metric == "quality":
                if "quality" in self._summarize_results(systems[0]):
                    values = [self._summarize_results(system).get("quality", 0) for system in systems]
                    plt.bar(systems, values)
                    plt.ylabel("Quality Score (0-10)")
                    plt.title("Quality Score Comparison")
            
            plt.tight_layout()
            plt.savefig(f"benchmark_{metric}.png")
            plt.close()
            
            print(f"Saved visualization to benchmark_{metric}.png")

# Simple agent system for demonstration
class SimpleAgentSystem:
    """A simple agent system for benchmarking"""
    
    def __init__(self, llm_config=None, system_name="Default"):
        self.llm_config = llm_config
        self.system_name = system_name
        self.mock_llm = MockLLM(["I'll help with that", "Here's the solution"])
    
    def process(self, input_text):
        """Process an input and return a result"""
        # In a real system, this would use actual agents
        # For this example, we'll simulate the processing
        
        with patch("autogen.agentchat.conversable_agent.get_llm_client", return_value=self.mock_llm):
            agent = AssistantAgent(
                name="assistant",
                llm_config=self.llm_config or {"config_list": [{"model": "mock"}]}
            )
            
            response = agent.generate_reply([
                {"role": "user", "content": input_text}
            ])
        
        # Simulate processing time based on input length
        time.sleep(len(input_text) / 1000)
        
        return {
            "output": response,
            "system_name": self.system_name,
            "input_length": len(input_text),
            "output_length": len(response)
        }

# Main function to demonstrate usage
def main():
    """Main function to demonstrate evaluation framework for AutoGen agents."""
    # Define test cases
    test_cases = [
        {
            "input": "What is the capital of France?",
            "expected_output": "Paris"
        },
        {
            "input": "Write a function to calculate the factorial of a number.",
            "expected_output": "factorial"
        },
        {
            "input": "Explain the concept of machine learning.",
            "expected_output": "learning"
        }
    ]
    
    # Create different agent system configurations
    system1 = SimpleAgentSystem(
        llm_config={"config_list": [{"model": "mock"}]},
        system_name="System A"
    )
    
    system2 = SimpleAgentSystem(
        llm_config={"config_list": [{"model": "mock"}]},
        system_name="System B"
    )
    
    # Create benchmark
    benchmark = AgentBenchmark(test_cases)
    
    # Run benchmark on different systems
    print("Benchmarking System A...")
    system1_results = benchmark.run(system1, "System A")
    print(f"System A results: {system1_results}")
    
    print("\nBenchmarking System B...")
    system2_results = benchmark.run(system2, "System B")
    print(f"System B results: {system2_results}")
    
    # Compare results
    comparison = benchmark.compare()
    print(f"\nComparison: {comparison}")
    
    # Save results
    benchmark.save_results()
    
    # Visualize results
    try:
        benchmark.visualize("success_rate")
        benchmark.visualize("response_time")
        benchmark.visualize("token_usage")
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation framework demo interrupted by user")
    except Exception as e:
        print(f"Error running evaluation framework demo: {e}")
        raise
