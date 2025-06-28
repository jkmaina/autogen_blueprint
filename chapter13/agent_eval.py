"""
AgentEval example for AutoGen v0.5

This demonstrates how to use AgentEval to evaluate agent performance using LLM-based evaluation.
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

# Import AgentEval components (simulated for this example)
class AgentEval:
    """Simulated AgentEval class for demonstration purposes."""
    
    def __init__(self, evaluator_model_client, criteria=None):
        self.evaluator_model_client = evaluator_model_client
        self.criteria = criteria or [
            "Accuracy",
            "Completeness",
            "Relevance",
            "Clarity",
            "Efficiency"
        ]
        self.results = {}
        
    async def evaluate_response(self, task, response, reference=None):
        """Evaluate a response using the evaluator model."""
        evaluation_prompt = self._create_evaluation_prompt(task, response, reference)
        
        # Call the evaluator model
        evaluation = await self._call_evaluator(evaluation_prompt)
        
        return evaluation
    
    def _create_evaluation_prompt(self, task, response, reference=None):
        """Create a prompt for the evaluator model."""
        prompt = f"""
        You are an expert evaluator of AI assistant responses. Please evaluate the following:
        
        TASK: {task}
        
        RESPONSE TO EVALUATE:
        {response}
        """
        
        if reference:
            prompt += f"""
            REFERENCE ANSWER:
            {reference}
            """
            
        prompt += f"""
        Please evaluate the response on the following criteria (score 1-10):
        {', '.join(self.criteria)}
        
        For each criterion, provide:
        1. The score (1-10)
        2. A brief justification for the score
        
        Then provide an overall score (1-10) and a summary of the evaluation.
        
        Format your response as JSON with the following structure:
        {{
            "criteria": {{
                "Criterion1": {{
                    "score": X,
                    "justification": "..."
                }},
                ...
            }},
            "overall": {{
                "score": X,
                "summary": "..."
            }}
        }}
        """
        
        return prompt
    
    async def _call_evaluator(self, prompt):
        """Call the evaluator model with the evaluation prompt."""
        # In a real implementation, this would call the actual model
        # For this simulation, we'll generate a mock evaluation
        
        # Create a simulated evaluation response
        evaluation = {
            "criteria": {},
            "overall": {
                "score": 0,
                "summary": ""
            }
        }
        
        # Generate random scores for each criterion
        import random
        total_score = 0
        
        for criterion in self.criteria:
            score = random.randint(7, 10)  # Simulated scores between 7-10
            total_score += score
            
            evaluation["criteria"][criterion] = {
                "score": score,
                "justification": f"The response demonstrates {criterion.lower()} at a level of {score}/10."
            }
        
        # Calculate overall score
        overall_score = round(total_score / len(self.criteria), 1)
        evaluation["overall"]["score"] = overall_score
        evaluation["overall"]["summary"] = f"The response achieves an overall score of {overall_score}/10, indicating a strong performance across the evaluated criteria."
        
        return evaluation
    
    async def batch_evaluate(self, tasks_and_responses):
        """Evaluate multiple task-response pairs."""
        results = {}
        
        for task_id, item in tasks_and_responses.items():
            task = item["task"]
            response = item["response"]
            reference = item.get("reference")
            
            evaluation = await self.evaluate_response(task, response, reference)
            results[task_id] = evaluation
        
        self.results = results
        return results
    
    def generate_report(self):
        """Generate an evaluation report."""
        if not self.results:
            return "No evaluation results available."
        
        report = "=== AgentEval Evaluation Report ===\n\n"
        
        # Calculate average scores
        criterion_totals = {criterion: 0 for criterion in self.criteria}
        overall_total = 0
        
        for task_id, evaluation in self.results.items():
            for criterion, details in evaluation["criteria"].items():
                criterion_totals[criterion] += details["score"]
            
            overall_total += evaluation["overall"]["score"]
        
        # Add average scores to the report
        report += "Average Scores:\n"
        for criterion, total in criterion_totals.items():
            avg = total / len(self.results)
            report += f"- {criterion}: {avg:.1f}/10\n"
        
        overall_avg = overall_total / len(self.results)
        report += f"\nOverall Average: {overall_avg:.1f}/10\n\n"
        
        # Add individual task evaluations
        report += "Individual Task Evaluations:\n"
        for task_id, evaluation in self.results.items():
            report += f"\nTask ID: {task_id}\n"
            report += f"Overall Score: {evaluation['overall']['score']}/10\n"
            report += f"Summary: {evaluation['overall']['summary']}\n"
            
            report += "Criteria Scores:\n"
            for criterion, details in evaluation["criteria"].items():
                report += f"- {criterion}: {details['score']}/10\n"
        
        return report
    
    def save_results(self, filename):
        """Save evaluation results to a file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Evaluation results saved to {filename}")

from utils.config import get_openai_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_agenteval_example():
    """Run an example with AgentEval to evaluate agent responses."""
    logger.info("Starting AgentEval example")
    
    # Create model clients
    config = get_openai_config()
    agent_model_client = OpenAIChatCompletionClient(**config)
    evaluator_model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent to evaluate
    assistant = AssistantAgent(
        name="evaluated_assistant",
        system_message="You are a helpful assistant focused on providing accurate and comprehensive responses.",
        model_client=agent_model_client,
    )
    
    # Create an evaluator
    evaluator = AgentEval(
        evaluator_model_client=evaluator_model_client,
        criteria=[
            "Accuracy",
            "Completeness",
            "Relevance",
            "Clarity",
            "Conciseness"
        ]
    )
    
    # Define tasks for evaluation
    tasks = {
        "task1": {
            "task": "Explain how photosynthesis works in plants.",
            "reference": "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in the chloroplasts, primarily in the leaves. Plants use chlorophyll to capture light energy, which is then used to convert water and carbon dioxide into glucose and oxygen."
        },
        "task2": {
            "task": "What are the main causes of climate change?",
            "reference": "The main causes of climate change include greenhouse gas emissions from burning fossil fuels, deforestation, industrial processes, and agricultural practices. These activities increase the concentration of greenhouse gases in the atmosphere, leading to global warming."
        },
        "task3": {
            "task": "Describe the water cycle on Earth.",
            "reference": "The water cycle is the continuous movement of water on, above, and below the Earth's surface. It involves processes such as evaporation, transpiration, condensation, precipitation, and runoff. Water evaporates from oceans and land, forms clouds, falls as precipitation, and returns to oceans and land."
        }
    }
    
    # Get responses from the agent
    logger.info("Getting responses from the agent...")
    tasks_and_responses = {}
    
    for task_id, task_info in tasks.items():
        logger.info(f"Processing task: {task_id}")
        response = await assistant.run(task=task_info["task"])
        
        tasks_and_responses[task_id] = {
            "task": task_info["task"],
            "response": response,
            "reference": task_info.get("reference")
        }
    
    # Evaluate the responses
    logger.info("Evaluating responses...")
    await evaluator.batch_evaluate(tasks_and_responses)
    
    # Generate and display the report
    report = evaluator.generate_report()
    print("\n" + report)
    
    # Save the results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    evaluator.save_results(os.path.join(results_dir, "agent_evaluation.json"))
    
    # Close the model clients
    await agent_model_client.close()
    await evaluator_model_client.close()

async def main():
    """Main function to run the AgentEval example."""
    await run_agenteval_example()

if __name__ == "__main__":
    asyncio.run(main())
