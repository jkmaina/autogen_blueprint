"""
Chapter 5: Code Execution and Tool Integration
Example 2: Complete Code Executor with Agent Integration

Description:
Demonstrates a complete code execution workflow integrating an assistant agent
with Docker-based code execution. Shows how to create a data science agent that
can generate, analyze, and visualize cryptocurrency data using Python tools.

Prerequisites:
- OpenAI API key set in .env file
- Docker installed and running
- AutoGen v0.5+ with code execution extensions
- Internet connectivity for pulling Docker images

Usage:
```bash
python -m chapter5.02_complete_code_executor
```

Expected Output:
A comprehensive cryptocurrency analysis demonstration:
1. Agent generates 30 days of sample Bitcoin price data
2. Calculates 5-day moving average using pandas
3. Creates visualization with matplotlib and saves as PNG
4. Provides statistical analysis (mean, median, standard deviation)
5. Delivers insights about price trends and patterns

Key Concepts:
- Agent-integrated code execution
- Docker container isolation
- Data science workflow automation
- Python tool integration
- File generation and persistence
- Statistical analysis automation
- Chart creation and visualization

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config


async def crypto_analysis_demo():
    """Complete cryptocurrency analysis demonstration using agent code execution."""
    work_dir = Path("crypto_demo")
    work_dir.mkdir(exist_ok=True)

    try:
        async with DockerCommandLineCodeExecutor(
            image="jupyter/scipy-notebook:latest", 
            work_dir=work_dir,
            timeout=180,                     # Extended timeout for data processing
            auto_remove=True                 # Clean up containers automatically
        ) as code_executor:
            
            # Configuration and setup
            config = get_openai_config()
            model_client = OpenAIChatCompletionClient(**config)
            
            # Create data scientist agent with code execution capabilities
            agent = AssistantAgent(
                name="DataScientist",
                model_client=model_client,
                tools=[PythonCodeExecutionTool(code_executor)],
                system_message="""
You are an expert data scientist specializing in cryptocurrency analysis.
Always show outputs clearly with print statements.
Generate sample data, compute comprehensive statistics, and create professional visualizations.
Save all charts as PNG files with descriptive names.
Provide clear insights and interpretations of the data.
                """
            )

            # Define comprehensive analysis task
            task = """
Create a comprehensive Bitcoin price analysis:

1. Generate 30 days of realistic sample Bitcoin price data with some volatility
2. Calculate multiple moving averages (5-day, 10-day, 20-day)
3. Compute key statistics: mean, median, standard deviation, min, max
4. Create a professional chart showing:
   - Daily prices as a line plot
   - Moving averages as different colored lines
   - Clear title, labels, and legend
5. Save the chart as 'bitcoin_analysis.png'
6. Provide insights about price trends, volatility, and patterns
7. Calculate the percentage change from start to end of period
            """

            print("=== Cryptocurrency Analysis with Agent Code Execution ===")
            
            # Process agent responses and display output
            async for message in agent.run_stream(task=task):
                if hasattr(message, 'content') and message.content:
                    content = str(message.content)
                    # Extract clean content from message format
                    if "content='" in content:
                        start = content.find("content='") + 9
                        end = content.find("', name=")
                        if start < len(content) and end > start:
                            print(content[start:end])
                    else:
                        print(content.strip())

            # Verify output file creation
            chart_path = work_dir / "bitcoin_analysis.png"
            if chart_path.exists():
                print(f"\n✅ Analysis complete! Chart saved: {chart_path.absolute()}")
                print(f"📊 File size: {chart_path.stat().st_size} bytes")
            else:
                print("\n❌ Chart file was not created successfully")
                
            # Cleanup
            await model_client.close()
                
    except Exception as e:
        print(f"Error during cryptocurrency analysis: {e}")


if __name__ == "__main__":
    asyncio.run(crypto_analysis_demo())
