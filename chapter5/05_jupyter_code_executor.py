"""
Chapter 5: Code Execution and Tool Integration
Example 5: Jupyter Code Executor

Description:
Demonstrates executing Python code in Jupyter kernel environments using AutoGen's
Jupyter Code Executor. Shows step-by-step data analysis workflow with persistent
state across code blocks, ideal for data science and research applications.

Prerequisites:
- AutoGen v0.5+ with Jupyter extensions
- Jupyter kernel installed (jupyter, ipykernel)
- Python data science libraries (pandas, numpy, matplotlib)
- Working directory write permissions

Usage:
```bash
python -m chapter5.05_jupyter_code_executor
```

Expected Output:
Comprehensive Jupyter-based data analysis:
1. Imports data science libraries (pandas, numpy, matplotlib)
2. Generates synthetic sales data for full year 2024
3. Performs statistical analysis and monthly aggregations
4. Creates professional visualization with trend lines
5. Saves analysis chart as high-quality PNG file

Key Concepts:
- Jupyter kernel integration
- Persistent execution state
- Multi-step data analysis workflows
- Interactive computing environments
- Data visualization pipelines
- Statistical analysis automation
- Professional chart generation

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor

async def jupyter_data_analysis():
    """Comprehensive data analysis using Jupyter kernel execution."""
    work_dir = Path("jupyter_analysis")
    work_dir.mkdir(exist_ok=True)
    
    try:
        print("=== Jupyter Kernel Data Analysis ===")
        
        async with JupyterCodeExecutor(        
            kernel_name="python3",
            timeout=60                     # 60-second timeout per execution
        ) as executor:
            
            # Step 1: Install and import libraries
            setup_code = CodeBlock(
                language="python",
                code="""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("Libraries imported successfully")
                """
            )
            
            # Step 2: Generate sample data
            data_generation = CodeBlock(
                language="python", 
                code="""
# Generate sample sales data
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
sales = np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 300

df = pd.DataFrame({
    'date': dates,
    'sales': np.maximum(sales, 0)  # Ensure non-negative sales
})

print(f"Generated {len(df)} days of sales data")
print(f"Sales range: ${df['sales'].min():.2f} - ${df['sales'].max():.2f}")
print(df.head())
                """
            )
            
            # Step 3: Perform analysis
            analysis_code = CodeBlock(
                language="python",
                code="""
# Calculate monthly statistics
df['month'] = df['date'].dt.month
monthly_stats = df.groupby('month')['sales'].agg(['mean', 'std', 'min', 'max']).round(2)

print("Monthly Sales Statistics:")
print(monthly_stats)

# Find best performing month
best_month = monthly_stats['mean'].idxmax()
best_sales = monthly_stats.loc[best_month, 'mean']
print(f"\\nBest performing month: {best_month} (Average: ${best_sales:.2f})")
                """
            )
            
            # Step 4: Create visualization (SIMPLE FIX)
            plot_code = CodeBlock(
                language="python",
                code="""
            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['sales'], alpha=0.7, linewidth=1)
            plt.plot(df['date'], df.groupby(df['date'].dt.month)['sales'].transform('mean'), 
                    color='red', linewidth=2, label='Monthly Average')
            plt.title('Daily Sales Throughout 2024')
            plt.xlabel('Date')
            plt.ylabel('Sales ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('sales_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("Analysis complete! Chart saved as 'sales_analysis.png'")
                """
            )
            
            # Execute all code blocks sequentially
            code_blocks = [setup_code, data_generation, analysis_code, plot_code]
            
            for i, code_block in enumerate(code_blocks, 1):
                print(f"\n=== Executing Step {i} ===")
                result = await executor.execute_code_blocks(
                    code_blocks=[code_block],
                    cancellation_token=CancellationToken(),
                )
                
                print(f"Step {i} Output:")
                print(result.output)
                
                if result.exit_code != 0:
                    print(f"Step {i} failed with exit code: {result.exit_code}")
                    break
                    
            print("\n=== Jupyter Data Analysis Complete ===")
            
            # Check for output file
            chart_path = Path("sales_analysis.png")
            if chart_path.exists():
                print(f"✅ Chart saved successfully: {chart_path.absolute()}")
                print(f"📊 File size: {chart_path.stat().st_size} bytes")
            else:
                print("❌ Chart file was not created")
            
    except Exception as e:
        print(f"Jupyter execution error: {e}")
        print("Please verify Jupyter kernel is available and libraries are installed.")


if __name__ == "__main__":
    asyncio.run(jupyter_data_analysis())