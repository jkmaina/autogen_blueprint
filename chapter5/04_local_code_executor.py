"""
Chapter 5: Code Execution and Tool Integration
Example 4: Local Code Executor

Description:
Demonstrates executing Python code locally using AutoGen's Local Command Line
Code Executor. Shows simple, direct code execution on the local system without
containerization for development and testing scenarios.

Prerequisites:
- AutoGen v0.5+ with local code execution support
- Python 3.9+ installed locally
- Write permissions in working directory

Usage:
```bash
python -m chapter5.04_local_code_executor
```

Expected Output:
Local code execution demonstration:
1. Creates local working directory
2. Executes simple Python print statement
3. Shows execution output and exit code
4. Demonstrates fastest execution method
5. Displays "Hello from AutoGen!" message

Key Concepts:
- Local code execution
- Working directory management
- Direct system execution
- Minimal overhead execution
- Development environment testing
- Simple execution workflows
- Local file system interaction

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

async def minimal_example():
    """Minimal example of local code execution."""
    try:
        # Create a working directory
        work_dir = Path("coding")
        work_dir.mkdir(exist_ok=True)
        
        print("=== Local Code Execution ===")
        
        # Initialize the executor
        executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
        
        # Execute simple code
        result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(
                    language="python", 
                    code="print('Hello from AutoGen!')"
                )
            ],
            cancellation_token=CancellationToken(),
        )
        
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
        
        if result.exit_code == 0:
            print("✅ Local code execution successful!")
        else:
            print("❌ Local code execution failed!")
            
    except Exception as e:
        print(f"Error during local code execution: {e}")


if __name__ == "__main__":
    asyncio.run(minimal_example())