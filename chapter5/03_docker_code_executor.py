"""
Chapter 5: Code Execution and Tool Integration
Example 3: Docker Code Executor

Description:
Demonstrates executing Python code in isolated Docker containers using AutoGen's
Docker Command Line Code Executor. Shows container-based code execution with
automatic cleanup and error handling capabilities.

Prerequisites:
- Docker installed and running
- AutoGen v0.5+ with Docker code execution extensions
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter5.03_docker_code_executor
```

Expected Output:
Docker container execution demonstration:
1. Creates isolated Python container environment
2. Executes factorial calculation code with error handling
3. Shows Python version and execution results
4. Demonstrates container auto-cleanup
5. Displays execution output and exit codes

Key Concepts:
- Docker container isolation
- Automated container lifecycle management
- Code execution with error handling
- Container auto-removal
- Timeout configuration
- Execution result processing
- Isolated environment benefits

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async def docker_example():
    """Demonstrate Docker-based code execution with error handling."""
    work_dir = Path("docker_coding")
    work_dir.mkdir(exist_ok=True)
    
    try:
        print("=== Docker Container Code Execution ===")
        
        # Use context manager for automatic cleanup
        async with DockerCommandLineCodeExecutor(
            image="python:3.11-slim",
            work_dir=work_dir,
            timeout=30,                    # 30-second timeout
            auto_remove=True               # Automatically remove container
        ) as executor:
            
            # Execute code with potential errors
            code_blocks = [
                CodeBlock(
                    language="python", 
                    code="""
import sys
print(f"Python version: {sys.version}")
print("Calculating factorial...")

def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    return 1 if n <= 1 else n * factorial(n-1)

try:
    result = factorial(5)
    print(f"5! = {result}")
except Exception as e:
    print(f"Error: {e}")
                    """
                )
            ]
            
            result = await executor.execute_code_blocks(
                code_blocks=code_blocks,
                cancellation_token=CancellationToken(),
            )
            
            print("\n=== Docker Execution Results ===")
            print(f"Exit Code: {result.exit_code}")
            print(f"Output:\n{result.output}")
            
            if result.exit_code == 0:
                print("✅ Docker code execution successful!")
            else:
                print("❌ Docker code execution failed!")
            
    except Exception as e:
        print(f"Docker execution error: {e}")
        print("Please verify Docker is running and accessible.")


if __name__ == "__main__":
    asyncio.run(docker_example())