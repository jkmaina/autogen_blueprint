"""
Chapter 5: Code Execution and Tool Integration
Example 1: Azure Code Executor

Description:
Demonstrates executing Python code in Azure Container Apps using AutoGen's Azure
Dynamic Sessions Code Executor. Shows cloud-based code execution with automatic
resource management and secure container environments.

Prerequisites:
- Azure subscription with Container Apps enabled
- Azure credentials configured (Azure CLI or environment variables)
- AutoGen v0.5+ with Azure extensions installed
- azure-identity package installed

Usage:
```bash
python -m chapter5.01_azure_code_executor
```

Expected Output:
Successful execution of Python code in Azure Container Apps environment:
1. Authenticates with Azure using DefaultAzureCredential
2. Creates temporary working directory
3. Executes simple Python print statement in cloud container
4. Returns execution output and exit code
Shows "Hello from Azure Container Apps!" message.

Key Concepts:
- Azure Container Apps Dynamic Sessions
- Cloud-based code execution
- Azure authentication and credentials
- Temporary directory management
- Remote code execution security
- Container-based isolation
- AutoGen Azure integration

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
import tempfile
from pathlib import Path

# Third-party imports
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.azure import ACADynamicSessionsCodeExecutor
from azure.identity import DefaultAzureCredential


async def minimal_azure_example():
    """Minimal working example of Azure Container Apps code execution."""
    
    # NOTE: Update this endpoint to match your Azure Container Apps configuration
    POOL_ENDPOINT = "https://eastus.dynamicsessions.io/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/YOUR_RESOURCE_GROUP/sessionPools/YOUR_SESSION_POOL"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("=== Azure Container Apps Code Execution ===")
            
            # Create executor with Azure credentials
            executor = ACADynamicSessionsCodeExecutor(
                pool_management_endpoint=POOL_ENDPOINT,
                credential=DefaultAzureCredential(),
                work_dir=temp_dir
            )
            
            # Execute simple Python code in Azure container
            result = await executor.execute_code_blocks(
                code_blocks=[
                    CodeBlock(
                        language="python", 
                        code="print('Hello from Azure Container Apps!')"
                    )
                ],
                cancellation_token=CancellationToken()
            )
            
            print(f"Output: {result.output}")
            print(f"Exit Code: {result.exit_code}")
            
            if result.exit_code == 0:
                print("✅ Azure code execution successful!")
            else:
                print("❌ Azure code execution failed!")
                
    except Exception as e:
        print(f"Error during Azure code execution: {e}")
        print("Please verify your Azure credentials and endpoint configuration.")


if __name__ == "__main__":
    asyncio.run(minimal_azure_example())