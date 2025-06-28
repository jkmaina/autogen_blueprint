"""
Chapter 14: Deployment Patterns
Example 1: Docker Deployment

Description:
Demonstrates comprehensive Docker deployment patterns for AutoGen v0.5 agents
including containerization, service creation, FastAPI web service integration,
and production-ready deployment configurations with Docker Compose.

Prerequisites:
- Docker and Docker Compose installed
- OpenAI API key set in .env file
- AutoGen v0.5+ with required dependencies
- Basic understanding of containerization

Usage:
```bash
python -m chapter14.01_docker_deployment
```

Expected Output:
Docker deployment setup demonstration:
1. Dockerfile generation for AutoGen services
2. FastAPI web service container setup
3. Docker Compose configuration creation
4. Environment configuration files
5. Production deployment instructions
6. Service availability verification

Key Concepts:
- Container-based deployment
- FastAPI service integration
- Docker Compose orchestration
- Environment configuration management
- Production-ready containerization
- Service health monitoring
- Multi-container orchestration
- Deployment automation

AutoGen Version: 0.5+
"""

# Standard library imports
import argparse
import os
from typing import Any, Dict

# Third-party imports (None required for this deployment script)

# Local imports (None required for this deployment script)

# Define the Dockerfile content
DOCKERFILE = """
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "agent_service.py"]
"""

# Define the requirements.txt content
REQUIREMENTS = """
autogen-core>=0.5.0
autogen-agentchat>=0.5.0
autogen-ext>=0.5.0
autogen-ext[openai]>=0.5.0
fastapi>=0.95.0
uvicorn>=0.21.0
python-dotenv>=1.0.0
"""

# Define the agent_service.py content
AGENT_SERVICE = """
import os
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="AutoGen Agent Service")

# Store active model clients
model_clients = {}

# Store active agents
agents = {}

# Define request and response models
class AgentRequest(BaseModel):
    agent_id: str
    system_message: str
    model_name: str = "gpt-4o"

class TaskRequest(BaseModel):
    agent_id: str
    task: str

class TaskResponse(BaseModel):
    agent_id: str
    task: str
    response: str

# Initialize the application
@app.on_event("startup")
async def startup_event():
    print("Starting AutoGen Agent Service")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down AutoGen Agent Service")
    for client in model_clients.values():
        await client.close()

# Create a new agent
@app.post("/agents", status_code=201)
async def create_agent(request: AgentRequest):
    if request.agent_id in agents:
        raise HTTPException(status_code=400, detail=f"Agent {request.agent_id} already exists")
    
    try:
        # Create model client
        model_client = OpenAIChatCompletionClient(
            model=request.model_name,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Create agent
        agent = AssistantAgent(
            name=request.agent_id,
            system_message=request.system_message,
            model_client=model_client,
        )
        
        # Store model client and agent
        model_clients[request.agent_id] = model_client
        agents[request.agent_id] = agent
        
        return {"message": f"Agent {request.agent_id} created successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

# Delete an agent
@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    try:
        # Close model client
        await model_clients[agent_id].close()
        
        # Remove agent and model client
        del agents[agent_id]
        del model_clients[agent_id]
        
        return {"message": f"Agent {agent_id} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")

# List all agents
@app.get("/agents")
async def list_agents():
    return {"agents": list(agents.keys())}

# Run a task with an agent
@app.post("/tasks", response_model=TaskResponse)
async def run_task(request: TaskRequest, background_tasks: BackgroundTasks):
    if request.agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
    
    try:
        # Get the agent
        agent = agents[request.agent_id]
        
        # Run the task
        response = await agent.run(task=request.task)
        
        return {
            "agent_id": request.agent_id,
            "task": request.task,
            "response": response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running task: {str(e)}")

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("agent_service:app", host="0.0.0.0", port=port, reload=False)
"""

# Define the docker-compose.yml content
DOCKER_COMPOSE = """
version: '3'

services:
  agent-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
"""

# Define the .env.example content
ENV_EXAMPLE = """
# API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Service Configuration
PORT=8000
"""

def create_deployment_files(output_dir: str) -> None:
    """
    Create the deployment files in the specified directory.
    
    Args:
        output_dir: The directory to create the files in
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the Dockerfile
    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(DOCKERFILE)
    
    # Create the requirements.txt file
    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS)
    
    # Create the agent_service.py file
    with open(os.path.join(output_dir, "agent_service.py"), "w") as f:
        f.write(AGENT_SERVICE)
    
    # Create the docker-compose.yml file
    with open(os.path.join(output_dir, "docker-compose.yml"), "w") as f:
        f.write(DOCKER_COMPOSE)
    
    # Create the .env.example file
    with open(os.path.join(output_dir, ".env.example"), "w") as f:
        f.write(ENV_EXAMPLE)
    
    # Create the logs directory
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    print(f"Deployment files created in {output_dir}")
    print("\nTo deploy the service:")
    print(f"1. cd {output_dir}")
    print("2. cp .env.example .env")
    print("3. Edit .env to add your API keys")
    print("4. docker-compose up -d")
    print("\nThe service will be available at http://localhost:8000")
    print("API documentation will be available at http://localhost:8000/docs")

def main() -> None:
    """Main function to demonstrate Docker deployment pattern creation."""
    parser = argparse.ArgumentParser(description="Create Docker deployment files for AutoGen")
    parser.add_argument("--output", type=str, default="./deployment", help="Output directory for deployment files")
    
    args = parser.parse_args()
    create_deployment_files(args.output)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDocker deployment demo interrupted by user")
    except Exception as e:
        print(f"Error running Docker deployment demo: {e}")
        raise
