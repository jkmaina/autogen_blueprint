
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

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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
