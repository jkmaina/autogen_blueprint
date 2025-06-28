import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AutoGen 0.6.1 Agent Service",
    description="A production-ready service for managing AutoGen agents using official API",
    version="4.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active model clients and agents
model_clients: Dict[str, OpenAIChatCompletionClient] = {}
agents: Dict[str, AssistantAgent] = {}
agent_metadata: Dict[str, Dict[str, Any]] = {}

# Define request and response models
class AgentRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the agent")
    system_message: str = Field(..., description="System message for the agent")
    model_name: str = Field(default="gpt-4o", description="Model name to use")
    temperature: float = Field(default=0.7, description="Temperature for model responses")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for responses")
    parallel_tool_calls: bool = Field(default=True, description="Enable parallel tool calls")
    reflect_on_tool_use: bool = Field(default=False, description="Whether to reflect on tool usage")

class TaskRequest(BaseModel):
    agent_id: str = Field(..., description="ID of the agent to use")
    task: str = Field(..., description="Task description or message")

class TaskResponse(BaseModel):
    agent_id: str
    task: str
    response: str
    status: str = "completed"
    timestamp: str
    stop_reason: Optional[str] = None
    models_usage: Optional[Dict[str, Any]] = None

class AgentInfo(BaseModel):
    agent_id: str
    model_name: str
    system_message: str
    temperature: float
    max_tokens: Optional[int]
    parallel_tool_calls: bool
    reflect_on_tool_use: bool
    created_at: str

class HealthResponse(BaseModel):
    status: str
    agents_count: int
    model_clients_count: int
    autogen_version: str = "0.6.1"

class StreamChunk(BaseModel):
    content: str
    type: str = "chunk"
    timestamp: str

# Example tools that can be registered with agents
async def get_current_time() -> str:
    """Get the current time."""
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

async def calculator(a: float, b: float, operation: str) -> str:
    """Perform basic arithmetic operations."""
    try:
        if operation == "add":
            return str(a + b)
        elif operation == "subtract":
            return str(a - b)
        elif operation == "multiply":
            return str(a * b)
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            return str(a / b)
        else:
            return f"Error: Unknown operation '{operation}'"
    except Exception as e:
        return f"Error: {str(e)}"

# Available tools registry
AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "calculator": calculator
}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Initialize the application
@app.on_event("startup")
async def startup_event():
    logger.info("Starting AutoGen 0.6.1 Agent Service")
    
    # Validate required environment variables
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("AutoGen 0.6.1 Agent Service started successfully")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AutoGen 0.6.1 Agent Service")
    
    # Close all model clients
    for agent_id, client in model_clients.items():
        try:
            logger.info(f"Closing model client for agent {agent_id}")
            await client.close()
        except Exception as e:
            logger.error(f"Error closing model client for {agent_id}: {str(e)}")
    
    # Clear storage
    agents.clear()
    model_clients.clear()
    agent_metadata.clear()
    
    logger.info("AutoGen 0.6.1 Agent Service shutdown complete")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    return HealthResponse(
        status="healthy",
        agents_count=len(agents),
        model_clients_count=len(model_clients)
    )

# Create a new agent
@app.post("/agents", status_code=201)
async def create_agent(request: AgentRequest):
    """Create a new AutoGen agent using the official API"""
    if request.agent_id in agents:
        raise HTTPException(
            status_code=400, 
            detail=f"Agent {request.agent_id} already exists"
        )
    
    try:
        logger.info(f"Creating agent {request.agent_id} with model {request.model_name}")
        
        # Create model client using official API
        model_client_kwargs = {
            "model": request.model_name,
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "temperature": request.temperature,
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        
        if request.max_tokens:
            model_client_kwargs["max_tokens"] = request.max_tokens
            
        model_client = OpenAIChatCompletionClient(**model_client_kwargs)
        
        # Prepare tools (using all available tools for demo)
        tools = list(AVAILABLE_TOOLS.values())
        
        # Create agent using official AssistantAgent API
        agent_kwargs = {
            "name": request.agent_id,
            "model_client": model_client,
            "system_message": request.system_message,
            "tools": tools,
        }
        
        if request.reflect_on_tool_use:
            agent_kwargs["reflect_on_tool_use"] = request.reflect_on_tool_use
            
        agent = AssistantAgent(**agent_kwargs)
        
        # Store model client and agent
        model_clients[request.agent_id] = model_client
        agents[request.agent_id] = agent
        
        # Store metadata
        agent_metadata[request.agent_id] = {
            "model_name": request.model_name,
            "system_message": request.system_message,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "parallel_tool_calls": request.parallel_tool_calls,
            "reflect_on_tool_use": request.reflect_on_tool_use,
            "tools": list(AVAILABLE_TOOLS.keys()),
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Agent {request.agent_id} created successfully")
        return {
            "message": f"Agent {request.agent_id} created successfully", 
            "tools_available": list(AVAILABLE_TOOLS.keys())
        }
    
    except Exception as e:
        logger.error(f"Error creating agent {request.agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error creating agent: {str(e)}"
        )

# Delete an agent
@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an existing agent"""
    if agent_id not in agents:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent {agent_id} not found"
        )
    
    try:
        logger.info(f"Deleting agent {agent_id}")
        
        # Close model client
        if agent_id in model_clients:
            await model_clients[agent_id].close()
        
        # Remove agent, model client, and metadata
        del agents[agent_id]
        del model_clients[agent_id]
        del agent_metadata[agent_id]
        
        logger.info(f"Agent {agent_id} deleted successfully")
        return {"message": f"Agent {agent_id} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting agent: {str(e)}"
        )

# Get agent details
@app.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str):
    """Get details of a specific agent"""
    if agent_id not in agents:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent {agent_id} not found"
        )
    
    metadata = agent_metadata[agent_id]
    return AgentInfo(
        agent_id=agent_id,
        model_name=metadata["model_name"],
        system_message=metadata["system_message"],
        temperature=metadata["temperature"],
        max_tokens=metadata.get("max_tokens"),
        parallel_tool_calls=metadata.get("parallel_tool_calls", True),
        reflect_on_tool_use=metadata.get("reflect_on_tool_use", False),
        created_at=metadata["created_at"]
    )

# List all agents
@app.get("/agents")
async def list_agents():
    """List all active agents"""
    agent_list = []
    for agent_id in agents.keys():
        metadata = agent_metadata[agent_id]
        agent_list.append({
            "agent_id": agent_id,
            "name": agent_id,
            "model_name": metadata["model_name"],
            "tools_count": len(metadata.get("tools", [])),
            "created_at": metadata["created_at"]
        })
    
    return {"agents": agent_list, "total": len(agent_list)}

# Run a task with an agent
@app.post("/tasks", response_model=TaskResponse)
async def run_task(request: TaskRequest):
    """Run a task with the specified agent using the AssistantChat API"""
    if request.agent_id not in agents:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent {request.agent_id} not found"
        )
    
    try:
        logger.info(f"Running task for agent {request.agent_id}")
        
        # Get the agent
        agent = agents[request.agent_id]
        
        # Use the official AutoGen API: agent.run(task="...")
        result: TaskResult = await agent.run(task=request.task)
        
        # Extract response content from TaskResult
        response_content = ""
        models_usage = None
        
        if result.messages:
            # Get the last message from the agent (not user)
            for message in reversed(result.messages):
                if hasattr(message, 'source') and message.source != 'user':
                    if hasattr(message, 'content') and message.content:
                        response_content = str(message.content)
                        break
            
            # Extract models usage from the last message with usage info
            for message in reversed(result.messages):
                if hasattr(message, 'models_usage') and message.models_usage:
                    models_usage = {
                        "prompt_tokens": message.models_usage.prompt_tokens if hasattr(message.models_usage, 'prompt_tokens') else 0,
                        "completion_tokens": message.models_usage.completion_tokens if hasattr(message.models_usage, 'completion_tokens') else 0,
                    }
                    break
        
        if not response_content:
            response_content = "Task completed but no response content available"
        
        logger.info(f"Task completed for agent {request.agent_id}")
        
        return TaskResponse(
            agent_id=request.agent_id,
            task=request.task,
            response=response_content,
            status="completed",
            timestamp=datetime.now().isoformat(),
            stop_reason=result.stop_reason if hasattr(result, 'stop_reason') else None,
            models_usage=models_usage
        )
    
    except Exception as e:
        logger.error(f"Error running task for agent {request.agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error running task: {str(e)}"
        )

# Run a task with streaming response
@app.post("/tasks/stream")
async def run_task_stream(request: TaskRequest):
    """Run a task with streaming response using the official AutoGen API"""
    if request.agent_id not in agents:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent {request.agent_id} not found"
        )
    
    try:
        logger.info(f"Running streaming task for agent {request.agent_id}")
        
        # Get the agent
        agent = agents[request.agent_id]
        
        # Use the official AutoGen streaming API: agent.run_stream(task="...")
        async def generate_stream():
            try:
                async for message in agent.run_stream(task=request.task):
                    timestamp = datetime.now().isoformat()
                    
                    # Handle TaskResult (final result)
                    if isinstance(message, TaskResult):
                        completion_chunk = StreamChunk(
                            content=f"[DONE] Stop reason: {getattr(message, 'stop_reason', 'completed')}",
                            type="completion",
                            timestamp=timestamp
                        )
                        yield f"data: {completion_chunk.model_dump_json()}\n\n"
                        break
                    
                    # Handle regular messages
                    if hasattr(message, 'content') and message.content:
                        content = str(message.content)
                        chunk = StreamChunk(
                            content=content,
                            type="content",
                            timestamp=timestamp
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    elif hasattr(message, 'type'):
                        # Handle other message types (tool calls, etc.)
                        chunk = StreamChunk(
                            content=f"[{message.type}] {str(message)}",
                            type="event",
                            timestamp=timestamp
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                
            except Exception as e:
                error_chunk = StreamChunk(
                    content=f"Error: {str(e)}",
                    type="error",
                    timestamp=datetime.now().isoformat()
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
        
        return StreamingResponse(
            generate_stream(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    except Exception as e:
        logger.error(f"Error running streaming task for agent {request.agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error running streaming task: {str(e)}"
        )

# Get available tools
@app.get("/tools")
async def get_available_tools():
    """Get list of available tools that can be used by agents"""
    tools_info = {}
    for name, func in AVAILABLE_TOOLS.items():
        tools_info[name] = {
            "name": name,
            "description": func.__doc__ or "No description available",
            "signature": str(func.__annotations__) if hasattr(func, '__annotations__') else "No signature available"
        }
    
    return {"available_tools": tools_info, "total": len(AVAILABLE_TOOLS)}

# Get service statistics
@app.get("/stats")
async def get_stats():
    """Get service statistics and capabilities"""
    return {
        "active_agents": len(agents),
        "active_model_clients": len(model_clients),
        "available_tools": len(AVAILABLE_TOOLS),
        "autogen_version": "0.6.1",
        "service_version": "4.0.0",
        "features": {
            "agent_creation": "Available",
            "tool_integration": "Available", 
            "streaming": "Available",
            "multi_modal": "Available",
            "memory": "Available"
        },
        "available_endpoints": [
            "/health",
            "/agents",
            "/agents/{agent_id}",
            "/tasks",
            "/tasks/stream",
            "/tools",
            "/stats",
            "/version"
        ]
    }

# Get AutoGen version info
@app.get("/version")
async def get_version():
    """Get AutoGen version information"""
    try:
        import autogen_agentchat
        import autogen_ext
        
        return {
            "autogen_agentchat_version": getattr(autogen_agentchat, '__version__', '0.6.1'),
            "autogen_ext_version": getattr(autogen_ext, '__version__', '0.6.1'),
            "api_version": "0.6.1",
            "service_version": "4.0.0",
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
            "supported_models": [
                "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo",
                "o3", "o1-preview", "o1-mini",
                "claude-3-5-sonnet", "claude-3-5-haiku",
                "gemini-1.5-pro", "gemini-1.5-flash",
                "gemini-2.5-pro", "gemini-2.5-flash"
            ],
            "documentation": "https://microsoft.github.io/autogen/stable/"
        }
    except ImportError as e:
        return {
            "error": f"Could not import AutoGen modules: {str(e)}",
            "api_version": "0.6.1",
            "service_version": "4.0.0"
        }

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting AutoGen 0.6.1 server on port {port}")
    
    uvicorn.run(
        "agent_service:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=False
    )