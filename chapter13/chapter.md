# Chapter 13  
## Deploying AutoGen Agent Services: A Beginner's Guide

### 13.1 Introduction

As you move from experimenting with AutoGen agents to deploying them in production, it's important to use best practices for reliability, scalability, and maintainability. This chapter will show you how to:

- Build a simple, production-ready API service for AutoGen agents using FastAPI.
- Containerize your service with Docker.
- Orchestrate your deployment with Docker Compose.
- Understand the basics of DevOps for AI agent systems.

---

### 13.2 Example: A Production-Ready AutoGen Agent API

The file `agent_service.py` in this folder is a complete, ready-to-use FastAPI service for managing and running AutoGen agents. It supports:

- Creating and deleting agents dynamically.
- Running tasks (including streaming responses).
- Health checks and statistics.
- Tool integration (e.g., calculator, current time).
- Easy extension for your own tools and agents.

#### Minimal Example: Creating and Running an Agent

```python
# agent_service.py (excerpt)
from fastapi import FastAPI
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

app = FastAPI()

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key="YOUR_OPENAI_API_KEY")

# Create an agent
agent = AssistantAgent(
    name="my_agent",
    system_message="You are a helpful assistant.",
    model_client=model_client,
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/run")
async def run_task(task: str):
    result = await agent.run(task=task)
    return {"response": result.messages[-1].content}
```

#### Full-Featured Service

See the full `agent_service.py` for:

- Dynamic agent creation (`/agents`)
- Task execution (`/tasks`, `/tasks/stream`)
- Tool integration
- Health and stats endpoints

---

### 13.3 Containerizing with Docker

**Dockerfile** (see `chapter13/Dockerfile`):

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent_service.py .
COPY .env* ./

# Expose port
EXPOSE 8000

CMD ["python", "agent_service.py"]
```

---

### 13.4 Orchestrating with Docker Compose

**docker-compose.yml** (see `chapter13/docker-compose.yml`):

```yaml
services:
  agent-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=your-openai-key
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

---

### 13.5 Running the Service

1. **Set your OpenAI API key** in a `.env` file or as an environment variable.
2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```
3. **Check the health endpoint:**
   ```
   curl http://localhost:8000/health
   ```
4. **Use the API** (see `/docs` for Swagger UI).

---

### 13.6 Requirements

**requirements.txt** (see `chapter13/requirements.txt`):

```
autogen-core>=0.6.1
autogen-agentchat>=0.6.1
autogen-ext[openai]>=0.6.1
fastapi>=0.115.13
uvicorn[standard]>=0.34.3
python-dotenv>=1.1.1
pydantic>=2.11.6
```

---

### 13.7 Summary

- You now have a template for deploying AutoGen agents as a robust API service.
- The provided code and Docker setup are suitable for both local development and production.
- You can extend the service with your own agents, tools, and endpoints.

---

**For more details, see the code in `agent_service.py` and the deployment files in this folder.**
