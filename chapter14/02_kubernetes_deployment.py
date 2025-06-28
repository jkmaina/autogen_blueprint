"""
Chapter 14: Deployment Patterns
Example 2: Kubernetes Deployment

Description:
Demonstrates comprehensive Kubernetes deployment patterns for AutoGen v0.5 agents
including container orchestration, service mesh integration, horizontal pod
autoscaling, and production-ready cluster deployment configurations.

Prerequisites:
- Kubernetes cluster access (local or cloud)
- kubectl CLI tool configured
- Docker registry access for image storage
- OpenAI API key for agent services
- Basic understanding of Kubernetes concepts

Usage:
```bash
python -m chapter14.02_kubernetes_deployment
```

Expected Output:
Kubernetes deployment setup demonstration:
1. Deployment manifest generation
2. Service and ingress configuration
3. Secrets management setup
4. Horizontal Pod Autoscaler configuration
5. Health check endpoint integration
6. Complete deployment instructions

Key Concepts:
- Kubernetes orchestration
- Container scaling strategies
- Service discovery and load balancing
- Secrets management
- Health monitoring and probes
- Ingress traffic routing
- Resource management and limits
- Production deployment best practices

AutoGen Version: 0.5+
"""

# Standard library imports
import argparse
import os
from typing import Any, Dict

# Third-party imports (None required for this deployment script)

# Local imports (None required for this deployment script)

# Define the Kubernetes deployment YAML
DEPLOYMENT_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autogen-agent-service
  labels:
    app: autogen-agent-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: autogen-agent-service
  template:
    metadata:
      labels:
        app: autogen-agent-service
    spec:
      containers:
      - name: autogen-agent-service
        image: ${DOCKER_REGISTRY}/autogen-agent-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: PORT
          value: "8000"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: autogen-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
"""

# Define the Kubernetes service YAML
SERVICE_YAML = """
apiVersion: v1
kind: Service
metadata:
  name: autogen-agent-service
spec:
  selector:
    app: autogen-agent-service
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
"""

# Define the Kubernetes ingress YAML
INGRESS_YAML = """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autogen-agent-service
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  rules:
  - host: ${DOMAIN_NAME}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: autogen-agent-service
            port:
              number: 80
  tls:
  - hosts:
    - ${DOMAIN_NAME}
    secretName: autogen-tls-secret
"""

# Define the Kubernetes secrets YAML
SECRETS_YAML = """
apiVersion: v1
kind: Secret
metadata:
  name: autogen-secrets
type: Opaque
data:
  openai-api-key: ${OPENAI_API_KEY_BASE64}
"""

# Define the Kubernetes horizontal pod autoscaler YAML
HPA_YAML = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: autogen-agent-service
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autogen-agent-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""

# Define the agent_service.py content with health endpoint
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
"""

def create_kubernetes_files(output_dir: str) -> None:
    """
    Create the Kubernetes deployment files in the specified directory.
    
    Args:
        output_dir: The directory to create the files in
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the deployment.yaml file
    with open(os.path.join(output_dir, "deployment.yaml"), "w") as f:
        f.write(DEPLOYMENT_YAML)
    
    # Create the service.yaml file
    with open(os.path.join(output_dir, "service.yaml"), "w") as f:
        f.write(SERVICE_YAML)
    
    # Create the ingress.yaml file
    with open(os.path.join(output_dir, "ingress.yaml"), "w") as f:
        f.write(INGRESS_YAML)
    
    # Create the secrets.yaml file
    with open(os.path.join(output_dir, "secrets.yaml"), "w") as f:
        f.write(SECRETS_YAML)
    
    # Create the hpa.yaml file
    with open(os.path.join(output_dir, "hpa.yaml"), "w") as f:
        f.write(HPA_YAML)
    
    # Create the agent_service.py file
    with open(os.path.join(output_dir, "agent_service.py"), "w") as f:
        f.write(AGENT_SERVICE)
    
    # Create a README.md file with deployment instructions
    readme_content = """# AutoGen Kubernetes Deployment

This directory contains Kubernetes manifests for deploying AutoGen agents.

## Prerequisites

- Kubernetes cluster
- kubectl configured to access your cluster
- Docker registry access
- Domain name for ingress (optional)

## Deployment Steps

1. Build and push the Docker image:
   ```bash
   docker build -t your-registry/autogen-agent-service:latest .
   docker push your-registry/autogen-agent-service:latest
   ```

2. Update the deployment files:
   - Replace `${DOCKER_REGISTRY}` in deployment.yaml with your Docker registry
   - Replace `${DOMAIN_NAME}` in ingress.yaml with your domain name
   - Create a base64-encoded API key:
     ```bash
     echo -n "your-openai-api-key" | base64
     ```
   - Replace `${OPENAI_API_KEY_BASE64}` in secrets.yaml with the base64-encoded API key

3. Apply the Kubernetes manifests:
   ```bash
   kubectl apply -f secrets.yaml
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f ingress.yaml
   kubectl apply -f hpa.yaml
   ```

4. Verify the deployment:
   ```bash
   kubectl get pods
   kubectl get services
   kubectl get ingress
   ```

5. Access the API:
   - If using ingress: https://your-domain-name/docs
   - If using port-forward: 
     ```bash
     kubectl port-forward svc/autogen-agent-service 8000:80
     ```
     Then access: http://localhost:8000/docs
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    print(f"Kubernetes deployment files created in {output_dir}")
    print("\nFollow the instructions in README.md to deploy to Kubernetes.")

def main() -> None:
    """Main function to demonstrate Kubernetes deployment pattern creation."""
    parser = argparse.ArgumentParser(description="Create Kubernetes deployment files for AutoGen")
    parser.add_argument("--output", type=str, default="./k8s-deployment", help="Output directory for deployment files")
    
    args = parser.parse_args()
    create_kubernetes_files(args.output)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKubernetes deployment demo interrupted by user")
    except Exception as e:
        print(f"Error running Kubernetes deployment demo: {e}")
        raise
