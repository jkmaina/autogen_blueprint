# AutoGen Kubernetes Deployment

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
