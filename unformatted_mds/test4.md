#Kubernetes Deployment Guide
This guide covers deploying microservices to Kubernetes.It assumes you have a working cluster and kubectl configured.
##Prerequisites
Before you begin,make sure you have the following tools installed:
-  kubectl (v1.28 or later)
- helm>=3.12
-   docker
- skaffold(optional, for local development)
- kustomize(optional)

You should also have access to a container registry.We recommend using  GitHub Container Registry or  Docker Hub for public images.
##Cluster Setup
###Creating a Namespace
First,create a dedicated namespace for your services.This keeps things  organized and allows for resource quotas.
```
kubectl create namespace myapp-production
kubectl create namespace myapp-staging

```

You can verify the namespace was created with:
```
kubectl get namespaces|grep myapp
```
###Configuring Resource Limits
Every deployment should have resource limits defined.Without them,a single pod can consume all cluster resources and cause  outages.Here is an example resource configuration:
```yaml
resources:
  requests:
    memory: "128Mi"
    cpu: "250m"
  limits:
    memory: "256Mi"
    cpu: "500m"
```
##Container Images
###Building Images
All services use multi-stage Docker builds to minimize image size.The build process involves:
1.Compiling the application in a builder stage
2.Copying only the binary to a minimal runtime image
3.Setting up health check endpoints
4.Configuring the entrypoint

Make sure to tag images with both the git SHA and  the semantic version.This makes rollbacks much easier when things go wrong in production.
###Image Registry
We push all images to `ghcr.io/myorg`.The CI pipeline handles this automatically,but you can also push manually:
```
docker build -t ghcr.io/myorg/api-server:v1.2.3 .
docker push ghcr.io/myorg/api-server:v1.2.3
```
##Helm Charts
###Chart Structure
Each microservice has its own Helm chart located in the `charts/` directory.The standard structure looks like:
|Directory|Purpose|Required|
|---|---|---|
|templates/|Kubernetes manifests|yes|
|values.yaml|Default configuration|yes|
|Chart.yaml|Chart metadata|yes|
|values-prod.yaml|Production overrides|no|
|values-staging.yaml|Staging overrides|no|

###Installing a Chart
To deploy a service,use helm install with the appropriate values file:
```
helm install api-server ./charts/api-server -f values-prod.yaml --namespace myapp-production
```
To upgrade an existing deployment:
```
helm upgrade api-server ./charts/api-server -f values-prod.yaml --namespace myapp-production
```
##Monitoring and Observability
###Health Checks
Every service exposes two health endpoints:
-  `/healthz` for liveness probes
-`/readyz` for readiness probes
-   `/metrics` for Prometheus scraping

Kubernetes uses these to determine if a pod is healthy.If the liveness probe fails three times,the pod is  restarted.If the readiness probe  fails,the pod is removed from the service load balancer.
###Logging
All services log to stdout in JSON format.Fluentd collects these logs and  forwards them to Elasticsearch.You can query logs using Kibana at `https://logs.internal.example.com`.
###Alerting
We use Alertmanager for routing alerts.Critical alerts go to PagerDuty,warnings go to  Slack.Alert rules are defined in `monitoring/alerts/` and are deployed via the prometheus-operator Helm chart.
##Troubleshooting
Common issues and their solutions:
1.  **Pod stuck in CrashLoopBackOff** - Check logs with `kubectl logs <pod-name> -n myapp-production`.Usually indicates a missing environment variable or  failed database connection.
2.**ImagePullBackOff** - Verify the image exists in the registry and that imagePullSecrets are configured correctly.
3.  **Service not reachable** - Check that the service selector matches pod labels.Run `kubectl get endpoints` to verify.
4.**Resource quota exceeded** - Request a quota increase or optimize existing resource  requests.

For additional help,reach out in the `#platform-engineering` Slack channel or file an issue  in the infrastructure repository.
##Rollback Procedures
If a deployment causes issues,rollback immediately:
```
helm rollback api-server 1 --namespace myapp-production
```
Always notify the team in Slack when performing a rollback.Document the reason and create a post-mortem if user-facing impact occurred.
