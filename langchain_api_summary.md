# Langchain API Summary

This document provides a summary of the Langchain API, based on the OpenAPI specification. It includes details on the most important endpoints, their parameters, and expected responses.

## Authentication

To authenticate with the LangSmith Deployment Control Plane API, set the `X-Api-Key` header to a valid LangSmith API key.

## API Versioning

Each endpoint path is prefixed with a version (e.g. `v1`, `v2`).

## Endpoints

### Deployments

*   **`GET /v2/deployments`**: List all deployments.
*   **`POST /v2/deployments`**: Create a new deployment.
*   **`GET /v2/deployments/{deployment_id}`**: Get a deployment by ID.
*   **`PATCH /v2/deployments/{deployment_id}`**: Patch a deployment by ID.
*   **`DELETE /v2/deployments/{deployment_id}`**: Delete a deployment by ID.

### Revisions

*   **`GET /v2/deployments/{deployment_id}/revisions`**: List all revisions for a deployment.
*   **`GET /v2/deployments/{deployment_id}/revisions/{revision_id}`**: Get a revision by ID for a deployment.
*   **`POST /v2/deployments/{deployment_id}/revisions/{revision_id}/redeploy`**: Redeploy a specific revision ID.

### Listeners

*   **`GET /v2/listeners`**: List all listeners.
*   **`POST /v2/listeners`**: Create a listener.
*   **`GET /v2/listeners/{listener_id}`**: Get a listener by ID.
*   **`PATCH /v2/listeners/{listener_id}`**: Patch a listener by ID.
*   **`DELETE /v2/listeners/{listener_id}`**: Delete a listener by ID.

### Authentication Service

*   **`GET /v2/auth/providers`**: List OAuth providers.
*   **`POST /v2/auth/providers`**: Create a new OAuth provider manually.
*   **`GET /v2/auth/providers/{provider_id}`**: Get a specific OAuth provider.
*   **`PATCH /v2/auth/providers/{provider_id}`**: Update an OAuth provider.
*   **`DELETE /v2/auth/providers/{provider_id}`**: Delete an OAuth provider.
*   **`POST /v2/auth/authenticate`**: Get OAuth token or start authentication flow if needed.
*   **`GET /v2/auth/wait/{auth_id}`**: Wait for OAuth authentication completion.
*   **`GET /v2/auth/tokens`**: List the calling user's tokens for a provider.
*   **`DELETE /v2/auth/tokens`**: Delete all tokens for the current user for the given provider (across agents).
*   **`PATCH /v2/auth/tokens/{token_id}/metadata`**: Update a token's provider_account_label.
*   **`DELETE /v2/auth/tokens/{token_id}`**: Delete a specific OAuth token.

### Integrations

*   **`GET /v1/integrations/github/install`**: List available GitHub integrations for LangGraph Platfom Cloud SaaS.
*   **`GET /v1/integrations/github/{integration_id}/repos`**: List available GitHub repositories for an integration that are available to deploy to LangSmith Deployment.