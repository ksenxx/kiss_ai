#API Reference
This document describes the REST API endpoints.All responses are in JSON format.
##Authentication
All requests must include an `Authorization` header with a valid  API key.
```
Authorization: Bearer <your-api-key>
```
To get an API key,visit the [dashboard](https://example.com/dashboard).Keys expire after 90 days.
##Endpoints
###GET /users
Returns a list of all users.Supports pagination.
**Parameters**:
|Name|Type|Required|Description|
|---|---|---|---|
|page|int|no|Page number(default: 1)|
|limit|int|no|Results per page(default: 20)|
|sort|string|no|Sort field|
|order|string|no|asc or desc(default: asc)|

**Example request**:
```
curl -H "Authorization: Bearer abc123" https://api.example.com/users?page=1&limit=10
```
**Example response**:
```json
{"users":[{"id":1,"name":"Alice","email":"alice@example.com"},{"id":2,"name":"Bob","email":"bob@example.com"}],"total":42,"page":1}
```
###POST /users
Creates a new user.The request body must be JSON.
**Request body**:
```json
{"name":"string","email":"string","role":"admin|user|viewer"}
```
**Response codes**:
- 201 - user created successfully
-  400 - invalid request body
-   409 - email already exists
- 500-internal server error

###DELETE /users/:id
Deletes a user by ID.This action is irreversible.
**Parameters**:
-`id` (path) - The user ID
**Response codes**:
* 204 - deleted successfully
*404 - user not found
*  403 - insufficient permissions

##Rate Limiting
The API enforces rate limits.The default limits are:
- 100 requests per minute for  authenticated users
-10 requests per minute for unauthenticated requests

When rate limited you will receive a `429 Too Many Requests` response.The response includes headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1620000000
```
##Error Handling
All errors follow this format:
```json
{"error":{"code":"VALIDATION_ERROR","message":"Invalid email format","details":{"field":"email","value":"not-an-email"}}}
```
Common error codes:
|Code|HTTP Status|Description|
|---|---|---|
|AUTH_REQUIRED|401|Missing or invalid API key|
|FORBIDDEN|403|Insufficient permissions|
|NOT_FOUND|404|Resource not found|
|RATE_LIMITED|429|Too many requests|
|INTERNAL|500|Internal server error|

> Important:never expose API keys in client-side code or public repositories.

##SDKs
We provide official SDKs for:
1.Python: `pip install example-sdk`
2.JavaScript: `npm install @example/sdk`
3.Go: `go get github.com/example/sdk-go`

For other languages,use the REST API directly.See the [examples](https://example.com/examples) page for code samples.
