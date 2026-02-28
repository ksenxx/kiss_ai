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

