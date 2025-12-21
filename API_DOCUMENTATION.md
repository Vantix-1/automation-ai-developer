# AI Learning API - API Documentation

**Version**: 1.0.0
**Generated**: 2025-12-21 13:32:20

Comprehensive API for AI learning and experimentation

## Base URL
https://api.example.com

## Authentication
All endpoints (except public ones) require authentication via Bearer token.

```bash
Authorization: Bearer <your_token>
```

## Endpoints

### /chat

#### POST
Send a message to the AI and get a response

**Request Body:**

```json
{
  "message": "What is machine learning?",
  "user_id": 1
}
```

**Responses:**

- **200**: Chat response

  Example:
  ```json
  {
  "response": "Machine learning is a subset of AI...",
  "message_id": 123,
  "processing_time": 0.45
}
  ```

### /users/{user_id}

#### GET
Retrieve a specific user by their ID

**Parameters:**

| Name | In | Type | Required | Description |
|------|----|------|----------|-------------|
| user_id | path | integer | True | User ID |

**Responses:**

- **200**: Successful response

  Example:
  ```json
  {
  "id": 1,
  "name": "John Doe",
  "email": "john@example.com",
  "role": "admin"
}
  ```
- **404**: User not found

## Schemas

### User
```json
{
  "type": "object",
  "required": [
    "id",
    "name",
    "email"
  ],
  "properties": {
    "id": {
      "type": "integer",
      "example": 1
    },
    "name": {
      "type": "string",
      "example": "John Doe"
    },
    "email": {
      "type": "string",
      "format": "email",
      "example": "john@example.com"
    },
    "role": {
      "type": "string",
      "default": "user",
      "enum": [
        "user",
        "admin"
      ]
    }
  }
}
```

### ChatMessage
```json
{
  "type": "object",
  "required": [
    "message",
    "user_id"
  ],
  "properties": {
    "message": {
      "type": "string",
      "example": "Hello, AI!"
    },
    "user_id": {
      "type": "integer",
      "example": 1
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

## Examples

### Python (requests)
```python
import requests

# Base configuration
BASE_URL = "https://api.example.com"
HEADERS = {
    "Authorization": "Bearer your_token_here",
    "Content-Type": "application/json"
}

# Example: Get user
response = requests.get(f"{BASE_URL}/users/123", headers=HEADERS)
print(response.json())

# Example: Create item
data = {
    "name": "New Item",
    "price": 29.99
}
response = requests.post(f"{BASE_URL}/items/", json=data, headers=HEADERS)
print(response.json())
```

### cURL
```bash
# Get user
curl -X GET "https://api.example.com/users/123" \
  -H "Authorization: Bearer your_token"

# Create item
curl -X POST "https://api.example.com/items/" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{"name": "New Item", "price": 29.99}'
```

### JavaScript (fetch)
```javascript
const baseUrl = 'https://api.example.com';
const headers = {
    'Authorization': 'Bearer your_token',
    'Content-Type': 'application/json'
};

// Get user
fetch(`${baseUrl}/users/123`, { headers })
    .then(response => response.json())
    .then(data => console.log(data));

// Create item
const itemData = {
    name: 'New Item',
    price: 29.99
};

fetch(`${baseUrl}/items/`, {
    method: 'POST',
    headers,
    body: JSON.stringify(itemData)
})
.then(response => response.json())
.then(data => console.log(data));
```

## Error Handling
The API uses standard HTTP status codes:

| Code | Description |
|------|-------------|
| 200  | Success |
| 201  | Created |
| 400  | Bad Request |
| 401  | Unauthorized |
| 403  | Forbidden |
| 404  | Not Found |
| 422  | Validation Error |
| 500  | Internal Server Error |

## Rate Limiting
- 100 requests per minute per IP
- 1000 requests per hour per API key

## Support
For support, contact:
- **Email**: support@example.com
- **Documentation**: https://docs.example.com
- **GitHub Issues**: https://github.com/example/api/issues
