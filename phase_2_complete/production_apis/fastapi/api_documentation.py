"""
API Documentation Generator
Creates comprehensive OpenAPI documentation with examples
Complete standalone implementation
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class APIDocumentation:
    """Generate comprehensive API documentation"""
    
    def __init__(self, title: str, version: str, description: str = ""):
        self.title = title
        self.version = version
        self.description = description
        self.openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": description,
                "contact": {
                    "name": "API Support",
                    "email": "support@example.com"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            }
        }
        
    def add_endpoint(self, method: str, path: str, endpoint_info: Dict[str, Any]):
        """Add endpoint documentation to OpenAPI spec"""
        if path not in self.openapi_spec["paths"]:
            self.openapi_spec["paths"][path] = {}
        
        self.openapi_spec["paths"][path][method.lower()] = endpoint_info
    
    def add_schema(self, name: str, schema: Dict[str, Any]):
        """Add JSON schema to components"""
        self.openapi_spec["components"]["schemas"][name] = schema
    
    def generate_markdown(self) -> str:
        """Generate Markdown documentation"""
        md = f"""# {self.title} - API Documentation

**Version**: {self.version}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{self.description}

## Base URL
https://api.example.com

## Authentication
All endpoints (except public ones) require authentication via Bearer token.

```bash
Authorization: Bearer <your_token>
```

## Endpoints
"""
        
        for path, methods in sorted(self.openapi_spec["paths"].items()):
            md += f"\n### {path}\n"
            
            for method, info in methods.items():
                md += f"\n#### {method.upper()}\n"
                md += f"{info.get('description', 'No description')}\n\n"
                
                if "parameters" in info:
                    md += "**Parameters:**\n\n"
                    md += "| Name | In | Type | Required | Description |\n"
                    md += "|------|----|------|----------|-------------|\n"
                    for param in info["parameters"]:
                        md += f"| {param['name']} | {param['in']} | {param['schema'].get('type', 'string')} | {param.get('required', False)} | {param.get('description', '')} |\n"
                    md += "\n"
                
                if "requestBody" in info:
                    md += "**Request Body:**\n\n"
                    md += "```json\n"
                    content = info["requestBody"]["content"]
                    for media_type, details in content.items():
                        if "example" in details:
                            example = json.dumps(details["example"], indent=2)
                            md += f"{example}\n"
                        elif "schema" in details and "$ref" in details["schema"]:
                            schema_ref = details["schema"]["$ref"].split("/")[-1]
                            md += f"See schema: {schema_ref}\n"
                    md += "```\n\n"
                
                if "responses" in info:
                    md += "**Responses:**\n\n"
                    for status_code, response in info["responses"].items():
                        md += f"- **{status_code}**: {response.get('description', '')}\n"
                        
                        if "content" in response:
                            content = response["content"]
                            for media_type, details in content.items():
                                if "example" in details:
                                    md += f"\n  Example:\n  ```json\n"
                                    example = json.dumps(details["example"], indent=2)
                                    md += f"  {example}\n"
                                    md += "  ```\n"
        
        # Add schemas section
        if self.openapi_spec["components"]["schemas"]:
            md += "\n## Schemas\n"
            for name, schema in self.openapi_spec["components"]["schemas"].items():
                md += f"\n### {name}\n"
                md += "```json\n"
                md += json.dumps(schema, indent=2)
                md += "\n```\n"
        
        # Add examples section
        md += """
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
curl -X GET "https://api.example.com/users/123" \\
  -H "Authorization: Bearer your_token"

# Create item
curl -X POST "https://api.example.com/items/" \\
  -H "Authorization: Bearer your_token" \\
  -H "Content-Type: application/json" \\
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
"""
        
        return md
    
    def save(self, format: str = "all"):
        """Save documentation in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ["json", "all"]:
            json_path = Path(f"api_documentation_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump(self.openapi_spec, f, indent=2)
            print(f"✅ JSON documentation saved to {json_path}")
        
        if format in ["yaml", "all"]:
            yaml_path = Path(f"api_documentation_{timestamp}.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(self.openapi_spec, f)
            print(f"✅ YAML documentation saved to {yaml_path}")
        
        if format in ["markdown", "all"]:
            md_path = Path(f"API_DOCUMENTATION.md")
            md_content = self.generate_markdown()
            with open(md_path, "w") as f:
                f.write(md_content)
            print(f"✅ Markdown documentation saved to {md_path}")
        
        # Generate HTML documentation using redoc-cli if available
        try:
            import subprocess
            if format in ["html", "all"]:
                json_path = f"api_documentation_{timestamp}.json"
                html_path = f"api_documentation_{timestamp}.html"
                subprocess.run([
                    "redoc-cli", "bundle", json_path,
                    "-o", html_path
                ])
                print(f"✅ HTML documentation saved to {html_path}")
        except Exception as e:
            print("⚠️  redoc-cli not available, skipping HTML generation")


def main():
    """Example usage"""
    # Create documentation generator
    doc = APIDocumentation(
        title="AI Learning API",
        version="1.0.0",
        description="Comprehensive API for AI learning and experimentation"
    )
    
    # Add schemas
    user_schema = {
        "type": "object",
        "required": ["id", "name", "email"],
        "properties": {
            "id": {"type": "integer", "example": 1},
            "name": {"type": "string", "example": "John Doe"},
            "email": {"type": "string", "format": "email", "example": "john@example.com"},
            "role": {"type": "string", "default": "user", "enum": ["user", "admin"]}
        }
    }
    
    chat_schema = {
        "type": "object",
        "required": ["message", "user_id"],
        "properties": {
            "message": {"type": "string", "example": "Hello, AI!"},
            "user_id": {"type": "integer", "example": 1},
            "timestamp": {"type": "string", "format": "date-time"}
        }
    }
    
    doc.add_schema("User", user_schema)
    doc.add_schema("ChatMessage", chat_schema)
    
    # Add endpoints
    doc.add_endpoint("GET", "/users/{user_id}", {
        "summary": "Get user by ID",
        "description": "Retrieve a specific user by their ID",
        "parameters": [
            {
                "name": "user_id",
                "in": "path",
                "required": True,
                "schema": {"type": "integer"},
                "description": "User ID"
            }
        ],
        "responses": {
            "200": {
                "description": "Successful response",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/User"},
                        "example": {
                            "id": 1,
                            "name": "John Doe",
                            "email": "john@example.com",
                            "role": "admin"
                        }
                    }
                }
            },
            "404": {
                "description": "User not found"
            }
        }
    })
    
    doc.add_endpoint("POST", "/chat", {
        "summary": "Send chat message",
        "description": "Send a message to the AI and get a response",
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ChatMessage"},
                    "example": {
                        "message": "What is machine learning?",
                        "user_id": 1
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Chat response",
                "content": {
                    "application/json": {
                        "example": {
                            "response": "Machine learning is a subset of AI...",
                            "message_id": 123,
                            "processing_time": 0.45
                        }
                    }
                }
            }
        }
    })
    
    # Save documentation
    doc.save("markdown")
    
    print("\n📚 Documentation Generation Complete!")
    print("====================================")
    print("Available formats:")
    print("- API_DOCUMENTATION.md (Markdown)")
    print(f"- api_documentation_*.json (OpenAPI JSON)")
    print(f"- api_documentation_*.yaml (OpenAPI YAML)")
    print("\nTo serve interactive documentation:")
    print("  pip install fastapi uvicorn")
    print("  uvicorn api_documentation:app --reload")


if __name__ == "__main__":
    main()