"""
Automated API Documentation Generator
Creates comprehensive documentation from OpenAPI specs
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import inspect
import re


class APIDocumentationGenerator:
    """Generate comprehensive API documentation"""
    
    def __init__(self, title: str, version: str, description: str = ""):
        self.title = title
        self.version = version
        self.description = description
        self.sections = []
        
    def add_section(self, title: str, content: str, level: int = 2):
        """Add a documentation section"""
        self.sections.append({
            "title": title,
            "content": content,
            "level": level,
            "id": self._slugify(title)
        })
    
    def add_endpoint(self, method: str, path: str, summary: str,
                    description: str = "", parameters: List[Dict] = None,
                    request_body: Dict = None, responses: List[Dict] = None,
                    examples: List[Dict] = None):
        """Add endpoint documentation"""
        endpoint_section = f"""
### {method.upper()} {path}

**{summary}**

{description}

"""
        
        if parameters:
            endpoint_section += "#### Parameters\n\n"
            endpoint_section += "| Name | In | Type | Required | Description |\n"
            endpoint_section += "|------|----|------|----------|-------------|\n"
            for param in parameters:
                endpoint_section += f"| {param.get('name', '')} | {param.get('in', '')} | {param.get('schema', {}).get('type', '')} | {param.get('required', False)} | {param.get('description', '')} |\n"
            endpoint_section += "\n"
        
        if request_body:
            endpoint_section += "#### Request Body\n\n"
            endpoint_section += "```json\n"
            if 'example' in request_body:
                endpoint_section += json.dumps(request_body['example'], indent=2)
            endpoint_section += "\n```\n\n"
        
        if responses:
            endpoint_section += "#### Responses\n\n"
            for response in responses:
                endpoint_section += f"- **{response.get('status', '')}**: {response.get('description', '')}\n"
                if 'example' in response:
                    endpoint_section += "  ```json\n  "
                    endpoint_section += json.dumps(response['example'], indent=2)
                    endpoint_section += "\n  ```\n"
            endpoint_section += "\n"
        
        if examples:
            endpoint_section += "#### Examples\n\n"
            for example in examples:
                endpoint_section += f"**{example.get('title', 'Example')}**\n\n"
                if 'code' in example:
                    endpoint_section += f"```{example.get('language', 'bash')}\n"
                    endpoint_section += example['code']
                    endpoint_section += "\n```\n\n"
        
        self.add_section(f"{method.upper()} {path}", endpoint_section, level=3)
    
    def add_code_example(self, language: str, code: str, title: str = "Example"):
        """Add code example"""
        code_section = f"**{title}**\n\n```{language}\n{code}\n```\n"
        self.sections.append({
            "title": title,
            "content": code_section,
            "type": "code",
            "language": language
        })
    
    def generate_markdown(self) -> str:
        """Generate markdown documentation"""
        md = f"""# {self.title}

**Version**: {self.version}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{self.description}

## Table of Contents

"""
        
        # Generate table of contents
        for section in self.sections:
            indent = "  " * (section["level"] - 2)
            md += f"{indent}- [{section['title']}](#{section.get('id', '')})\n"
        
        md += "\n"
        
        # Add sections
        for section in self.sections:
            md += "#" * section["level"] + f" {section['title']}\n\n"
            md += section["content"] + "\n\n"
        
        # Add examples section
        md += self._generate_examples_section()
        
        # Add error handling section
        md += self._generate_error_handling_section()
        
        # Add deployment section
        md += self._generate_deployment_section()
        
        return md
    
    def _generate_examples_section(self) -> str:
        """Generate examples section"""
        examples = """
## Examples

### Python (requests)

```python
import requests
import json

BASE_URL = "https://api.example.com"
HEADERS = {
    "Authorization": "Bearer YOUR_API_TOKEN",
    "Content-Type": "application/json"
}

# Example 1: Get user
response = requests.get(f"{BASE_URL}/users/123", headers=HEADERS)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Example 2: Create item
item_data = {
    "name": "New Item",
    "description": "Item description",
    "price": 29.99
}

response = requests.post(
    f"{BASE_URL}/items/",
    headers=HEADERS,
    json=item_data
)

if response.status_code == 201:
    print("Item created successfully")
    print(f"Item ID: {response.json()['id']}")
else:
    print(f"Error: {response.status_code}")
    print(f"Message: {response.json()['detail']}")
```

### cURL

```bash
# Get user
curl -X GET "https://api.example.com/users/123" \\
  -H "Authorization: Bearer YOUR_API_TOKEN" \\
  -H "Content-Type: application/json"

# Create item
curl -X POST "https://api.example.com/items/" \\
  -H "Authorization: Bearer YOUR_API_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"name": "New Item", "price": 29.99}'

# Upload file
curl -X POST "https://api.example.com/upload" \\
  -H "Authorization: Bearer YOUR_API_TOKEN" \\
  -F "file=@document.pdf" \\
  -F "metadata='{\\"category\\": \\"documents\\"}'"
```

### JavaScript (fetch)

```javascript
const baseUrl = 'https://api.example.com';
const headers = {
    'Authorization': 'Bearer YOUR_API_TOKEN',
    'Content-Type': 'application/json'
};

// Get user
fetch(`${baseUrl}/users/123`, { headers })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));

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
.then(data => console.log('Created:', data))
.catch(error => console.error('Error:', error));
```

### Python (async with aiohttp)

```python
import aiohttp
import asyncio

async def make_request():
    async with aiohttp.ClientSession() as session:
        headers = {
            'Authorization': 'Bearer YOUR_API_TOKEN',
            'Content-Type': 'application/json'
        }
        
        # Get user
        async with session.get(
            'https://api.example.com/users/123',
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(f"User: {data}")
            else:
                print(f"Error: {response.status}")
        
        # Create item
        item_data = {
            "name": "New Item",
            "price": 29.99
        }
        
        async with session.post(
            'https://api.example.com/items/',
            headers=headers,
            json=item_data
        ) as response:
            if response.status == 201:
                data = await response.json()
                print(f"Created item: {data['id']}")
            else:
                error = await response.json()
                print(f"Error: {error['detail']}")

# Run the async function
asyncio.run(make_request())
```
"""
        return examples

    def _generate_error_handling_section(self) -> str:
        """Generate error handling section"""
        errors = """
## Error Handling

The API uses standard HTTP status codes and returns error details in JSON format.

### Common HTTP Status Codes

| Code | Description | Typical Use Case |
|------|-------------|------------------|
| 200 | OK | Successful GET request |
| 201 | Created | Resource created successfully |
| 204 | No Content | Successful DELETE request |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Resource conflict (e.g., duplicate) |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |

### Error Response Format

```json
{
    "error": {
        "code": "validation_error",
        "message": "Invalid input data",
        "details": {
            "field": "email",
            "error": "Invalid email format"
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

### Common Error Codes

| Error Code | Description | Resolution |
|------------|-------------|------------|
| invalid_token | Invalid or expired token | Refresh token or re-authenticate |
| rate_limit_exceeded | Too many requests | Wait and retry later |
| validation_error | Input validation failed | Check request parameters |
| not_found | Resource not found | Verify resource ID |
| permission_denied | Insufficient permissions | Check user roles |

### Retry Logic Example

```python
import requests
import time
from requests.exceptions import RequestException

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(e.response.headers.get('Retry-After', 5))
                print(f"Rate limited. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            elif e.response.status_code >= 500:
                # Server error - retry with exponential backoff
                wait_time = (2 ** attempt) + 1
                print(f"Server error. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                # Client error - don't retry
                raise
        except RequestException as e:
            # Network error - retry
            wait_time = (2 ** attempt) + 1
            print(f"Network error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            continue
    
    raise Exception(f"Failed after {max_retries} attempts")
```
"""
        return errors

    def _generate_deployment_section(self) -> str:
        """Generate deployment section"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_year = datetime.now().year
        
        deployment = f"""
## Deployment

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| OPENAI_API_KEY | Yes | - | OpenAI API key |
| DATABASE_URL | Yes | - | PostgreSQL connection string |
| REDIS_URL | No | redis://localhost:6379/0 | Redis connection URL |
| SECRET_KEY | Yes | - | JWT secret key |
| ENVIRONMENT | No | production | Runtime environment |
| LOG_LEVEL | No | info | Logging level |
| CORS_ORIGINS | No | * | CORS allowed origins |

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale ai-api=3

# Run database migrations
docker-compose exec ai-api python manage.py migrate
```

### Kubernetes Deployment (Example)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-api
  template:
    metadata:
      labels:
        app: ai-api
    spec:
      containers:
      - name: ai-api
        image: your-registry/ai-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Health Checks

The API provides health check endpoints:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Ready check (for load balancers)
curl http://localhost:8000/health/ready

# Metrics (Prometheus format)
curl http://localhost:8000/metrics
```

### Monitoring and Alerting

- **Prometheus Metrics**: Available at /metrics
- **Grafana Dashboards**: Pre-configured dashboards
- **Alerting Rules**: Configured in Prometheus
- **Log Aggregation**: Structured JSON logs

### Backup and Recovery

```bash
# Backup database
docker-compose exec db pg_dump -U postgres ai_api > backup.sql

# Backup uploads
tar czf uploads_backup.tar.gz ./uploads

# Restore database
cat backup.sql | docker-compose exec -T db psql -U postgres ai_api
```

### Support

- **Documentation**: https://docs.example.com
- **API Reference**: https://api.example.com/docs
- **GitHub Issues**: https://github.com/your-org/ai-api/issues
- **Email Support**: support@example.com
- **Status Page**: https://status.example.com

### Changelog

**v2.0.0 (2024-01-15)**
- Added streaming support
- Improved error handling
- Added rate limiting
- Enhanced monitoring

**v1.0.0 (2023-12-01)**
- Initial release
- Basic CRUD operations
- Authentication
- File uploads

---

*Documentation generated automatically. Last updated: {current_date}*
"""
        return deployment

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug"""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')

    def save(self, format: str = "markdown", output_dir: str = "./docs"):
        """Save documentation in specified format"""
        Path(output_dir).mkdir(exist_ok=True)
        
        if format == "markdown":
            filename = Path(output_dir) / "API_DOCUMENTATION.md"
            content = self.generate_markdown()
            filename.write_text(content)
            print(f"✅ Markdown documentation saved to {filename}")
        
        elif format == "html":
            # Convert markdown to HTML
            try:
                import markdown
                md_content = self.generate_markdown()
                html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
            except ImportError:
                # Fallback if markdown library not available
                html_content = f"<pre>{self.generate_markdown()}</pre>"
            
            current_year = datetime.now().year
            
            # Wrap in HTML template
            html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title} - Documentation</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown-dark.min.css">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0d1117;
            color: #c9d1d9;
        }}
        .markdown-body {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 45px;
            background: #161b22;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #30363d;
        }}
        .header h1 {{
            color: #58a6ff;
            margin-bottom: 10px;
        }}
        .version {{
            color: #8b949e;
            font-size: 0.9em;
        }}
        .toc {{
            background: #21262d;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .toc ul {{
            list-style: none;
            padding-left: 20px;
        }}
        .toc a {{
            color: #58a6ff;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        code {{
            background: #0d1117;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }}
        pre code {{
            display: block;
            padding: 16px;
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #30363d;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #161b22;
            font-weight: 600;
        }}
        .endpoint {{
            background: #21262d;
            border-left: 4px solid #58a6ff;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .method {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.9em;
            margin-right: 10px;
        }}
        .get {{ background: #238636; color: white; }}
        .post {{ background: #1f6feb; color: white; }}
        .put {{ background: #9e6a03; color: white; }}
        .delete {{ background: #da3633; color: white; }}
        .patch {{ background: #8957e5; color: white; }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #30363d;
            text-align: center;
            color: #8b949e;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="markdown-body">
        <div class="header">
            <h1>{self.title}</h1>
            <div class="version">Version {self.version}</div>
            <p>{self.description}</p>
        </div>
        
        <div class="toc">
            <h2>Table of Contents</h2>
            {self._generate_toc_html()}
        </div>
        
        {html_content}
        
        <div class="footer">
            <p>Documentation generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>© {current_year} Your Company. All rights reserved.</p>
        </div>
    </div>
    
    <script>
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const targetElement = document.querySelector(targetId);
                if (targetElement) {{
                    window.scrollTo({{
                        top: targetElement.offsetTop - 20,
                        behavior: 'smooth'
                    }});
                }}
            }});
        }});
        
        // Add method badges to endpoints
        document.querySelectorAll('h3').forEach(h3 => {{
            const text = h3.textContent;
            const methodMatch = text.match(/^(GET|POST|PUT|DELETE|PATCH)\\s/);
            if (methodMatch) {{
                const method = methodMatch[1].toLowerCase();
                h3.innerHTML = `<span class="method ${{method}}">${{methodMatch[1]}}</span>` + 
                               text.substring(methodMatch[0].length);
            }}
        }});
    </script>
</body>
</html>"""
            
            filename = Path(output_dir) / "index.html"
            filename.write_text(html_template)
            print(f"✅ HTML documentation saved to {filename}")
        
        elif format == "json":
            # Generate OpenAPI-compatible JSON
            openapi_spec = {
                "openapi": "3.0.0",
                "info": {
                    "title": self.title,
                    "version": self.version,
                    "description": self.description
                },
                "paths": {},
                "components": {
                    "schemas": {},
                    "securitySchemes": {
                        "BearerAuth": {
                            "type": "http",
                            "scheme": "bearer"
                        }
                    }
                }
            }
            
            filename = Path(output_dir) / "openapi.json"
            with open(filename, "w") as f:
                json.dump(openapi_spec, f, indent=2)
            print(f"✅ OpenAPI JSON saved to {filename}")

    def _generate_toc_html(self) -> str:
        """Generate HTML table of contents"""
        toc_html = "<ul>"
        for section in self.sections:
            indent = 20 * (section["level"] - 2)
            toc_html += f'<li style="margin-left: {indent}px;">'
            toc_html += f'<a href="#{section["id"]}">{section["title"]}</a>'
            toc_html += "</li>"
        toc_html += "</ul>"
        return toc_html


class PythonCodeDocumentation:
    """Generate documentation from Python code"""
    
    @staticmethod
    def generate_module_docs(module_path: str) -> str:
        """Generate documentation for a Python module"""
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module"] = module
        spec.loader.exec_module(module)
        
        docs = f"# Module: {module_path}\n\n"
        
        # Get classes
        classes = inspect.getmembers(module, inspect.isclass)
        for name, cls in classes:
            if cls.__module__ == "module":
                docs += f"## Class: {name}\n\n"
                docs += f"*{cls.__doc__ or 'No documentation'}*\n\n"
                
                # Get methods
                methods = inspect.getmembers(cls, inspect.isfunction)
                for method_name, method in methods:
                    if not method_name.startswith('_'):
                        docs += f"### {method_name}()\n\n"
                        docs += f"```python\n{inspect.signature(method)}\n```\n\n"
                        docs += f"*{method.__doc__ or 'No documentation'}*\n\n"
        
        # Get functions
        functions = inspect.getmembers(module, inspect.isfunction)
        for name, func in functions:
            if func.__module__ == "module":
                docs += f"## Function: {name}\n\n"
                docs += f"```python\n{inspect.signature(func)}\n```\n\n"
                docs += f"*{func.__doc__ or 'No documentation'}*\n\n"
        
        return docs


# Example usage
def generate_complete_documentation():
    """Generate complete API documentation"""
    
    # Create documentation generator
    doc_gen = APIDocumentationGenerator(
        title="AI Learning API",
        version="2.0.0",
        description="Comprehensive API for AI learning and experimentation with support for chat, document processing, and RAG systems."
    )
    
    # Add overview section
    doc_gen.add_section("Overview", """
The AI Learning API provides a comprehensive set of endpoints for building AI-powered applications.
It includes features for natural language processing, document analysis, and machine learning.

### Key Features

- **Chat Completions**: Interactive conversations with AI models
- **Document Processing**: Upload and analyze PDFs, text files
- **RAG Systems**: Retrieval-Augmented Generation for document Q&A
- **Streaming**: Real-time responses via Server-Sent Events
- **Authentication**: JWT-based secure authentication
- **Rate Limiting**: Protection against abuse
- **Monitoring**: Comprehensive metrics and health checks

### Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   API Clients   │───▶│   FastAPI    │───▶│   OpenAI    │
│   (Web/Mobile)  │    │   Server     │    │    API      │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │                    │
                         ┌────┴────┐          ┌────┴────┐
                         │   DB    │          │ Vector  │
                         │(Postgres)│         │   DB    │
                         └──────────┘         └─────────┘
```

### Quick Start

1. **Get API Key**: Register at platform.openai.com
2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY='your-key-here'
   export SECRET_KEY='your-secret-key'
   ```
3. **Run the API**:
   ```bash
   docker-compose up -d
   ```
4. **Test the API**:
   ```bash
   curl http://localhost:8000/health
   ```
""")
    
    # Add authentication section
    doc_gen.add_section("Authentication", """
All API endpoints (except public ones) require authentication using Bearer tokens.

### Getting Started

1. **Register a user**:
   ```bash
   curl -X POST "https://api.example.com/auth/register" \\
     -H "Content-Type: application/json" \\
     -d '{
       "username": "your_username",
       "email": "your@email.com",
       "password": "SecurePass123!"
     }'
   ```

2. **Login to get tokens**:
   ```bash
   curl -X POST "https://api.example.com/auth/login" \\
     -H "Content-Type: application/json" \\
     -d '{
       "username": "your_username",
       "password": "SecurePass123!"
     }'
   ```

3. **Use the access token**:
   ```bash
   curl -X GET "https://api.example.com/secure/endpoint" \\
     -H "Authorization: Bearer your_access_token"
   ```

### Token Types

| Token Type | Purpose | Expiration |
|------------|---------|------------|
| Access Token | API authentication | 30 minutes |
| Refresh Token | Get new access tokens | 7 days |

### Refresh Tokens

When your access token expires, use the refresh token to get a new one:

```bash
curl -X POST "https://api.example.com/auth/refresh" \\
  -H "Content-Type: application/json" \\
  -d '{
    "refresh_token": "your_refresh_token"
  }'
```
""")
    
    # Add example endpoints
    doc_gen.add_endpoint(
        method="POST",
        path="/chat/completions",
        summary="Send chat message",
        description="Send a message to the AI and get a response. Supports streaming.",
        parameters=[
            {
                "name": "model",
                "in": "query",
                "required": False,
                "schema": {"type": "string"},
                "description": "AI model to use (default: gpt-3.5-turbo)"
            }
        ],
        request_body={
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is machine learning?"}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        },
        responses=[
            {
                "status": "200",
                "description": "Successful response",
                "example": {
                    "id": "chat-123",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Machine learning is a subset of artificial intelligence..."
                        }
                    }],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 150,
                        "total_tokens": 170
                    }
                }
            }
        ],
        examples=[
            {
                "title": "Basic chat",
                "language": "bash",
                "code": """curl -X POST "https://api.example.com/chat/completions" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, AI!"}
    ]
  }'"""
            }
        ]
    )
    
    doc_gen.add_endpoint(
        method="POST",
        path="/upload",
        summary="Upload document",
        description="Upload a document for processing and analysis.",
        request_body={
            "example": {
                "file": "(binary file data)",
                "metadata": {
                    "category": "research",
                    "tags": ["ai", "machine-learning"]
                }
            }
        },
        responses=[
            {
                "status": "200",
                "description": "Document uploaded successfully",
                "example": {
                    "id": "doc-123",
                    "filename": "research.pdf",
                    "size": 1024000,
                    "chunks": 45,
                    "status": "processing"
                }
            }
        ]
    )
    
    # Save documentation in multiple formats
    doc_gen.save("markdown", "./docs")
    doc_gen.save("html", "./docs")
    doc_gen.save("json", "./docs")
    
    print("✅ Complete documentation generated!")
    print("📄 Markdown: ./docs/API_DOCUMENTATION.md")
    print("🌐 HTML: ./docs/index.html")
    print("📊 OpenAPI: ./docs/openapi.json")


if __name__ == "__main__":
    generate_complete_documentation()