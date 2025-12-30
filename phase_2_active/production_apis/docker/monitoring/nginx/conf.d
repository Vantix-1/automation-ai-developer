events {
    worker_connections 1024;
}

http {
    upstream ai_api {
        server ai-api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://ai_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}