FROM nginx:1.25-alpine

# Install certbot for SSL certificates
RUN apk add --no-cache certbot certbot-nginx

# Remove default configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create directories for SSL and static files
RUN mkdir -p /etc/nginx/ssl /usr/share/nginx/html/static /var/www/certbot

# Create self-signed certificate for initial setup
# Replace with real certificates in production
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/key.pem \
    -out /etc/nginx/ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/nginx_status || exit 1

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]