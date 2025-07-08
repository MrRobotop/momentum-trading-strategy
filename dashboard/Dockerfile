# Multi-stage Dockerfile for React Frontend
# ==========================================
# 
# Production-ready Docker image for momentum strategy dashboard
# with optimized build, security, and performance.
# 
# Author: Rishabh Ashok Patil

# Build stage
FROM node:18-alpine as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=2.0.0
ARG VCS_REF

# Add metadata
LABEL maintainer="Rishabh Ashok Patil" \
      version="${VERSION}" \
      description="Momentum Trading Strategy Dashboard" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine as production

# Install security updates
RUN apk update && apk upgrade && \
    apk add --no-cache curl && \
    rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 -S momentum && \
    adduser -S momentum -u 1001

# Copy built application from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create necessary directories and set permissions
RUN mkdir -p /var/cache/nginx /var/log/nginx /var/run && \
    chown -R momentum:momentum /var/cache/nginx /var/log/nginx /var/run /usr/share/nginx/html

# Copy startup script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Switch to non-root user
USER momentum

# Start nginx
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
