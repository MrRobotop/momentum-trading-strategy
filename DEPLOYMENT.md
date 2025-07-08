# 🚀 Deployment Guide

**Momentum Trading Strategy - Production Deployment**

Author: Rishabh Ashok Patil  
Version: 2.0.0  
Last Updated: January 2025

---

## 📋 **Prerequisites**

### **System Requirements**
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection for market data

### **Software Requirements**
```bash
Docker 20.10+
Docker Compose 2.0+
Git 2.30+
PostgreSQL 12+ (if not using Docker)
Node.js 18+ (for development)
Python 3.8+ (for development)
```

---

## 🐳 **Docker Deployment (Recommended)**

### **1. Quick Start**
```bash
# Clone repository
git clone https://github.com/yourusername/momentum-trading-strategy.git
cd momentum-trading-strategy

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### **2. Environment Configuration**
Create `.env` file:
```bash
# Database Configuration
DB_PASSWORD=your_secure_password_here
DB_HOST=postgres
DB_PORT=5432
DB_NAME=momentum_strategy
DB_USER=postgres

# Redis Configuration
REDIS_PASSWORD=your_redis_password_here

# Monitoring Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=your_grafana_password

# Email Alerts (Optional)
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
TO_EMAILS=alerts@yourcompany.com

# Slack Alerts (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# SSL Configuration (Production)
SSL_CERT_PATH=/path/to/ssl/cert.pem
SSL_KEY_PATH=/path/to/ssl/private.key
```

### **3. Service Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │   React         │    │   Python        │
│   Load Balancer │────│   Frontend      │────│   Backend       │
│   Port: 80/443  │    │   Port: 3000    │    │   Port: 9000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   WebSocket     │              │
         └──────────────│   Real-time API │──────────────┘
                        │   Port: 9001    │
                        └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis         │    │   Monitoring    │
│   Database      │    │   Cache         │    │   Stack         │
│   Port: 5432    │    │   Port: 6379    │    │   Various Ports │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🔧 **Manual Deployment**

### **1. Database Setup**
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE momentum_strategy;
CREATE USER momentum_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE momentum_strategy TO momentum_user;
\q

# Test connection
psql -h localhost -U momentum_user -d momentum_strategy
```

### **2. Backend Setup**
```bash
# Clone and setup Python environment
git clone https://github.com/yourusername/momentum-trading-strategy.git
cd momentum-trading-strategy

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=momentum_strategy
export DB_USER=momentum_user
export DB_PASSWORD=your_password

# Initialize database
cd momentum_strategy
python -c "from database import DatabaseManager; DatabaseManager().create_tables()"

# Start API server
python simple_api.py &

# Start WebSocket server
python realtime_api.py &
```

### **3. Frontend Setup**
```bash
# Setup React application
cd dashboard
npm install
npm run build

# Serve with nginx or serve
npm install -g serve
serve -s build -l 3000
```

---

## 📊 **Monitoring Setup**

### **1. Prometheus Metrics**
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'momentum-strategy'
    static_configs:
      - targets: ['strategy_backend:9000']
  
  - job_name: 'system-metrics'
    static_configs:
      - targets: ['node-exporter:9100']
```

### **2. Grafana Dashboards**
Access Grafana at `http://localhost:3001`
- Username: admin
- Password: (from .env file)

Import dashboard templates from `monitoring/grafana/dashboards/`

### **3. Log Aggregation**
```bash
# View aggregated logs
docker-compose logs -f strategy_backend
docker-compose logs -f frontend
docker-compose logs -f postgres

# Access Kibana for log analysis
# http://localhost:5601
```

---

## 🔒 **Security Configuration**

### **1. SSL/TLS Setup**
```bash
# Generate SSL certificates (Let's Encrypt)
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com

# Update nginx configuration
# Copy certificates to ssl/ directory
```

### **2. Firewall Configuration**
```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw deny 5432   # PostgreSQL (internal only)
sudo ufw deny 6379   # Redis (internal only)
```

### **3. Database Security**
```sql
-- Create read-only user for monitoring
CREATE USER monitoring_user WITH PASSWORD 'monitoring_password';
GRANT CONNECT ON DATABASE momentum_strategy TO monitoring_user;
GRANT USAGE ON SCHEMA public TO monitoring_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring_user;
```

---

## 🚀 **Production Deployment**

### **1. AWS Deployment**
```bash
# Using AWS ECS with Fargate
aws ecs create-cluster --cluster-name momentum-strategy

# Deploy using CloudFormation template
aws cloudformation deploy \
  --template-file aws-infrastructure.yml \
  --stack-name momentum-strategy-stack \
  --capabilities CAPABILITY_IAM
```

### **2. Google Cloud Deployment**
```bash
# Using Google Cloud Run
gcloud run deploy momentum-backend \
  --image gcr.io/PROJECT_ID/momentum-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

gcloud run deploy momentum-frontend \
  --image gcr.io/PROJECT_ID/momentum-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### **3. Azure Deployment**
```bash
# Using Azure Container Instances
az container create \
  --resource-group momentum-strategy-rg \
  --name momentum-backend \
  --image youracr.azurecr.io/momentum-backend:latest \
  --dns-name-label momentum-backend \
  --ports 9000
```

---

## 📈 **Performance Optimization**

### **1. Database Optimization**
```sql
-- Create indexes for better performance
CREATE INDEX idx_strategy_runs_created_at ON strategy_runs(created_at);
CREATE INDEX idx_performance_metrics_run_id_date ON performance_metrics(run_id, metric_date);
CREATE INDEX idx_portfolio_positions_date ON portfolio_positions(position_date);

-- Configure PostgreSQL for performance
-- In postgresql.conf:
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

### **2. Application Optimization**
```python
# Enable Redis caching
REDIS_CACHE_TTL = 300  # 5 minutes
REDIS_MAX_CONNECTIONS = 20

# Configure connection pooling
DATABASE_POOL_SIZE = 10
DATABASE_MAX_OVERFLOW = 20
```

### **3. Frontend Optimization**
```javascript
// Enable service worker for caching
// Build with production optimizations
npm run build

// Enable gzip compression in nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript;
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

**1. Database Connection Failed**
```bash
# Check database status
docker-compose ps postgres
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U postgres -d momentum_strategy
```

**2. API Not Responding**
```bash
# Check backend logs
docker-compose logs strategy_backend

# Test API endpoint
curl http://localhost:9000/api/health
```

**3. Frontend Not Loading**
```bash
# Check frontend logs
docker-compose logs frontend

# Rebuild frontend
docker-compose build frontend
docker-compose up -d frontend
```

**4. High Memory Usage**
```bash
# Monitor resource usage
docker stats

# Restart services
docker-compose restart
```

---

## 📞 **Support**

For deployment issues or questions:

- **Author**: Rishabh Ashok Patil
- **Documentation**: See README.md and API documentation
- **Logs**: Check `logs/` directory for detailed error information
- **Monitoring**: Access Grafana dashboard for system metrics

---

## 🔄 **Maintenance**

### **Regular Tasks**
```bash
# Daily backup
./scripts/backup.sh

# Weekly log rotation
docker-compose exec strategy_backend logrotate /etc/logrotate.conf

# Monthly security updates
docker-compose pull
docker-compose up -d --build

# Quarterly performance review
python run_tests.py --performance
```

### **Scaling**
```bash
# Scale backend services
docker-compose up -d --scale strategy_backend=3

# Add load balancer configuration
# Update nginx.conf with upstream servers
```

---

**🎯 Deployment Complete!**

Your momentum trading strategy is now running in production with:
- ✅ High availability architecture
- ✅ Comprehensive monitoring
- ✅ Automated backups
- ✅ Security best practices
- ✅ Performance optimization

**Author: Rishabh Ashok Patil**
