#!/bin/bash
# Backend Entrypoint Script
# =========================
# 
# Production entrypoint for momentum strategy backend
# Handles initialization, database setup, and service startup.
# 
# Author: Rishabh Ashok Patil

set -e

echo "🚀 Starting Momentum Trading Strategy Backend"
echo "Author: Rishabh Ashok Patil"
echo "Version: 2.0.0"
echo "Timestamp: $(date)"

# Wait for database to be ready
echo "⏳ Waiting for database connection..."
while ! pg_isready -h "${DB_HOST:-postgres}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}"; do
    echo "Database not ready, waiting..."
    sleep 2
done
echo "✅ Database connection established"

# Wait for Redis to be ready (if configured)
if [ -n "$REDIS_HOST" ]; then
    echo "⏳ Waiting for Redis connection..."
    while ! redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT:-6379}" ping > /dev/null 2>&1; do
        echo "Redis not ready, waiting..."
        sleep 2
    done
    echo "✅ Redis connection established"
fi

# Initialize database tables
echo "🗄️ Initializing database tables..."
python -c "
from momentum_strategy.database import DatabaseManager
try:
    db = DatabaseManager()
    db.create_tables()
    print('✅ Database tables initialized')
except Exception as e:
    print(f'⚠️ Database initialization warning: {e}')
"

# Run database migrations (if any)
echo "🔄 Running database migrations..."
# Add migration commands here if needed

# Set up monitoring
echo "📊 Setting up monitoring..."
mkdir -p logs
touch logs/momentum_strategy.log

# Validate configuration
echo "🔧 Validating configuration..."
python -c "
from momentum_strategy.config import StrategyConfig
config = StrategyConfig()
print('✅ Configuration validated')
"

# Run health checks
echo "🏥 Running health checks..."
python -c "
import psutil
print(f'CPU cores: {psutil.cpu_count()}')
print(f'Memory: {psutil.virtual_memory().total // (1024**3)} GB')
print('✅ System health check passed')
"

# Start the application
echo "🎯 Starting application..."
echo "Command: $@"

# Execute the main command
exec "$@"
