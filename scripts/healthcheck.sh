#!/bin/bash
# Health Check Script
# ===================
# 
# Comprehensive health check for momentum strategy backend
# Checks API endpoints, database connectivity, and system resources.
# 
# Author: Rishabh Ashok Patil

set -e

# Configuration
API_HOST=${API_HOST:-localhost}
API_PORT=${API_PORT:-9000}
DB_HOST=${DB_HOST:-postgres}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-postgres}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check API health endpoint
check_api_health() {
    log "Checking API health endpoint..."
    
    if curl -f -s "http://${API_HOST}:${API_PORT}/api/health" > /dev/null; then
        log "✅ API health check passed"
        return 0
    else
        error "❌ API health check failed"
        return 1
    fi
}

# Check database connectivity
check_database() {
    log "Checking database connectivity..."
    
    if pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" > /dev/null 2>&1; then
        log "✅ Database connectivity check passed"
        return 0
    else
        error "❌ Database connectivity check failed"
        return 1
    fi
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Check memory usage
    memory_usage=$(python3 -c "
import psutil
mem = psutil.virtual_memory()
print(mem.percent)
")
    
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        warning "⚠️ High memory usage: ${memory_usage}%"
    else
        log "✅ Memory usage OK: ${memory_usage}%"
    fi
    
    # Check disk usage
    disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -gt 90 ]; then
        warning "⚠️ High disk usage: ${disk_usage}%"
    else
        log "✅ Disk usage OK: ${disk_usage}%"
    fi
    
    return 0
}

# Check log files
check_logs() {
    log "Checking log files..."
    
    if [ -f "logs/momentum_strategy.log" ]; then
        # Check for recent errors in logs
        recent_errors=$(tail -n 100 logs/momentum_strategy.log | grep -i error | wc -l)
        
        if [ "$recent_errors" -gt 5 ]; then
            warning "⚠️ Found ${recent_errors} recent errors in logs"
        else
            log "✅ Log file check passed"
        fi
    else
        warning "⚠️ Log file not found"
    fi
    
    return 0
}

# Main health check function
main() {
    log "🏥 Starting health check..."
    log "Author: Rishabh Ashok Patil"
    
    local exit_code=0
    
    # Run all checks
    check_api_health || exit_code=1
    check_database || exit_code=1
    check_system_resources || exit_code=1
    check_logs || exit_code=1
    
    if [ $exit_code -eq 0 ]; then
        log "✅ All health checks passed"
    else
        error "❌ Some health checks failed"
    fi
    
    return $exit_code
}

# Run main function
main "$@"
