"""
Monitoring and Alerting System
==============================

Comprehensive monitoring system for momentum strategy with metrics collection,
alerting, logging, and performance tracking.

Author: Rishabh Ashok Patil
"""

import os
import time
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/momentum_strategy.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict
    process_count: int
    
class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.alert_handlers = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Thresholds for alerts
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'strategy_drawdown': -15.0,
            'sharpe_ratio_min': 0.5,
            'max_position_weight': 0.3
        }
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
    
    def start_monitoring(self, interval: int = 60):
        """Start system monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Performance monitoring started (interval: {interval}s)")
        logger.info("Author: Rishabh Ashok Patil")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics (about 16 hours at 1min interval)
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Check for alerts
                self._check_system_alerts(metrics)
                
                # Sleep until next collection
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk_percent,
                network_io=network_io,
                process_count=process_count
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against thresholds"""
        if not metrics:
            return
        
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(Alert(
                id=f"cpu_high_{int(time.time())}",
                level="WARNING",
                title="High CPU Usage",
                message=f"CPU usage is {metrics.cpu_percent:.1f}% (threshold: {self.thresholds['cpu_percent']}%)",
                timestamp=metrics.timestamp,
                source="system_monitor",
                metadata={'cpu_percent': metrics.cpu_percent}
            ))
        
        # Memory usage alert
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(Alert(
                id=f"memory_high_{int(time.time())}",
                level="WARNING",
                title="High Memory Usage",
                message=f"Memory usage is {metrics.memory_percent:.1f}% (threshold: {self.thresholds['memory_percent']}%)",
                timestamp=metrics.timestamp,
                source="system_monitor",
                metadata={'memory_percent': metrics.memory_percent}
            ))
        
        # Disk usage alert
        if metrics.disk_usage_percent > self.thresholds['disk_usage_percent']:
            alerts.append(Alert(
                id=f"disk_high_{int(time.time())}",
                level="ERROR",
                title="High Disk Usage",
                message=f"Disk usage is {metrics.disk_usage_percent:.1f}% (threshold: {self.thresholds['disk_usage_percent']}%)",
                timestamp=metrics.timestamp,
                source="system_monitor",
                metadata={'disk_usage_percent': metrics.disk_usage_percent}
            ))
        
        # Process alerts
        for alert in alerts:
            self.add_alert(alert)
    
    def add_alert(self, alert: Alert):
        """Add alert and trigger handlers"""
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Log alert
        log_level = getattr(logging, alert.level, logging.INFO)
        logger.log(log_level, f"ALERT: {alert.title} - {alert.message}")
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def get_recent_metrics(self, hours: int = 1) -> List[SystemMetrics]:
        """Get metrics from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a.timestamp >= cutoff]
    
    def get_system_health_summary(self) -> Dict:
        """Get current system health summary"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available"}
        
        latest = self.metrics_history[-1]
        recent_alerts = self.get_recent_alerts(hours=1)
        
        # Determine overall health status
        critical_alerts = [a for a in recent_alerts if a.level == "CRITICAL"]
        error_alerts = [a for a in recent_alerts if a.level == "ERROR"]
        warning_alerts = [a for a in recent_alerts if a.level == "WARNING"]
        
        if critical_alerts:
            status = "critical"
        elif error_alerts:
            status = "error"
        elif warning_alerts:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": latest.timestamp.isoformat(),
            "metrics": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_usage_percent": latest.disk_usage_percent,
                "process_count": latest.process_count
            },
            "alerts_last_hour": {
                "critical": len(critical_alerts),
                "error": len(error_alerts),
                "warning": len(warning_alerts),
                "info": len([a for a in recent_alerts if a.level == "INFO"])
            },
            "author": "Rishabh Ashok Patil"
        }

class StrategyMonitor:
    """Monitor strategy-specific metrics and performance"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.strategy_metrics = {}
        
    def monitor_strategy_performance(self, strategy_results: Dict):
        """Monitor strategy performance and generate alerts"""
        try:
            if not strategy_results:
                return
            
            # Extract key metrics
            performance = strategy_results.get('performance', {})
            risk_metrics = strategy_results.get('risk_metrics', {})
            
            # Check drawdown alert
            max_drawdown = performance.get('max_drawdown', 0)
            if max_drawdown < self.performance_monitor.thresholds['strategy_drawdown']:
                alert = Alert(
                    id=f"drawdown_high_{int(time.time())}",
                    level="ERROR",
                    title="High Strategy Drawdown",
                    message=f"Strategy drawdown is {max_drawdown:.2f}% (threshold: {self.performance_monitor.thresholds['strategy_drawdown']}%)",
                    timestamp=datetime.now(),
                    source="strategy_monitor",
                    metadata={'max_drawdown': max_drawdown}
                )
                self.performance_monitor.add_alert(alert)
            
            # Check Sharpe ratio alert
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            if sharpe_ratio < self.performance_monitor.thresholds['sharpe_ratio_min']:
                alert = Alert(
                    id=f"sharpe_low_{int(time.time())}",
                    level="WARNING",
                    title="Low Sharpe Ratio",
                    message=f"Strategy Sharpe ratio is {sharpe_ratio:.2f} (threshold: {self.performance_monitor.thresholds['sharpe_ratio_min']})",
                    timestamp=datetime.now(),
                    source="strategy_monitor",
                    metadata={'sharpe_ratio': sharpe_ratio}
                )
                self.performance_monitor.add_alert(alert)
            
            # Store metrics
            self.strategy_metrics = {
                'timestamp': datetime.now().isoformat(),
                'performance': performance,
                'risk_metrics': risk_metrics
            }
            
        except Exception as e:
            logger.error(f"Error monitoring strategy performance: {e}")
    
    def monitor_portfolio_positions(self, positions: pd.DataFrame):
        """Monitor portfolio positions for concentration risk"""
        try:
            if positions.empty:
                return
            
            # Get latest positions
            latest_positions = positions.iloc[-1]
            max_weight = latest_positions.max()
            
            # Check concentration alert
            if max_weight > self.performance_monitor.thresholds['max_position_weight']:
                alert = Alert(
                    id=f"concentration_high_{int(time.time())}",
                    level="WARNING",
                    title="High Position Concentration",
                    message=f"Maximum position weight is {max_weight:.2f}% (threshold: {self.performance_monitor.thresholds['max_position_weight']*100}%)",
                    timestamp=datetime.now(),
                    source="portfolio_monitor",
                    metadata={'max_weight': max_weight}
                )
                self.performance_monitor.add_alert(alert)
            
        except Exception as e:
            logger.error(f"Error monitoring portfolio positions: {e}")

class AlertNotifier:
    """Handle alert notifications via email, Slack, etc."""
    
    def __init__(self):
        self.email_config = self._load_email_config()
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
    
    def _load_email_config(self) -> Dict:
        """Load email configuration from environment"""
        return {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'from_email': os.getenv('FROM_EMAIL'),
            'to_emails': os.getenv('TO_EMAILS', '').split(',')
        }
    
    def send_email_alert(self, alert: Alert):
        """Send alert via email"""
        try:
            if not all([self.email_config['username'], self.email_config['password'], 
                       self.email_config['from_email']]):
                logger.warning("Email configuration incomplete, skipping email alert")
                return
            
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"[{alert.level}] {alert.title}"
            
            body = f"""
            Alert Details:
            
            Level: {alert.level}
            Title: {alert.title}
            Message: {alert.message}
            Source: {alert.source}
            Timestamp: {alert.timestamp}
            
            Metadata: {json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}
            
            ---
            Momentum Trading Strategy Alert System
            Author: Rishabh Ashok Patil
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        try:
            if not self.slack_webhook:
                logger.warning("Slack webhook not configured, skipping Slack alert")
                return
            
            color_map = {
                'INFO': '#36a64f',
                'WARNING': '#ff9500',
                'ERROR': '#ff0000',
                'CRITICAL': '#8b0000'
            }
            
            payload = {
                'attachments': [{
                    'color': color_map.get(alert.level, '#36a64f'),
                    'title': f"[{alert.level}] {alert.title}",
                    'text': alert.message,
                    'fields': [
                        {'title': 'Source', 'value': alert.source, 'short': True},
                        {'title': 'Timestamp', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                    ],
                    'footer': 'Momentum Strategy Monitor | Rishabh Ashok Patil',
                    'ts': int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

# Global monitoring instance
_monitor_instance = None

def get_monitor() -> PerformanceMonitor:
    """Get global monitoring instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance

def setup_monitoring(email_alerts: bool = False, slack_alerts: bool = False) -> PerformanceMonitor:
    """Set up monitoring with optional alert handlers"""
    monitor = get_monitor()
    
    # Set up alert notifier
    if email_alerts or slack_alerts:
        notifier = AlertNotifier()
        
        if email_alerts:
            monitor.add_alert_handler(notifier.send_email_alert)
        
        if slack_alerts:
            monitor.add_alert_handler(notifier.send_slack_alert)
    
    # Start monitoring
    monitor.start_monitoring()
    
    logger.info("Monitoring system initialized")
    logger.info("Author: Rishabh Ashok Patil")
    
    return monitor
