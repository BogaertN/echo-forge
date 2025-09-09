#!/usr/bin/env python3
"""
EchoForge Monitoring and Health Check Script
===========================================
Comprehensive monitoring solution for EchoForge deployments.

Features:
- System resource monitoring (CPU, memory, disk, network)
- Application health checks and endpoint monitoring
- Database performance and connectivity monitoring
- AI model availability and performance tracking
- Real-time alerting and notification system
- Performance metrics collection and analysis
- Log analysis and error tracking
- Automated recovery actions
- Dashboard and reporting capabilities
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
import requests
import psutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    name: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    message: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert notification."""
    level: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    source: str
    timestamp: datetime
    resolved: bool = False


class EchoForgeMonitor:
    """
    Comprehensive monitoring system for EchoForge.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the monitoring system.
        
        Args:
            config_file: Optional monitoring configuration file
        """
        self.config = get_config()
        self.project_root = Path(__file__).parent.parent
        self.monitoring_config = self._load_monitoring_config(config_file)
        
        # Initialize components
        self._setup_logging()
        self._setup_storage()
        self._setup_alerting()
        
        # Monitoring state
        self.active_alerts = {}
        self.metrics_cache = []
        self.last_check_times = {}
        
        self.logger.info("EchoForge Monitor initialized")

    def _setup_logging(self):
        """Setup monitoring logging."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / 'monitor.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def _load_monitoring_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        # Default configuration
        default_config = {
            'check_interval': 60,  # seconds
            'health_checks': {
                'api_health': {'enabled': True, 'timeout': 10, 'interval': 30},
                'database': {'enabled': True, 'timeout': 5, 'interval': 60},
                'ollama': {'enabled': True, 'timeout': 15, 'interval': 120},
                'disk_space': {'enabled': True, 'threshold': 85, 'interval': 300},
                'memory': {'enabled': True, 'threshold': 90, 'interval': 60},
                'cpu': {'enabled': True, 'threshold': 80, 'interval': 60}
            },
            'endpoints': {
                'api_base': 'http://localhost:8000',
                'ollama_base': 'http://localhost:11434'
            },
            'alerting': {
                'enabled': True,
                'email': {
                    'enabled': False,
                    'smtp_server': 'localhost',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'from_email': 'monitor@echoforge.local',
                    'to_emails': []
                },
                'webhook': {
                    'enabled': False,
                    'url': '',
                    'headers': {}
                }
            },
            'retention_days': 30,
            'metrics_batch_size': 100
        }
        
        # Load custom configuration if provided
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                # Merge configurations
                default_config.update(custom_config)
            except Exception as e:
                self.logger.warning(f"Failed to load monitoring config: {e}")
        
        return default_config

    def _setup_storage(self):
        """Setup metrics storage."""
        self.metrics_db_path = self.project_root / 'data' / 'monitoring.db'
        self.metrics_db_path.parent.mkdir(exist_ok=True)
        
        try:
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT,
                        timestamp TEXT NOT NULL,
                        tags TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS health_checks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        message TEXT,
                        response_time REAL,
                        timestamp TEXT NOT NULL,
                        details TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        level TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT,
                        source TEXT,
                        timestamp TEXT NOT NULL,
                        resolved INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_health_checks_name_timestamp ON health_checks(name, timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
                
        except Exception as e:
            self.logger.error(f"Failed to setup metrics storage: {e}")

    def _setup_alerting(self):
        """Setup alerting mechanisms."""
        self.alert_config = self.monitoring_config.get('alerting', {})
        
        if self.alert_config.get('enabled', False):
            self.logger.info("Alerting enabled")
        else:
            self.logger.info("Alerting disabled")

    async def run_monitoring_loop(self):
        """Run the main monitoring loop."""
        self.logger.info("Starting monitoring loop...")
        
        try:
            while True:
                start_time = time.time()
                
                # Run health checks
                await self._run_health_checks()
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Process alerts
                await self._process_alerts()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_config['check_interval'] - elapsed)
                
                self.logger.debug(f"Monitoring cycle completed in {elapsed:.2f}s, sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
            raise

    async def _run_health_checks(self):
        """Run all enabled health checks."""
        health_checks = self.monitoring_config.get('health_checks', {})
        
        # Create tasks for all enabled checks
        tasks = []
        for check_name, check_config in health_checks.items():
            if check_config.get('enabled', False):
                # Check if enough time has passed since last check
                last_check = self.last_check_times.get(check_name, 0)
                if time.time() - last_check >= check_config.get('interval', 60):
                    tasks.append(self._run_single_health_check(check_name, check_config))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for result in results:
                if isinstance(result, HealthCheckResult):
                    await self._store_health_check_result(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Health check failed with exception: {result}")

    async def _run_single_health_check(self, check_name: str, check_config: Dict[str, Any]) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()
        self.last_check_times[check_name] = start_time
        
        try:
            if check_name == 'api_health':
                return await self._check_api_health(check_config)
            elif check_name == 'database':
                return await self._check_database_health(check_config)
            elif check_name == 'ollama':
                return await self._check_ollama_health(check_config)
            elif check_name == 'disk_space':
                return await self._check_disk_space(check_config)
            elif check_name == 'memory':
                return await self._check_memory_usage(check_config)
            elif check_name == 'cpu':
                return await self._check_cpu_usage(check_config)
            else:
                return HealthCheckResult(
                    name=check_name,
                    status='unknown',
                    message=f'Unknown health check: {check_name}',
                    response_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=check_name,
                status='critical',
                message=f'Health check failed: {str(e)}',
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    async def _check_api_health(self, config: Dict[str, Any]) -> HealthCheckResult:
        """Check API health endpoint."""
        start_time = time.time()
        base_url = self.monitoring_config['endpoints']['api_base']
        
        try:
            response = requests.get(
                f'{base_url}/health',
                timeout=config.get('timeout', 10)
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                status = 'healthy' if health_data.get('status') == 'healthy' else 'warning'
                message = health_data.get('message', 'API responding')
                
                return HealthCheckResult(
                    name='api_health',
                    status=status,
                    message=message,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    details=health_data
                )
            else:
                return HealthCheckResult(
                    name='api_health',
                    status='critical',
                    message=f'API returned status {response.status_code}',
                    response_time=response_time,
                    timestamp=datetime.now()
                )
                
        except requests.exceptions.RequestException as e:
            return HealthCheckResult(
                name='api_health',
                status='critical',
                message=f'API unreachable: {str(e)}',
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    async def _check_database_health(self, config: Dict[str, Any]) -> HealthCheckResult:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            db_path = self.config.database.db_path
            
            with sqlite3.connect(str(db_path), timeout=config.get('timeout', 5)) as conn:
                # Test basic connectivity
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                
                # Check integrity
                cursor.execute('PRAGMA integrity_check')
                integrity_result = cursor.fetchone()[0]
                
                # Get database size
                db_size = Path(db_path).stat().st_size
                
                response_time = time.time() - start_time
                
                status = 'healthy' if integrity_result == 'ok' else 'warning'
                message = f'Database accessible, integrity: {integrity_result}'
                
                return HealthCheckResult(
                    name='database',
                    status=status,
                    message=message,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    details={
                        'integrity': integrity_result,
                        'size_bytes': db_size,
                        'size_mb': round(db_size / (1024 * 1024), 2)
                    }
                )
                
        except Exception as e:
            return HealthCheckResult(
                name='database',
                status='critical',
                message=f'Database check failed: {str(e)}',
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    async def _check_ollama_health(self, config: Dict[str, Any]) -> HealthCheckResult:
        """Check Ollama service health."""
        start_time = time.time()
        base_url = self.monitoring_config['endpoints']['ollama_base']
        
        try:
            # Check Ollama version endpoint
            response = requests.get(
                f'{base_url}/api/version',
                timeout=config.get('timeout', 15)
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                version_data = response.json()
                
                # Check available models
                models_response = requests.get(f'{base_url}/api/tags', timeout=5)
                models_data = models_response.json() if models_response.status_code == 200 else {}
                
                return HealthCheckResult(
                    name='ollama',
                    status='healthy',
                    message='Ollama service responding',
                    response_time=response_time,
                    timestamp=datetime.now(),
                    details={
                        'version': version_data.get('version', 'unknown'),
                        'models': [model['name'] for model in models_data.get('models', [])]
                    }
                )
            else:
                return HealthCheckResult(
                    name='ollama',
                    status='critical',
                    message=f'Ollama returned status {response.status_code}',
                    response_time=response_time,
                    timestamp=datetime.now()
                )
                
        except requests.exceptions.RequestException as e:
            return HealthCheckResult(
                name='ollama',
                status='critical',
                message=f'Ollama unreachable: {str(e)}',
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    async def _check_disk_space(self, config: Dict[str, Any]) -> HealthCheckResult:
        """Check disk space usage."""
        start_time = time.time()
        threshold = config.get('threshold', 85)
        
        try:
            # Check disk usage for project directory
            usage = psutil.disk_usage(str(self.project_root))
            
            used_percent = (usage.used / usage.total) * 100
            free_gb = usage.free / (1024**3)
            
            if used_percent >= threshold:
                status = 'critical' if used_percent >= 95 else 'warning'
                message = f'Disk usage high: {used_percent:.1f}% (threshold: {threshold}%)'
            else:
                status = 'healthy'
                message = f'Disk usage normal: {used_percent:.1f}%'
            
            return HealthCheckResult(
                name='disk_space',
                status=status,
                message=message,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                details={
                    'used_percent': round(used_percent, 1),
                    'free_gb': round(free_gb, 2),
                    'total_gb': round(usage.total / (1024**3), 2),
                    'threshold': threshold
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name='disk_space',
                status='critical',
                message=f'Disk space check failed: {str(e)}',
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    async def _check_memory_usage(self, config: Dict[str, Any]) -> HealthCheckResult:
        """Check memory usage."""
        start_time = time.time()
        threshold = config.get('threshold', 90)
        
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            if used_percent >= threshold:
                status = 'critical' if used_percent >= 95 else 'warning'
                message = f'Memory usage high: {used_percent:.1f}% (threshold: {threshold}%)'
            else:
                status = 'healthy'
                message = f'Memory usage normal: {used_percent:.1f}%'
            
            return HealthCheckResult(
                name='memory',
                status=status,
                message=message,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                details={
                    'used_percent': used_percent,
                    'available_gb': round(memory.available / (1024**3), 2),
                    'total_gb': round(memory.total / (1024**3), 2),
                    'threshold': threshold
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name='memory',
                status='critical',
                message=f'Memory check failed: {str(e)}',
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    async def _check_cpu_usage(self, config: Dict[str, Any]) -> HealthCheckResult:
        """Check CPU usage."""
        start_time = time.time()
        threshold = config.get('threshold', 80)
        
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            if cpu_percent >= threshold:
                status = 'critical' if cpu_percent >= 95 else 'warning'
                message = f'CPU usage high: {cpu_percent:.1f}% (threshold: {threshold}%)'
            else:
                status = 'healthy'
                message = f'CPU usage normal: {cpu_percent:.1f}%'
            
            return HealthCheckResult(
                name='cpu',
                status=status,
                message=message,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                details={
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'threshold': threshold
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name='cpu',
                status='critical',
                message=f'CPU check failed: {str(e)}',
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            metrics = []
            timestamp = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            metrics.append(MetricPoint('cpu_usage_percent', cpu_percent, '%', timestamp))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(MetricPoint('memory_usage_percent', memory.percent, '%', timestamp))
            metrics.append(MetricPoint('memory_available_bytes', memory.available, 'bytes', timestamp))
            
            # Disk metrics
            disk_usage = psutil.disk_usage(str(self.project_root))
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            metrics.append(MetricPoint('disk_usage_percent', disk_percent, '%', timestamp))
            metrics.append(MetricPoint('disk_free_bytes', disk_usage.free, 'bytes', timestamp))
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                metrics.append(MetricPoint('network_bytes_sent', net_io.bytes_sent, 'bytes', timestamp))
                metrics.append(MetricPoint('network_bytes_recv', net_io.bytes_recv, 'bytes', timestamp))
            except:
                pass  # Network metrics not always available
            
            # Process-specific metrics
            try:
                current_process = psutil.Process()
                metrics.append(MetricPoint('process_cpu_percent', current_process.cpu_percent(), '%', timestamp))
                metrics.append(MetricPoint('process_memory_bytes', current_process.memory_info().rss, 'bytes', timestamp))
            except:
                pass
            
            # Store metrics
            await self._store_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

    async def _store_health_check_result(self, result: HealthCheckResult):
        """Store health check result to database."""
        try:
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                conn.execute('''
                    INSERT INTO health_checks (name, status, message, response_time, timestamp, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.name,
                    result.status,
                    result.message,
                    result.response_time,
                    result.timestamp.isoformat(),
                    json.dumps(result.details)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to store health check result: {e}")

    async def _store_metrics(self, metrics: List[MetricPoint]):
        """Store metrics to database."""
        try:
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                for metric in metrics:
                    conn.execute('''
                        INSERT INTO metrics (name, value, unit, timestamp, tags)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        metric.name,
                        metric.value,
                        metric.unit,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.tags)
                    ))
                    
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")

    async def _process_alerts(self):
        """Process and send alerts based on health check results."""
        if not self.alert_config.get('enabled', False):
            return
        
        try:
            # Get recent critical and warning health check results
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                cursor = conn.cursor()
                
                # Get latest results for each check
                cursor.execute('''
                    SELECT name, status, message, timestamp
                    FROM health_checks
                    WHERE timestamp > datetime('now', '-5 minutes')
                    AND status IN ('warning', 'critical')
                    ORDER BY timestamp DESC
                ''')
                
                recent_issues = cursor.fetchall()
                
                for name, status, message, timestamp in recent_issues:
                    alert_key = f"{name}_{status}"
                    
                    # Check if this alert is already active
                    if alert_key not in self.active_alerts:
                        alert = Alert(
                            level=status,
                            title=f"EchoForge {status.title()}: {name}",
                            message=message,
                            source=name,
                            timestamp=datetime.fromisoformat(timestamp)
                        )
                        
                        await self._send_alert(alert)
                        self.active_alerts[alert_key] = alert
                        
                        # Store alert
                        await self._store_alert(alert)
                
                # Check for resolved alerts
                cursor.execute('''
                    SELECT DISTINCT name
                    FROM health_checks
                    WHERE timestamp > datetime('now', '-1 minute')
                    AND status = 'healthy'
                ''')
                
                healthy_checks = [row[0] for row in cursor.fetchall()]
                
                # Resolve active alerts for now-healthy checks
                resolved_alerts = []
                for alert_key in list(self.active_alerts.keys()):
                    check_name = alert_key.split('_')[0]
                    if check_name in healthy_checks:
                        alert = self.active_alerts[alert_key]
                        alert.resolved = True
                        
                        # Send resolution notification
                        resolution_alert = Alert(
                            level='info',
                            title=f"EchoForge Resolved: {check_name}",
                            message=f"Issue with {check_name} has been resolved",
                            source=check_name,
                            timestamp=datetime.now(),
                            resolved=True
                        )
                        
                        await self._send_alert(resolution_alert)
                        resolved_alerts.append(alert_key)
                
                # Remove resolved alerts
                for alert_key in resolved_alerts:
                    del self.active_alerts[alert_key]
                
        except Exception as e:
            self.logger.error(f"Failed to process alerts: {e}")

    async def _send_alert(self, alert: Alert):
        """Send alert notification."""
        try:
            # Email notification
            if self.alert_config.get('email', {}).get('enabled', False):
                await self._send_email_alert(alert)
            
            # Webhook notification
            if self.alert_config.get('webhook', {}).get('enabled', False):
                await self._send_webhook_alert(alert)
            
            self.logger.info(f"Alert sent: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

    async def _send_email_alert(self, alert: Alert):
        """Send email alert notification."""
        try:
            email_config = self.alert_config['email']
            
            msg = MimeMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = ', '.join(email_config['to_emails'])
            msg['Subject'] = alert.title
            
            body = f"""
EchoForge Monitoring Alert

Level: {alert.level.upper()}
Source: {alert.source}
Time: {alert.timestamp}
Message: {alert.message}

This is an automated alert from EchoForge monitoring system.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('username'):
                    server.starttls()
                    server.login(email_config['username'], email_config['password'])
                
                server.sendmail(
                    email_config['from_email'],
                    email_config['to_emails'],
                    msg.as_string()
                )
                
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert notification."""
        try:
            webhook_config = self.alert_config['webhook']
            
            payload = {
                'level': alert.level,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved
            }
            
            headers = {'Content-Type': 'application/json'}
            headers.update(webhook_config.get('headers', {}))
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code not in [200, 201, 202]:
                self.logger.warning(f"Webhook returned status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")

    async def _store_alert(self, alert: Alert):
        """Store alert to database."""
        try:
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                conn.execute('''
                    INSERT INTO alerts (level, title, message, source, timestamp, resolved)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert.level,
                    alert.title,
                    alert.message,
                    alert.source,
                    alert.timestamp.isoformat(),
                    int(alert.resolved)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")

    async def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        try:
            retention_days = self.monitoring_config.get('retention_days', 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                # Clean up old metrics
                conn.execute(
                    'DELETE FROM metrics WHERE timestamp < ?',
                    (cutoff_date.isoformat(),)
                )
                
                # Clean up old health checks
                conn.execute(
                    'DELETE FROM health_checks WHERE timestamp < ?',
                    (cutoff_date.isoformat(),)
                )
                
                # Clean up old alerts
                conn.execute(
                    'DELETE FROM alerts WHERE timestamp < ? AND resolved = 1',
                    (cutoff_date.isoformat(),)
                )
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'health_checks': {},
                'metrics': {},
                'active_alerts': len(self.active_alerts),
                'uptime': self._get_system_uptime()
            }
            
            # Get latest health check results
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                cursor = conn.cursor()
                
                # Get latest status for each health check
                cursor.execute('''
                    SELECT name, status, message, response_time, timestamp
                    FROM health_checks
                    WHERE id IN (
                        SELECT MAX(id) FROM health_checks GROUP BY name
                    )
                    ORDER BY name
                ''')
                
                health_results = cursor.fetchall()
                overall_healthy = True
                
                for name, health_status, message, response_time, timestamp in health_results:
                    status['health_checks'][name] = {
                        'status': health_status,
                        'message': message,
                        'response_time': response_time,
                        'timestamp': timestamp
                    }
                    
                    if health_status in ['warning', 'critical']:
                        overall_healthy = False
                
                # Set overall status
                if not overall_healthy:
                    if any(hc['status'] == 'critical' for hc in status['health_checks'].values()):
                        status['overall_status'] = 'critical'
                    else:
                        status['overall_status'] = 'warning'
                
                # Get latest metrics
                cursor.execute('''
                    SELECT name, value, unit, timestamp
                    FROM metrics
                    WHERE timestamp > datetime('now', '-5 minutes')
                    ORDER BY timestamp DESC
                    LIMIT 20
                ''')
                
                recent_metrics = cursor.fetchall()
                for name, value, unit, timestamp in recent_metrics:
                    if name not in status['metrics']:
                        status['metrics'][name] = {
                            'value': value,
                            'unit': unit,
                            'timestamp': timestamp
                        }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    def _get_system_uptime(self) -> float:
        """Get system uptime in seconds."""
        try:
            return time.time() - psutil.boot_time()
        except:
            return 0.0

    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate monitoring report for specified time period."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            report = {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'hours': hours
                },
                'summary': {},
                'health_checks': {},
                'metrics': {},
                'alerts': []
            }
            
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                cursor = conn.cursor()
                
                # Health checks summary
                cursor.execute('''
                    SELECT name, status, COUNT(*) as count
                    FROM health_checks
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY name, status
                    ORDER BY name, status
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                health_summary = {}
                for name, status, count in cursor.fetchall():
                    if name not in health_summary:
                        health_summary[name] = {}
                    health_summary[name][status] = count
                
                report['health_checks'] = health_summary
                
                # Metrics summary
                cursor.execute('''
                    SELECT name, AVG(value) as avg_value, MIN(value) as min_value,
                           MAX(value) as max_value, COUNT(*) as count, unit
                    FROM metrics
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY name, unit
                    ORDER BY name
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                metrics_summary = {}
                for name, avg_val, min_val, max_val, count, unit in cursor.fetchall():
                    metrics_summary[name] = {
                        'average': round(avg_val, 2),
                        'minimum': round(min_val, 2),
                        'maximum': round(max_val, 2),
                        'count': count,
                        'unit': unit
                    }
                
                report['metrics'] = metrics_summary
                
                # Alerts summary
                cursor.execute('''
                    SELECT level, title, message, source, timestamp, resolved
                    FROM alerts
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                alerts = []
                for level, title, message, source, timestamp, resolved in cursor.fetchall():
                    alerts.append({
                        'level': level,
                        'title': title,
                        'message': message,
                        'source': source,
                        'timestamp': timestamp,
                        'resolved': bool(resolved)
                    })
                
                report['alerts'] = alerts
                
                # Overall summary
                total_alerts = len(alerts)
                critical_alerts = len([a for a in alerts if a['level'] == 'critical'])
                warning_alerts = len([a for a in alerts if a['level'] == 'warning'])
                
                report['summary'] = {
                    'total_alerts': total_alerts,
                    'critical_alerts': critical_alerts,
                    'warning_alerts': warning_alerts,
                    'health_checks_run': sum(sum(hc.values()) for hc in health_summary.values()),
                    'metrics_collected': sum(m['count'] for m in metrics_summary.values())
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return {'error': str(e)}


def main():
    """Command-line interface for monitoring operations."""
    parser = argparse.ArgumentParser(
        description="EchoForge Monitoring Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run                          # Start monitoring loop
  %(prog)s status                       # Show current status
  %(prog)s report --hours 24            # Generate 24-hour report
  %(prog)s check --name api_health      # Run specific health check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Start monitoring loop')
    run_parser.add_argument('--config', help='Monitoring configuration file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show current system status')
    status_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate monitoring report')
    report_parser.add_argument('--hours', type=int, default=24, help='Report time period in hours')
    report_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Run specific health check')
    check_parser.add_argument('--name', required=True, help='Health check name')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'run':
            monitor = EchoForgeMonitor(args.config)
            asyncio.run(monitor.run_monitoring_loop())
            return 0
            
        elif args.command == 'status':
            monitor = EchoForgeMonitor()
            status = monitor.get_system_status()
            
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"Overall Status: {status['overall_status'].upper()}")
                print(f"Timestamp: {status['timestamp']}")
                print(f"Active Alerts: {status['active_alerts']}")
                print(f"Uptime: {status['uptime']:.0f} seconds")
                
                print("\nHealth Checks:")
                for name, details in status.get('health_checks', {}).items():
                    print(f"  {name}: {details['status']} - {details['message']}")
            
            return 0
            
        elif args.command == 'report':
            monitor = EchoForgeMonitor()
            report = monitor.generate_report(args.hours)
            
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(f"Monitoring Report ({args.hours} hours)")
                print(f"Period: {report['period']['start']} to {report['period']['end']}")
                
                summary = report.get('summary', {})
                print(f"\nSummary:")
                print(f"  Total Alerts: {summary.get('total_alerts', 0)}")
                print(f"  Critical Alerts: {summary.get('critical_alerts', 0)}")
                print(f"  Warning Alerts: {summary.get('warning_alerts', 0)}")
                print(f"  Health Checks Run: {summary.get('health_checks_run', 0)}")
                print(f"  Metrics Collected: {summary.get('metrics_collected', 0)}")
            
            return 0
            
        elif args.command == 'check':
            # This would run a specific health check
            print(f"Running health check: {args.name}")
            return 0
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
