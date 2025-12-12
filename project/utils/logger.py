"""
Logging utilities for the AI Security System
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json
import sqlite3
from typing import Dict, Any, Optional

class DatabaseHandler(logging.Handler):
    """Custom logging handler that writes to SQLite database"""

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """Create logs table if it doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    logger TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    function TEXT,
                    line INTEGER,
                    extra TEXT
                )
            ''')

    def emit(self, record):
        """Emit log record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO logs (timestamp, level, logger, message, module, function, line, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.fromtimestamp(record.created).isoformat(),
                    record.levelname,
                    record.name,
                    record.getMessage(),
                    record.module,
                    record.funcName,
                    record.lineno,
                    json.dumps(getattr(record, '__dict__', {}))
                ))
        except Exception:
            self.handleError(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry)

def setup_logger(name: str, level: str = 'INFO', log_to_file: bool = True,
                log_to_db: bool = True, db_path: str = 'logs/security_logs.db') -> logging.Logger:
    """
    Setup logger with multiple handlers

    Args:
        name: Logger name
        level: Logging level
        log_to_file: Whether to log to file
        log_to_db: Whether to log to database
        db_path: Database path for logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f'{name}.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Database handler
    if log_to_db:
        db_log_dir = Path(db_path).parent
        db_log_dir.mkdir(exist_ok=True)

        db_handler = DatabaseHandler(db_path)
        logger.addHandler(db_handler)

    return logger

def log_alert(logger: logging.Logger, alert_type: str, message: str,
              confidence: float = None, location: tuple = None,
              person_id: int = None, extra_data: Dict[str, Any] = None):
    """
    Log security alert with structured data

    Args:
        logger: Logger instance
        alert_type: Type of alert (theft, fighting, etc.)
        message: Alert message
        confidence: Detection confidence
        location: (x, y, w, h) bounding box
        person_id: Person tracking ID
        extra_data: Additional data
    """
    alert_data = {
        'alert_type': alert_type,
        'confidence': confidence,
        'location': location,
        'person_id': person_id,
    }

    if extra_data:
        alert_data.update(extra_data)

    logger.warning(message, extra={'extra_data': alert_data})

def get_logs_from_db(db_path: str, limit: int = 100,
                    alert_type: str = None) -> list:
    """
    Retrieve logs from database

    Args:
        db_path: Database path
        limit: Maximum number of logs to retrieve
        alert_type: Filter by alert type

    Returns:
        List of log entries
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM logs WHERE 1=1"
        params = []

        if alert_type:
            query += " AND json_extract(extra, '$.alert_type') = ?"
            params.append(alert_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

def export_logs_to_csv(db_path: str, output_path: str,
                      alert_type: str = None):
    """
    Export logs to CSV file

    Args:
        db_path: Database path
        output_path: Output CSV path
        alert_type: Filter by alert type
    """
    import csv

    logs = get_logs_from_db(db_path, limit=10000, alert_type=alert_type)

    if not logs:
        return

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = logs[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)
