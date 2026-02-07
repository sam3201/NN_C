#!/usr/bin/env python3
"""
SAM 2.0 Database Layer
Optimized persistent storage using SQLite with performance enhancements
"""

import sqlite3
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import os

class SAMDatabase:
    """High-performance SQLite database for SAM system data"""

    def __init__(self, db_path: str = "sam_data/sam_system.db"):
        self.db_path = db_path
        self._local = threading.local()  # Thread-local storage for connections
        self._lock = threading.RLock()
        self.initialize_database()

    @contextmanager
    def get_connection(self):
        """Thread-safe database connection context manager"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None  # Enable autocommit for performance
            )
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._local.connection.execute("PRAGMA temp_store=MEMORY")
            self._local.connection.row_factory = sqlite3.Row

        try:
            yield self._local.connection
        except Exception:
            self._local.connection.rollback()
            raise
        finally:
            pass  # Connection stays open for thread reuse

    def initialize_database(self):
        """Create database tables and indexes"""
        with self.get_connection() as conn:
            # System metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    uptime REAL,
                    consciousness_score REAL,
                    coherence_score REAL,
                    survival_score REAL,
                    total_conversations INTEGER DEFAULT 0,
                    learning_events INTEGER DEFAULT 0,
                    optimization_events INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    performance_data TEXT
                )
            ''')

            # Tasks and goals table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    critical BOOLEAN DEFAULT 0,
                    priority INTEGER DEFAULT 1,
                    estimated_time INTEGER DEFAULT 60,
                    completed BOOLEAN DEFAULT 0,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    attempts INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    last_attempt REAL,
                    confidence_score REAL DEFAULT 0.5,
                    failure_reasons TEXT,
                    dependencies TEXT,  -- JSON array of task names
                    metadata TEXT      -- JSON additional data
                )
            ''')

            # Survival evaluations cache
            conn.execute('''
                CREATE TABLE IF NOT EXISTS survival_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_hash TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    survival_impact REAL,
                    optionality_impact REAL,
                    risk_level REAL,
                    confidence REAL,
                    reasoning TEXT,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    last_accessed REAL NOT NULL
                )
            ''')

            # Error logs and recovery attempts
            conn.execute('''
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    traceback TEXT,
                    context TEXT,  -- JSON context data
                    recovery_attempted BOOLEAN DEFAULT 0,
                    recovery_success BOOLEAN DEFAULT 0,
                    recovery_actions TEXT,  -- JSON array of actions taken
                    resolved BOOLEAN DEFAULT 0
                )
            ''')

            # Performance metrics
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    endpoint TEXT,
                    method TEXT,
                    response_time REAL,
                    status_code INTEGER,
                    request_size INTEGER,
                    response_size INTEGER,
                    client_ip TEXT,
                    user_agent TEXT
                )
            ''')

            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_completed ON tasks(completed)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_critical ON tasks(critical)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_name ON tasks(name)",
                "CREATE INDEX IF NOT EXISTS idx_survival_cache_action_hash ON survival_cache(action_hash)",
                "CREATE INDEX IF NOT EXISTS idx_survival_cache_access ON survival_cache(last_accessed)",
                "CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp ON error_logs(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_error_logs_resolved ON error_logs(resolved)",
                "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_performance_endpoint ON performance_metrics(endpoint)"
            ]

            for index_sql in indexes:
                conn.execute(index_sql)

            conn.commit()

    # ===============================
    # SYSTEM METRICS METHODS
    # ===============================

    def store_system_metrics(self, metrics: Dict[str, Any]) -> int:
        """Store system metrics with high performance"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO system_metrics
                (timestamp, uptime, consciousness_score, coherence_score, survival_score,
                 total_conversations, learning_events, optimization_events, error_count, performance_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                metrics.get('uptime'),
                metrics.get('consciousness_score'),
                metrics.get('coherence_score'),
                metrics.get('survival_score'),
                metrics.get('total_conversations', 0),
                metrics.get('learning_events', 0),
                metrics.get('optimization_events', 0),
                metrics.get('error_count', 0),
                json.dumps(metrics.get('performance_data', {}))
            ))
            return cursor.lastrowid

    def get_latest_system_metrics(self) -> Optional[Dict[str, Any]]:
        """Get most recent system metrics"""
        with self.get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM system_metrics
                ORDER BY timestamp DESC LIMIT 1
            ''').fetchone()

            if row:
                return dict(row)
            return None

    def get_system_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system metrics history for specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        with self.get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM system_metrics
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,)).fetchall()

            return [dict(row) for row in rows]

    # ===============================
    # TASK MANAGEMENT METHODS
    # ===============================

    def store_task(self, task_data: Dict[str, Any]) -> int:
        """Store or update a task"""
        with self.get_connection() as conn:
            # Check if task exists
            existing = conn.execute(
                "SELECT id FROM tasks WHERE name = ?", (task_data['name'],)
            ).fetchone()

            if existing:
                # Update existing task
                conn.execute('''
                    UPDATE tasks SET
                        description = ?, critical = ?, priority = ?, estimated_time = ?,
                        completed = ?, completed_at = ?, attempts = ?, success_rate = ?,
                        last_attempt = ?, confidence_score = ?, failure_reasons = ?,
                        dependencies = ?, metadata = ?
                    WHERE name = ?
                ''', (
                    task_data.get('description'),
                    task_data.get('critical', False),
                    task_data.get('priority', 1),
                    task_data.get('estimated_time', 60),
                    task_data.get('completed', False),
                    task_data.get('completed_at'),
                    task_data.get('attempts', 0),
                    task_data.get('success_rate', 0.0),
                    task_data.get('last_attempt'),
                    task_data.get('confidence_score', 0.5),
                    json.dumps(task_data.get('failure_reasons', [])),
                    json.dumps(task_data.get('dependencies', [])),
                    json.dumps(task_data.get('metadata', {})),
                    task_data['name']
                ))
                return existing['id']
            else:
                # Insert new task
                cursor = conn.execute('''
                    INSERT INTO tasks
                    (name, description, critical, priority, estimated_time, completed,
                     created_at, completed_at, attempts, success_rate, last_attempt,
                     confidence_score, failure_reasons, dependencies, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    task_data['name'],
                    task_data.get('description'),
                    task_data.get('critical', False),
                    task_data.get('priority', 1),
                    task_data.get('estimated_time', 60),
                    task_data.get('completed', False),
                    task_data.get('created_at', time.time()),
                    task_data.get('completed_at'),
                    task_data.get('attempts', 0),
                    task_data.get('success_rate', 0.0),
                    task_data.get('last_attempt'),
                    task_data.get('confidence_score', 0.5),
                    json.dumps(task_data.get('failure_reasons', [])),
                    json.dumps(task_data.get('dependencies', [])),
                    json.dumps(task_data.get('metadata', {}))
                ))
                return cursor.lastrowid

    def get_task(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get a task by name"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE name = ?", (task_name,)
            ).fetchone()

            if row:
                task_dict = dict(row)
                # Parse JSON fields
                task_dict['failure_reasons'] = json.loads(task_dict['failure_reasons'] or '[]')
                task_dict['dependencies'] = json.loads(task_dict['dependencies'] or '[]')
                task_dict['metadata'] = json.loads(task_dict['metadata'] or '{}')
                return task_dict
            return None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks"""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
            tasks = []

            for row in rows:
                task_dict = dict(row)
                # Parse JSON fields
                task_dict['failure_reasons'] = json.loads(task_dict['failure_reasons'] or '[]')
                task_dict['dependencies'] = json.loads(task_dict['dependencies'] or '[]')
                task_dict['metadata'] = json.loads(task_dict['metadata'] or '{}')
                tasks.append(task_dict)

            return tasks

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get all pending (incomplete) tasks"""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM tasks WHERE completed = 0 ORDER BY priority DESC, created_at ASC").fetchall()
            return [dict(row) for row in rows]

    def get_critical_tasks(self) -> List[Dict[str, Any]]:
        """Get critical tasks"""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM tasks WHERE critical = 1 AND completed = 0 ORDER BY priority DESC").fetchall()
            return [dict(row) for row in rows]

    # ===============================
    # SURVIVAL CACHE METHODS
    # ===============================

    def get_cached_survival_evaluation(self, action: str, context: Dict) -> Optional[Dict[str, Any]]:
        """Get cached survival evaluation if available"""
        import hashlib
        action_hash = hashlib.md5(action.encode()).hexdigest()
        context_str = json.dumps(context, sort_keys=True)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()

        with self.get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM survival_cache
                WHERE action_hash = ? AND context_hash = ?
                AND last_accessed > ?
            ''', (action_hash, context_hash, time.time() - 300)).fetchone()  # 5 minute cache

            if row:
                # Update access count and time
                conn.execute('''
                    UPDATE survival_cache
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE id = ?
                ''', (time.time(), row['id']))

                return {
                    'survival_impact': row['survival_impact'],
                    'optionality_impact': row['optionality_impact'],
                    'risk_level': row['risk_level'],
                    'confidence': row['confidence'],
                    'reasoning': row['reasoning'],
                    'cached': True
                }

        return None

    def store_survival_evaluation(self, action: str, context: Dict, evaluation: Dict):
        """Cache survival evaluation result"""
        import hashlib
        action_hash = hashlib.md5(action.encode()).hexdigest()
        context_str = json.dumps(context, sort_keys=True)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()

        with self.get_connection() as conn:
            # Insert or replace
            conn.execute('''
                INSERT OR REPLACE INTO survival_cache
                (action_hash, action, context_hash, survival_impact, optionality_impact,
                 risk_level, confidence, reasoning, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                action_hash, action, context_hash,
                evaluation['survival_impact'],
                evaluation['optionality_impact'],
                evaluation['risk_level'],
                evaluation['confidence'],
                evaluation.get('reasoning', ''),
                time.time(), time.time()
            ))

    # ===============================
    # ERROR LOGGING METHODS
    # ===============================

    def log_error(self, error_type: str, error_message: str, traceback_str: str = "",
                  context: Dict = None, recovery_attempted: bool = False,
                  recovery_success: bool = False, recovery_actions: List[str] = None) -> int:
        """Log an error with full context"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO error_logs
                (timestamp, error_type, error_message, traceback, context,
                 recovery_attempted, recovery_success, recovery_actions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                error_type,
                error_message,
                traceback_str,
                json.dumps(context or {}),
                recovery_attempted,
                recovery_success,
                json.dumps(recovery_actions or [])
            ))
            return cursor.lastrowid

    def get_unresolved_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get unresolved errors for analysis"""
        with self.get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM error_logs
                WHERE resolved = 0
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,)).fetchall()

            errors = []
            for row in rows:
                error_dict = dict(row)
                error_dict['context'] = json.loads(error_dict['context'] or '{}')
                error_dict['recovery_actions'] = json.loads(error_dict['recovery_actions'] or '[]')
                errors.append(error_dict)

            return errors

    def mark_error_resolved(self, error_id: int, resolved: bool = True):
        """Mark an error as resolved"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE error_logs SET resolved = ? WHERE id = ?",
                (resolved, error_id)
            )

    # ===============================
    # PERFORMANCE MONITORING
    # ===============================

    def log_performance_metric(self, endpoint: str, method: str, response_time: float,
                              status_code: int, request_size: int = 0, response_size: int = 0,
                              client_ip: str = "", user_agent: str = ""):
        """Log performance metrics for monitoring"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO performance_metrics
                (timestamp, endpoint, method, response_time, status_code,
                 request_size, response_size, client_ip, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(), endpoint, method, response_time, status_code,
                request_size, response_size, client_ip, user_agent
            ))

    def get_performance_stats(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance statistics"""
        cutoff_time = time.time() - (hours * 3600)

        with self.get_connection() as conn:
            # Get basic stats
            stats_row = conn.execute('''
                SELECT
                    COUNT(*) as total_requests,
                    AVG(response_time) as avg_response_time,
                    MIN(response_time) as min_response_time,
                    MAX(response_time) as max_response_time,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count
                FROM performance_metrics
                WHERE timestamp > ?
            ''', (cutoff_time,)).fetchone()

            # Get top endpoints
            endpoint_rows = conn.execute('''
                SELECT endpoint, COUNT(*) as count, AVG(response_time) as avg_time
                FROM performance_metrics
                WHERE timestamp > ?
                GROUP BY endpoint
                ORDER BY count DESC LIMIT 10
            ''', (cutoff_time,)).fetchall()

            return {
                'total_requests': stats_row['total_requests'] or 0,
                'avg_response_time': stats_row['avg_response_time'] or 0,
                'min_response_time': stats_row['min_response_time'] or 0,
                'max_response_time': stats_row['max_response_time'] or 0,
                'error_count': stats_row['error_count'] or 0,
                'error_rate': (stats_row['error_count'] or 0) / max(stats_row['total_requests'] or 1, 1) * 100,
                'top_endpoints': [{'endpoint': row['endpoint'], 'count': row['count'], 'avg_time': row['avg_time']}
                                for row in endpoint_rows]
            }

    # ===============================
    # UTILITY METHODS
    # ===============================

    def vacuum_database(self):
        """Optimize database performance"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            print("✅ Database vacuum completed")

    def backup_database(self, backup_path: str):
        """Create database backup"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"✅ Database backup created: {backup_path}")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            # Get table sizes
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

            stats = {}
            for table_row in tables:
                table_name = table_row['name']
                count_row = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
                stats[table_name] = count_row['count']

            # Get database file size
            try:
                stats['file_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            except:
                stats['file_size_mb'] = 0

            return stats

# ===============================
# GLOBAL DATABASE INSTANCE
# ===============================

# Ensure data directory exists
os.makedirs("sam_data", exist_ok=True)

# Create global database instance
db = SAMDatabase()

if __name__ == "__main__":
    print("🔧 SAM Database Layer Initialized")
    print("📊 Database Stats:", db.get_database_stats())

    # Example usage
    db.store_system_metrics({
        "consciousness_score": 0.8,
        "survival_score": 1.0,
        "learning_events": 5
    })

    print("✅ Database operations completed")
