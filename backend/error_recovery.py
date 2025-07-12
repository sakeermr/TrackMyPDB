"""
TrackMyPDB Error Recovery System
@author Anu Gamage

Robust error recovery system that handles API failures, network issues, and data corruption.
Includes automatic checkpointing and resume capabilities for long-running analyses.
Licensed under MIT License - Open Source Project
"""

import os
import json
import pickle
import hashlib
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from pathlib import Path
import sqlite3
import asyncio
import traceback
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

class ErrorType(Enum):
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"

class CheckpointStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

@dataclass
class ErrorRecord:
    error_id: str
    error_type: ErrorType
    timestamp: float
    error_message: str
    traceback_info: str
    context: Dict[str, Any]
    retry_count: int = 0
    resolved: bool = False
    resolution_strategy: Optional[str] = None

@dataclass
class CheckpointData:
    checkpoint_id: str
    task_id: str
    task_name: str
    timestamp: float
    status: CheckpointStatus
    current_state: Dict[str, Any]
    progress: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None

class CheckpointManager:
    """
    Manages checkpoints for long-running tasks
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize SQLite database for checkpoint metadata
        self.db_path = self.checkpoint_dir / "checkpoints.db"
        self._init_database()
        
        # Active checkpoints cache
        self.active_checkpoints: Dict[str, CheckpointData] = {}
        self.lock = threading.RLock()
    
    def _init_database(self):
        """Initialize SQLite database for checkpoint storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    file_path TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_id ON checkpoints(task_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON checkpoints(timestamp)
            """)
    
    def create_checkpoint(self, 
                         task_id: str, 
                         task_name: str, 
                         current_state: Dict[str, Any],
                         progress: Dict[str, Any],
                         metadata: Optional[Dict[str, Any]] = None) -> CheckpointData:
        """Create a new checkpoint"""
        with self.lock:
            checkpoint_id = str(uuid.uuid4())
            timestamp = time.time()
            
            checkpoint = CheckpointData(
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                task_name=task_name,
                timestamp=timestamp,
                status=CheckpointStatus.ACTIVE,
                current_state=current_state,
                progress=progress,
                metadata=metadata or {}
            )
            
            # Save checkpoint data to file
            file_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            checkpoint.file_path = str(file_path)
            
            # Save metadata to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO checkpoints 
                    (checkpoint_id, task_id, task_name, timestamp, status, file_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint_id,
                    task_id,
                    task_name,
                    timestamp,
                    checkpoint.status.value,
                    str(file_path),
                    json.dumps(metadata or {})
                ))
            
            self.active_checkpoints[checkpoint_id] = checkpoint
            
            logging.info(f"Created checkpoint {checkpoint_id} for task {task_id}")
            return checkpoint
    
    def update_checkpoint(self, 
                         checkpoint_id: str, 
                         current_state: Dict[str, Any],
                         progress: Dict[str, Any],
                         status: Optional[CheckpointStatus] = None):
        """Update an existing checkpoint"""
        with self.lock:
            if checkpoint_id not in self.active_checkpoints:
                # Try to load from database
                checkpoint = self.load_checkpoint(checkpoint_id)
                if checkpoint is None:
                    raise ValueError(f"Checkpoint {checkpoint_id} not found")
            else:
                checkpoint = self.active_checkpoints[checkpoint_id]
            
            # Update data
            checkpoint.current_state.update(current_state)
            checkpoint.progress.update(progress)
            checkpoint.timestamp = time.time()
            
            if status:
                checkpoint.status = status
            
            # Save updated data
            if checkpoint.file_path:
                with open(checkpoint.file_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE checkpoints 
                    SET timestamp = ?, status = ?
                    WHERE checkpoint_id = ?
                """, (checkpoint.timestamp, checkpoint.status.value, checkpoint_id))
            
            logging.info(f"Updated checkpoint {checkpoint_id}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """Load a checkpoint from storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_path FROM checkpoints 
                    WHERE checkpoint_id = ?
                """, (checkpoint_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                file_path = row[0]
                if not os.path.exists(file_path):
                    logging.warning(f"Checkpoint file {file_path} not found")
                    return None
                
                with open(file_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                self.active_checkpoints[checkpoint_id] = checkpoint
                return checkpoint
                
        except Exception as e:
            logging.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            return None
    
    def get_latest_checkpoint(self, task_id: str) -> Optional[CheckpointData]:
        """Get the latest checkpoint for a task"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT checkpoint_id FROM checkpoints 
                    WHERE task_id = ? AND status = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (task_id, CheckpointStatus.ACTIVE.value))
                
                row = cursor.fetchone()
                if row:
                    return self.load_checkpoint(row[0])
                
        except Exception as e:
            logging.error(f"Error getting latest checkpoint for task {task_id}: {e}")
        
        return None
    
    def complete_checkpoint(self, checkpoint_id: str):
        """Mark a checkpoint as completed"""
        self.update_checkpoint(checkpoint_id, {}, {}, CheckpointStatus.COMPLETED)
        
        # Remove from active cache
        with self.lock:
            if checkpoint_id in self.active_checkpoints:
                del self.active_checkpoints[checkpoint_id]
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24):
        """Clean up old checkpoints"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get old checkpoint files
                cursor = conn.execute("""
                    SELECT checkpoint_id, file_path FROM checkpoints 
                    WHERE timestamp < ? AND status IN (?, ?)
                """, (cutoff_time, CheckpointStatus.COMPLETED.value, CheckpointStatus.FAILED.value))
                
                old_checkpoints = cursor.fetchall()
                
                # Delete files and database records
                for checkpoint_id, file_path in old_checkpoints:
                    try:
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                        
                        conn.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
                        
                        logging.info(f"Cleaned up checkpoint {checkpoint_id}")
                        
                    except Exception as e:
                        logging.error(f"Error cleaning up checkpoint {checkpoint_id}: {e}")
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error during checkpoint cleanup: {e}")

class ErrorRecoverySystem:
    """
    Comprehensive error recovery system with automatic retry and checkpointing
    """
    
    def __init__(self, 
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0):
        
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Error handling strategies
        self.error_handlers: Dict[ErrorType, Callable] = {
            ErrorType.NETWORK_ERROR: self._handle_network_error,
            ErrorType.API_ERROR: self._handle_api_error,
            ErrorType.TIMEOUT_ERROR: self._handle_timeout_error,
            ErrorType.DATA_CORRUPTION: self._handle_data_corruption,
            ErrorType.MEMORY_ERROR: self._handle_memory_error,
            ErrorType.VALIDATION_ERROR: self._handle_validation_error,
            ErrorType.UNKNOWN_ERROR: self._handle_unknown_error
        }
        
        # Error history
        self.error_history: List[ErrorRecord] = []
        self.lock = threading.RLock()
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        """Classify an exception into an error type"""
        error_message = str(exception).lower()
        
        if any(keyword in error_message for keyword in ['network', 'connection', 'timeout', 'unreachable']):
            return ErrorType.NETWORK_ERROR
        elif any(keyword in error_message for keyword in ['api', 'http', 'status', 'response']):
            return ErrorType.API_ERROR
        elif 'timeout' in error_message:
            return ErrorType.TIMEOUT_ERROR
        elif any(keyword in error_message for keyword in ['memory', 'allocation', 'oom']):
            return ErrorType.MEMORY_ERROR
        elif any(keyword in error_message for keyword in ['validation', 'invalid', 'format']):
            return ErrorType.VALIDATION_ERROR
        elif any(keyword in error_message for keyword in ['corrupt', 'damaged', 'integrity']):
            return ErrorType.DATA_CORRUPTION
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        delay = self.base_delay * (2 ** attempt)
        # Add jitter
        import random
        jitter = delay * 0.1 * random.random()
        return min(delay + jitter, self.max_delay)
    
    def _handle_network_error(self, error: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Handle network-related errors"""
        logging.warning(f"Handling network error: {error.error_message}")
        
        if error.retry_count < self.max_retries:
            delay = self._calculate_retry_delay(error.retry_count)
            logging.info(f"Retrying after {delay:.2f} seconds...")
            time.sleep(delay)
            return True
        
        return False
    
    def _handle_api_error(self, error: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Handle API-related errors"""
        logging.warning(f"Handling API error: {error.error_message}")
        
        # Check if it's a rate limit error
        if any(keyword in error.error_message.lower() for keyword in ['rate limit', 'too many requests']):
            delay = self._calculate_retry_delay(error.retry_count) * 2  # Longer delay for rate limits
            logging.info(f"Rate limit detected, waiting {delay:.2f} seconds...")
            time.sleep(delay)
            return True
        
        # For other API errors, try limited retries
        if error.retry_count < self.max_retries:
            delay = self._calculate_retry_delay(error.retry_count)
            time.sleep(delay)
            return True
        
        return False
    
    def _handle_timeout_error(self, error: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Handle timeout errors"""
        logging.warning(f"Handling timeout error: {error.error_message}")
        
        if error.retry_count < self.max_retries:
            # Increase timeout for next attempt
            new_timeout = context.get('timeout', 30) * 1.5
            context['timeout'] = min(new_timeout, 300)  # Max 5 minutes
            
            delay = self._calculate_retry_delay(error.retry_count)
            time.sleep(delay)
            return True
        
        return False
    
    def _handle_data_corruption(self, error: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Handle data corruption errors"""
        logging.error(f"Data corruption detected: {error.error_message}")
        
        # Try to recover from checkpoint
        task_id = context.get('task_id')
        if task_id:
            checkpoint = self.checkpoint_manager.get_latest_checkpoint(task_id)
            if checkpoint:
                logging.info(f"Attempting recovery from checkpoint {checkpoint.checkpoint_id}")
                context['recovery_checkpoint'] = checkpoint
                return True
        
        return False
    
    def _handle_memory_error(self, error: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Handle memory-related errors"""
        logging.warning(f"Memory error detected: {error.error_message}")
        
        # Try to reduce batch size or clear caches
        if 'batch_size' in context:
            context['batch_size'] = max(1, context['batch_size'] // 2)
            logging.info(f"Reduced batch size to {context['batch_size']}")
        
        # Clear any caches
        if 'clear_cache' in context and callable(context['clear_cache']):
            context['clear_cache']()
            logging.info("Cleared caches to free memory")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return error.retry_count < self.max_retries
    
    def _handle_validation_error(self, error: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Handle validation errors"""
        logging.error(f"Validation error: {error.error_message}")
        
        # Validation errors usually shouldn't be retried unless we can fix the data
        if 'data_fixer' in context and callable(context['data_fixer']):
            try:
                context['data_fixer'](error)
                return True
            except Exception as e:
                logging.error(f"Failed to fix data: {e}")
        
        return False
    
    def _handle_unknown_error(self, error: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Handle unknown errors"""
        logging.error(f"Unknown error: {error.error_message}")
        
        # Limited retries for unknown errors
        if error.retry_count < min(2, self.max_retries):
            delay = self._calculate_retry_delay(error.retry_count)
            time.sleep(delay)
            return True
        
        return False
    
    def record_error(self, 
                    exception: Exception, 
                    context: Dict[str, Any]) -> ErrorRecord:
        """Record an error for tracking and analysis"""
        with self.lock:
            error_id = str(uuid.uuid4())
            error_type = self._classify_error(exception)
            
            error_record = ErrorRecord(
                error_id=error_id,
                error_type=error_type,
                timestamp=time.time(),
                error_message=str(exception),
                traceback_info=traceback.format_exc(),
                context=context.copy(),
                retry_count=0
            )
            
            self.error_history.append(error_record)
            
            logging.error(f"Recorded error {error_id}: {error_type.value} - {error_record.error_message}")
            
            return error_record
    
    def handle_error(self, 
                    error_record: ErrorRecord, 
                    context: Dict[str, Any]) -> bool:
        """Handle an error using the appropriate strategy"""
        handler = self.error_handlers.get(error_record.error_type, self._handle_unknown_error)
        
        try:
            should_retry = handler(error_record, context)
            error_record.retry_count += 1
            
            if should_retry:
                logging.info(f"Error {error_record.error_id} will be retried (attempt {error_record.retry_count})")
            else:
                logging.error(f"Error {error_record.error_id} cannot be recovered")
                error_record.resolved = False
            
            return should_retry
            
        except Exception as e:
            logging.error(f"Error in error handler: {e}")
            return False
    
    @contextmanager
    def recoverable_operation(self, 
                            task_id: str,
                            task_name: str,
                            operation_context: Optional[Dict[str, Any]] = None):
        """Context manager for recoverable operations"""
        context = operation_context or {}
        context['task_id'] = task_id
        
        # Check for existing checkpoint
        existing_checkpoint = self.checkpoint_manager.get_latest_checkpoint(task_id)
        if existing_checkpoint:
            logging.info(f"Found existing checkpoint for task {task_id}, resuming...")
            context['resume_checkpoint'] = existing_checkpoint
        
        try:
            yield context
            
            # Mark any checkpoints as completed
            if existing_checkpoint:
                self.checkpoint_manager.complete_checkpoint(existing_checkpoint.checkpoint_id)
            
        except Exception as e:
            error_record = self.record_error(e, context)
            
            # Try to handle the error
            if self.handle_error(error_record, context):
                # If error can be handled, create a checkpoint for recovery
                if 'current_state' in context and 'progress' in context:
                    checkpoint = self.checkpoint_manager.create_checkpoint(
                        task_id=task_id,
                        task_name=task_name,
                        current_state=context['current_state'],
                        progress=context['progress'],
                        metadata={'error_id': error_record.error_id}
                    )
                    context['recovery_checkpoint'] = checkpoint
            
            raise
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns"""
        with self.lock:
            if not self.error_history:
                return {}
            
            error_counts = {}
            for error in self.error_history:
                error_counts[error.error_type.value] = error_counts.get(error.error_type.value, 0) + 1
            
            recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors': len(recent_errors),
                'error_types': error_counts,
                'average_retries': sum(e.retry_count for e in self.error_history) / len(self.error_history),
                'resolution_rate': sum(1 for e in self.error_history if e.resolved) / len(self.error_history)
            }

# Utility functions for easy integration

def create_error_recovery_system(checkpoint_dir: str = "checkpoints") -> ErrorRecoverySystem:
    """Create a default error recovery system"""
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    return ErrorRecoverySystem(checkpoint_manager)

@contextmanager
def auto_checkpoint(task_id: str, 
                   task_name: str,
                   checkpoint_interval: int = 100,
                   recovery_system: Optional[ErrorRecoverySystem] = None):
    """Auto-checkpointing context manager"""
    if recovery_system is None:
        recovery_system = create_error_recovery_system()
    
    with recovery_system.recoverable_operation(task_id, task_name) as context:
        # Setup auto-checkpointing
        context['checkpoint_counter'] = 0
        context['checkpoint_interval'] = checkpoint_interval
        context['recovery_system'] = recovery_system
        
        def auto_checkpoint_func(current_state: Dict[str, Any], progress: Dict[str, Any]):
            context['checkpoint_counter'] += 1
            if context['checkpoint_counter'] % checkpoint_interval == 0:
                recovery_system.checkpoint_manager.create_checkpoint(
                    task_id=task_id,
                    task_name=task_name,
                    current_state=current_state,
                    progress=progress
                )
        
        context['auto_checkpoint'] = auto_checkpoint_func
        yield context