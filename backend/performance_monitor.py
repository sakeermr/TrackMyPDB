"""
TrackMyPDB Performance Monitoring and Smart Progress Tracking
@author Anu Gamage

Comprehensive monitoring system that tracks API response times, memory usage,
processing speed, and provides real-time progress updates with ETA calculations.
Licensed under MIT License - Open Source Project
"""

import time
import psutil
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import streamlit as st
import logging
import functools
import tracemalloc
from contextlib import contextmanager

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PerformanceMetrics:
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start: float = 0
    memory_end: float = 0
    memory_peak: float = 0
    cpu_percent: float = 0
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class TaskProgress:
    task_id: str
    task_name: str
    total_items: int
    completed_items: int = 0
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    estimated_completion: Optional[float] = None
    current_operation: str = ""
    sub_tasks: Dict[str, 'TaskProgress'] = field(default_factory=dict)
    metrics: PerformanceMetrics = field(default_factory=lambda: PerformanceMetrics(start_time=time.time()))
    error_message: Optional[str] = None

class SmartProgressTracker:
    """
    Intelligent progress tracking with ETA calculations and real-time updates
    """
    
    def __init__(self):
        self.tasks: Dict[str, TaskProgress] = {}
        self.metrics_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.lock = threading.RLock()
        self._progress_callbacks: List[Callable] = []
        
        # Performance monitoring
        self.cpu_monitor = CPUMonitor()
        self.memory_monitor = MemoryMonitor()
        
    def create_task(self, task_id: str, task_name: str, total_items: int) -> TaskProgress:
        """Create a new task for tracking"""
        with self.lock:
            task = TaskProgress(
                task_id=task_id,
                task_name=task_name,
                total_items=total_items,
                start_time=time.time(),
                status=TaskStatus.PENDING
            )
            self.tasks[task_id] = task
            return task
    
    def start_task(self, task_id: str):
        """Start tracking a task"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()
                task.metrics.start_time = time.time()
                task.metrics.memory_start = self.memory_monitor.get_current_usage()
                self._notify_progress_update(task)
    
    def update_progress(self, task_id: str, completed_items: int, current_operation: str = ""):
        """Update task progress"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.completed_items = completed_items
                task.current_operation = current_operation
                
                # Calculate ETA
                if task.start_time and completed_items > 0:
                    elapsed = time.time() - task.start_time
                    items_per_second = completed_items / elapsed
                    remaining_items = task.total_items - completed_items
                    if items_per_second > 0:
                        eta_seconds = remaining_items / items_per_second
                        task.estimated_completion = time.time() + eta_seconds
                
                self._notify_progress_update(task)
    
    def complete_task(self, task_id: str, success: bool = True, error_message: Optional[str] = None):
        """Mark task as completed"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.completed_items = task.total_items if success else task.completed_items
                task.error_message = error_message
                
                # Finalize metrics
                task.metrics.end_time = time.time()
                task.metrics.duration = task.metrics.end_time - task.metrics.start_time
                task.metrics.memory_end = self.memory_monitor.get_current_usage()
                task.metrics.cpu_percent = self.cpu_monitor.get_average_usage()
                
                # Store metrics history
                self.metrics_history[task_id].append(task.metrics)
                
                self._notify_progress_update(task)
    
    def add_subtask(self, parent_task_id: str, subtask_id: str, subtask_name: str, total_items: int):
        """Add a subtask to a parent task"""
        with self.lock:
            if parent_task_id in self.tasks:
                subtask = TaskProgress(
                    task_id=subtask_id,
                    task_name=subtask_name,
                    total_items=total_items,
                    start_time=time.time(),
                    status=TaskStatus.PENDING
                )
                self.tasks[parent_task_id].sub_tasks[subtask_id] = subtask
    
    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get current progress for a task"""
        return self.tasks.get(task_id)
    
    def get_all_active_tasks(self) -> List[TaskProgress]:
        """Get all currently active tasks"""
        return [task for task in self.tasks.values() 
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]]
    
    def register_progress_callback(self, callback: Callable[[TaskProgress], None]):
        """Register a callback for progress updates"""
        self._progress_callbacks.append(callback)
    
    def _notify_progress_update(self, task: TaskProgress):
        """Notify all registered callbacks of progress update"""
        for callback in self._progress_callbacks:
            try:
                callback(task)
            except Exception as e:
                logging.error(f"Error in progress callback: {e}")
    
    def get_performance_summary(self, task_id: str) -> Dict[str, Any]:
        """Get performance summary for a task"""
        if task_id not in self.metrics_history:
            return {}
        
        metrics_list = self.metrics_history[task_id]
        if not metrics_list:
            return {}
        
        latest = metrics_list[-1]
        return {
            'total_duration': latest.duration,
            'memory_usage': {
                'start_mb': latest.memory_start / (1024 * 1024),
                'end_mb': latest.memory_end / (1024 * 1024),
                'peak_mb': latest.memory_peak / (1024 * 1024)
            },
            'cpu_usage_percent': latest.cpu_percent,
            'api_calls': latest.api_calls,
            'cache_performance': {
                'hits': latest.cache_hits,
                'misses': latest.cache_misses,
                'hit_rate': latest.cache_hits / (latest.cache_hits + latest.cache_misses) if (latest.cache_hits + latest.cache_misses) > 0 else 0
            },
            'errors': latest.errors,
            'warnings': latest.warnings
        }

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        # Thresholds for alerts
        self.cpu_threshold = 80.0  # CPU usage %
        self.memory_threshold = 1024 * 1024 * 1024  # 1GB memory usage
        self.response_time_threshold = 5.0  # 5 seconds
        
    def monitor_analysis_performance(self, func: Callable) -> Callable:
        """Decorator for monitoring function performance"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            # Start memory tracing for detailed analysis
            tracemalloc.start()
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                metrics = {
                    'function': func.__name__,
                    'duration': duration,
                    'memory_delta': memory_delta,
                    'memory_peak': peak,
                    'timestamp': end_time,
                    'success': True
                }
                
                with self.lock:
                    self.metrics[func.__name__].append(metrics)
                
                # Check for performance alerts
                self._check_performance_alerts(metrics)
                
                return result
                
            except Exception as e:
                # Record failed execution
                end_time = time.time()
                duration = end_time - start_time
                
                metrics = {
                    'function': func.__name__,
                    'duration': duration,
                    'timestamp': end_time,
                    'success': False,
                    'error': str(e)
                }
                
                with self.lock:
                    self.metrics[func.__name__].append(metrics)
                
                tracemalloc.stop()
                raise
        
        return wrapper
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance issues and generate alerts"""
        alerts = []
        
        if metrics['duration'] > self.response_time_threshold:
            alerts.append(f"Slow response time: {metrics['duration']:.2f}s for {metrics['function']}")
        
        if metrics.get('memory_delta', 0) > self.memory_threshold:
            alerts.append(f"High memory usage: {metrics['memory_delta'] / (1024*1024):.1f}MB for {metrics['function']}")
        
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > self.cpu_threshold:
            alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if alerts:
            with self.lock:
                self.alerts.extend(alerts)
    
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific function"""
        with self.lock:
            if function_name not in self.metrics:
                return {}
            
            metrics_list = self.metrics[function_name]
            successful_calls = [m for m in metrics_list if m.get('success', True)]
            failed_calls = [m for m in metrics_list if not m.get('success', True)]
            
            if not successful_calls:
                return {'total_calls': len(metrics_list), 'success_rate': 0}
            
            durations = [m['duration'] for m in successful_calls]
            memory_deltas = [m.get('memory_delta', 0) for m in successful_calls]
            
            return {
                'total_calls': len(metrics_list),
                'successful_calls': len(successful_calls),
                'failed_calls': len(failed_calls),
                'success_rate': len(successful_calls) / len(metrics_list),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_memory_delta': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
                'total_memory_used': sum(memory_deltas)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system performance statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            }
        }
    
    def get_alerts(self, clear_after_read: bool = True) -> List[str]:
        """Get current performance alerts"""
        with self.lock:
            alerts = self.alerts.copy()
            if clear_after_read:
                self.alerts.clear()
            return alerts
    
    def clear_metrics(self):
        """Clear all collected metrics"""
        with self.lock:
            self.metrics.clear()
            self.alerts.clear()

class CPUMonitor:
    """CPU usage monitoring"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.usage_history = deque(maxlen=60)  # Keep last 60 samples
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous CPU monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop CPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=self.sample_interval)
            self.usage_history.append(cpu_percent)
    
    def get_current_usage(self) -> float:
        """Get current CPU usage"""
        return psutil.cpu_percent()
    
    def get_average_usage(self, samples: int = 10) -> float:
        """Get average CPU usage over last N samples"""
        if not self.usage_history:
            return 0.0
        recent_samples = list(self.usage_history)[-samples:]
        return sum(recent_samples) / len(recent_samples)

class MemoryMonitor:
    """Memory usage monitoring"""
    
    def __init__(self):
        self.usage_history = deque(maxlen=100)  # Keep last 100 samples
    
    def get_current_usage(self) -> float:
        """Get current memory usage in bytes"""
        return psutil.Process().memory_info().rss
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get system memory statistics"""
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'percent': mem.percent,
            'used': mem.used
        }
    
    def track_usage(self):
        """Track current memory usage"""
        self.usage_history.append(self.get_current_usage())
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage from history"""
        return max(self.usage_history) if self.usage_history else 0.0

# Streamlit UI Components for Progress Tracking

def display_progress_dashboard(tracker: SmartProgressTracker):
    """Display real-time progress dashboard in Streamlit"""
    st.subheader("🔄 Analysis Progress Dashboard")
    
    active_tasks = tracker.get_all_active_tasks()
    
    if not active_tasks:
        st.info("No active tasks")
        return
    
    for task in active_tasks:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                progress_percent = (task.completed_items / task.total_items) * 100 if task.total_items > 0 else 0
                st.progress(progress_percent / 100)
                st.write(f"**{task.task_name}** ({task.completed_items}/{task.total_items})")
                if task.current_operation:
                    st.caption(f"Current: {task.current_operation}")
            
            with col2:
                if task.estimated_completion:
                    eta = task.estimated_completion - time.time()
                    if eta > 0:
                        st.metric("ETA", f"{eta:.0f}s")
                    else:
                        st.metric("ETA", "Almost done")
                else:
                    st.metric("ETA", "Calculating...")
            
            with col3:
                status_color = {
                    TaskStatus.PENDING: "🟡",
                    TaskStatus.RUNNING: "🟢",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌"
                }
                st.write(f"{status_color.get(task.status, '⚪')} {task.status.value.title()}")

def display_performance_metrics(monitor: PerformanceMonitor):
    """Display performance metrics in Streamlit"""
    st.subheader("📊 Performance Metrics")
    
    # System stats
    system_stats = monitor.get_system_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CPU Usage", f"{system_stats['cpu_percent']:.1f}%")
    with col2:
        st.metric("Memory Usage", f"{system_stats['memory']['percent']:.1f}%")
    with col3:
        memory_used_gb = system_stats['memory']['used'] / (1024**3)
        st.metric("Memory Used", f"{memory_used_gb:.1f} GB")
    with col4:
        disk_percent = system_stats['disk']['percent']
        st.metric("Disk Usage", f"{disk_percent:.1f}%")
    
    # Alerts
    alerts = monitor.get_alerts(clear_after_read=False)
    if alerts:
        st.warning("⚠️ Performance Alerts:")
        for alert in alerts[-5:]:  # Show last 5 alerts
            st.write(f"• {alert}")

# Context manager for easy performance tracking
@contextmanager
def track_performance(task_name: str, tracker: SmartProgressTracker, total_items: int = 1):
    """Context manager for easy performance tracking"""
    import uuid
    task_id = str(uuid.uuid4())
    
    try:
        task = tracker.create_task(task_id, task_name, total_items)
        tracker.start_task(task_id)
        yield task
        tracker.complete_task(task_id, success=True)
    except Exception as e:
        tracker.complete_task(task_id, success=False, error_message=str(e))
        raise