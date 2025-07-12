"""
TrackMyPDB Adaptive Batch Processor
@author Anu Gamage

Adaptive batch processing system that dynamically adjusts batch sizes based on
API response times, system resources, and error rates to optimize throughput.
Licensed under MIT License - Open Source Project
"""

import time
import asyncio
import psutil
import statistics
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import threading
import math
import json
from datetime import datetime, timedelta

class BatchStrategy(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"

class SystemMetrics(Enum):
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_LATENCY = "network_latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"

@dataclass
class BatchPerformanceMetrics:
    batch_size: int
    processing_time: float
    items_per_second: float
    error_count: int
    memory_usage: float
    cpu_usage: float
    timestamp: float
    success_rate: float
    api_response_times: List[float] = field(default_factory=list)

@dataclass
class SystemResourceMetrics:
    cpu_percent: float
    memory_percent: float
    available_memory: int
    network_io: Dict[str, int]
    timestamp: float

class AdaptiveBatchProcessor:
    """
    Dynamically adjusts batch sizes based on system performance and API responses
    """
    
    def __init__(self, 
                 initial_batch_size: int = 50,
                 min_batch_size: int = 1,
                 max_batch_size: int = 500,
                 strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
                 performance_window: int = 10,
                 adjustment_factor: float = 0.2):
        
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.strategy = strategy
        self.performance_window = performance_window
        self.adjustment_factor = adjustment_factor
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=performance_window)
        self.system_metrics: deque = deque(maxlen=performance_window)
        self.api_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=performance_window))
        
        # Optimization parameters
        self.target_cpu_usage = 70.0  # Target CPU usage percentage
        self.target_memory_usage = 80.0  # Target memory usage percentage
        self.max_error_rate = 0.1  # Maximum acceptable error rate (10%)
        self.target_response_time = 5.0  # Target API response time in seconds
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Resource monitoring
        self.monitor_system_resources = True
        self.resource_check_interval = 1.0  # seconds
        
        logging.info(f"Initialized AdaptiveBatchProcessor with strategy: {strategy.value}")
    
    def _get_system_metrics(self) -> SystemResourceMetrics:
        """Get current system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            network_io = psutil.net_io_counters()._asdict()
            
            return SystemResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                available_memory=memory.available,
                network_io=network_io,
                timestamp=time.time()
            )
        except Exception as e:
            logging.warning(f"Error getting system metrics: {e}")
            return SystemResourceMetrics(0, 0, 0, {}, time.time())
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on current metrics"""
        with self.lock:
            if not self.performance_history:
                return self.current_batch_size
            
            recent_metrics = list(self.performance_history)[-5:]  # Last 5 batches
            
            if self.strategy == BatchStrategy.CONSERVATIVE:
                return self._conservative_adjustment(recent_metrics)
            elif self.strategy == BatchStrategy.AGGRESSIVE:
                return self._aggressive_adjustment(recent_metrics)
            elif self.strategy == BatchStrategy.BALANCED:
                return self._balanced_adjustment(recent_metrics)
            else:  # ADAPTIVE
                return self._adaptive_adjustment(recent_metrics)
    
    def _conservative_adjustment(self, metrics: List[BatchPerformanceMetrics]) -> int:
        """Conservative batch size adjustment - prioritizes stability"""
        if not metrics:
            return self.current_batch_size
        
        latest = metrics[-1]
        
        # Reduce batch size if error rate is high or resources are stressed
        if (latest.error_count > 0 or 
            latest.cpu_usage > self.target_cpu_usage or 
            latest.memory_usage > self.target_memory_usage):
            
            reduction = max(1, int(self.current_batch_size * 0.1))
            new_size = max(self.min_batch_size, self.current_batch_size - reduction)
        else:
            # Gradual increase if everything is stable
            increase = max(1, int(self.current_batch_size * 0.05))
            new_size = min(self.max_batch_size, self.current_batch_size + increase)
        
        return new_size
    
    def _aggressive_adjustment(self, metrics: List[BatchPerformanceMetrics]) -> int:
        """Aggressive batch size adjustment - prioritizes throughput"""
        if not metrics:
            return min(self.max_batch_size, self.current_batch_size * 2)
        
        latest = metrics[-1]
        avg_throughput = statistics.mean([m.items_per_second for m in metrics])
        
        # Increase batch size aggressively if performance allows
        if (latest.error_count == 0 and 
            latest.cpu_usage < self.target_cpu_usage * 0.8 and
            latest.memory_usage < self.target_memory_usage * 0.8):
            
            increase = max(5, int(self.current_batch_size * 0.3))
            new_size = min(self.max_batch_size, self.current_batch_size + increase)
        elif latest.error_count > 0:
            # Quick reduction on errors
            reduction = max(5, int(self.current_batch_size * 0.2))
            new_size = max(self.min_batch_size, self.current_batch_size - reduction)
        else:
            new_size = self.current_batch_size
        
        return new_size
    
    def _balanced_adjustment(self, metrics: List[BatchPerformanceMetrics]) -> int:
        """Balanced batch size adjustment"""
        if not metrics:
            return self.current_batch_size
        
        latest = metrics[-1]
        
        # Calculate adjustment based on multiple factors
        adjustment_factors = []
        
        # Error rate factor
        if latest.success_rate < 0.95:
            adjustment_factors.append(-0.2)  # Reduce by 20%
        elif latest.success_rate > 0.98:
            adjustment_factors.append(0.1)   # Increase by 10%
        
        # Resource usage factor
        cpu_factor = (self.target_cpu_usage - latest.cpu_usage) / self.target_cpu_usage * 0.1
        memory_factor = (self.target_memory_usage - latest.memory_usage) / self.target_memory_usage * 0.1
        adjustment_factors.extend([cpu_factor, memory_factor])
        
        # Response time factor
        if latest.api_response_times:
            avg_response_time = statistics.mean(latest.api_response_times)
            response_factor = (self.target_response_time - avg_response_time) / self.target_response_time * 0.15
            adjustment_factors.append(response_factor)
        
        # Calculate final adjustment
        total_adjustment = sum(adjustment_factors) / len(adjustment_factors)
        adjustment_amount = int(self.current_batch_size * total_adjustment)
        
        new_size = self.current_batch_size + adjustment_amount
        return max(self.min_batch_size, min(self.max_batch_size, new_size))
    
    def _adaptive_adjustment(self, metrics: List[BatchPerformanceMetrics]) -> int:
        """Adaptive adjustment using machine learning-like approach"""
        if len(metrics) < 3:
            return self._balanced_adjustment(metrics)
        
        # Analyze trends
        throughput_trend = self._calculate_trend([m.items_per_second for m in metrics])
        error_trend = self._calculate_trend([m.error_count for m in metrics])
        response_time_trend = self._calculate_trend([
            statistics.mean(m.api_response_times) if m.api_response_times else 0 
            for m in metrics
        ])
        
        # Predict optimal batch size based on trends
        current_performance_score = self._calculate_performance_score(metrics[-1])
        
        if len(metrics) >= 2:
            previous_performance_score = self._calculate_performance_score(metrics[-2])
            performance_trend = current_performance_score - previous_performance_score
        else:
            performance_trend = 0
        
        # Adaptive decision making
        if performance_trend > 0.1 and error_trend <= 0:
            # Performance improving, increase batch size
            increase = max(1, int(self.current_batch_size * 0.15))
            new_size = min(self.max_batch_size, self.current_batch_size + increase)
        elif performance_trend < -0.1 or error_trend > 0:
            # Performance degrading, decrease batch size
            decrease = max(1, int(self.current_batch_size * 0.15))
            new_size = max(self.min_batch_size, self.current_batch_size - decrease)
        else:
            # Stable performance, make small adjustments based on resource usage
            latest = metrics[-1]
            if latest.cpu_usage < self.target_cpu_usage * 0.7 and latest.memory_usage < self.target_memory_usage * 0.7:
                new_size = min(self.max_batch_size, self.current_batch_size + 1)
            else:
                new_size = self.current_batch_size
        
        return new_size
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return max(-1, min(1, slope))  # Normalize to [-1, 1]
    
    def _calculate_performance_score(self, metrics: BatchPerformanceMetrics) -> float:
        """Calculate overall performance score (0-1)"""
        # Normalize and weight different factors
        throughput_score = min(1.0, metrics.items_per_second / 100)  # Assume 100 items/sec is excellent
        success_rate_score = metrics.success_rate
        
        # Resource efficiency (lower usage is better within reason)
        cpu_score = 1.0 - abs(metrics.cpu_usage - self.target_cpu_usage * 0.7) / 100
        memory_score = 1.0 - abs(metrics.memory_usage - self.target_memory_usage * 0.7) / 100
        
        # Response time score
        if metrics.api_response_times:
            avg_response_time = statistics.mean(metrics.api_response_times)
            response_score = max(0, 1.0 - avg_response_time / (self.target_response_time * 2))
        else:
            response_score = 0.5
        
        # Weighted average
        weights = {
            'throughput': 0.3,
            'success_rate': 0.3,
            'cpu': 0.15,
            'memory': 0.15,
            'response': 0.1
        }
        
        score = (
            weights['throughput'] * throughput_score +
            weights['success_rate'] * success_rate_score +
            weights['cpu'] * cpu_score +
            weights['memory'] * memory_score +
            weights['response'] * response_score
        )
        
        return max(0, min(1, score))
    
    def record_batch_performance(self,
                               batch_size: int,
                               processing_time: float,
                               items_processed: int,
                               error_count: int,
                               api_response_times: Optional[List[float]] = None):
        """Record performance metrics for a completed batch"""
        with self.lock:
            # Get current system metrics
            system_metrics = self._get_system_metrics()
            self.system_metrics.append(system_metrics)
            
            # Calculate performance metrics
            items_per_second = items_processed / processing_time if processing_time > 0 else 0
            success_rate = (items_processed - error_count) / items_processed if items_processed > 0 else 0
            
            metrics = BatchPerformanceMetrics(
                batch_size=batch_size,
                processing_time=processing_time,
                items_per_second=items_per_second,
                error_count=error_count,
                memory_usage=system_metrics.memory_percent,
                cpu_usage=system_metrics.cpu_percent,
                timestamp=time.time(),
                success_rate=success_rate,
                api_response_times=api_response_times or []
            )
            
            self.performance_history.append(metrics)
            
            # Update batch size based on performance
            new_batch_size = self._calculate_optimal_batch_size()
            
            if new_batch_size != self.current_batch_size:
                logging.info(f"Adjusting batch size from {self.current_batch_size} to {new_batch_size}")
                self.current_batch_size = new_batch_size
            
            logging.debug(f"Recorded batch performance: {items_per_second:.2f} items/sec, "
                         f"{success_rate:.2%} success rate, {error_count} errors")
    
    def get_current_batch_size(self) -> int:
        """Get the current optimal batch size"""
        return self.current_batch_size
    
    def should_pause_processing(self) -> bool:
        """Check if processing should be paused due to resource constraints"""
        if not self.monitor_system_resources:
            return False
        
        system_metrics = self._get_system_metrics()
        
        # Pause if system resources are critically high
        if (system_metrics.cpu_percent > 95 or 
            system_metrics.memory_percent > 95 or
            system_metrics.available_memory < 100 * 1024 * 1024):  # Less than 100MB available
            
            logging.warning("Pausing processing due to high resource usage")
            return True
        
        return False
    
    async def process_uniprot_batch(self, 
                                  uniprot_ids: List[str],
                                  fetch_function: Callable,
                                  max_concurrent: Optional[int] = None) -> List[Any]:
        """
        Process a batch of UniProt IDs with adaptive batch sizing
        """
        if not uniprot_ids:
            return []
        
        # Use current batch size or provided max_concurrent
        batch_size = min(
            len(uniprot_ids),
            max_concurrent or self.current_batch_size
        )
        
        # Check if we should pause processing
        if self.should_pause_processing():
            await asyncio.sleep(2)  # Brief pause
        
        start_time = time.time()
        results = []
        errors = 0
        api_response_times = []
        
        try:
            # Process in batches
            for i in range(0, len(uniprot_ids), batch_size):
                batch = uniprot_ids[i:i + batch_size]
                
                # Process batch concurrently
                batch_start = time.time()
                
                tasks = []
                for uniprot_id in batch:
                    task_start = time.time()
                    try:
                        task = fetch_function(uniprot_id)
                        if asyncio.iscoroutine(task):
                            tasks.append(task)
                        else:
                            # Handle synchronous functions
                            result = task
                            results.append(result)
                            api_response_times.append(time.time() - task_start)
                    except Exception as e:
                        errors += 1
                        logging.error(f"Error processing {uniprot_id}: {e}")
                
                # Await async tasks
                if tasks:
                    try:
                        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for i, result in enumerate(batch_results):
                            if isinstance(result, Exception):
                                errors += 1
                                logging.error(f"Async error: {result}")
                            else:
                                results.append(result)
                            
                            api_response_times.append(time.time() - batch_start / len(batch))
                    
                    except Exception as e:
                        errors += len(tasks)
                        logging.error(f"Batch processing error: {e}")
                
                # Brief pause between batches if needed
                if self.should_pause_processing():
                    await asyncio.sleep(1)
        
        except Exception as e:
            logging.error(f"Critical error in batch processing: {e}")
            errors += len(uniprot_ids) - len(results)
        
        finally:
            # Record performance metrics
            total_time = time.time() - start_time
            self.record_batch_performance(
                batch_size=batch_size,
                processing_time=total_time,
                items_processed=len(uniprot_ids),
                error_count=errors,
                api_response_times=api_response_times
            )
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics"""
        with self.lock:
            if not self.performance_history:
                return {"status": "No performance data available"}
            
            recent_metrics = list(self.performance_history)
            
            return {
                "current_batch_size": self.current_batch_size,
                "strategy": self.strategy.value,
                "total_batches_processed": len(recent_metrics),
                "average_throughput": statistics.mean([m.items_per_second for m in recent_metrics]),
                "average_success_rate": statistics.mean([m.success_rate for m in recent_metrics]),
                "total_errors": sum(m.error_count for m in recent_metrics),
                "average_response_time": statistics.mean([
                    statistics.mean(m.api_response_times) if m.api_response_times else 0
                    for m in recent_metrics
                ]),
                "performance_trend": self._calculate_trend([
                    self._calculate_performance_score(m) for m in recent_metrics
                ]),
                "last_updated": datetime.fromtimestamp(recent_metrics[-1].timestamp).isoformat()
            }
    
    def export_performance_data(self, filepath: str):
        """Export performance data to JSON file"""
        with self.lock:
            data = {
                "configuration": {
                    "strategy": self.strategy.value,
                    "min_batch_size": self.min_batch_size,
                    "max_batch_size": self.max_batch_size,
                    "current_batch_size": self.current_batch_size
                },
                "performance_history": [
                    {
                        "batch_size": m.batch_size,
                        "processing_time": m.processing_time,
                        "items_per_second": m.items_per_second,
                        "error_count": m.error_count,
                        "success_rate": m.success_rate,
                        "cpu_usage": m.cpu_usage,
                        "memory_usage": m.memory_usage,
                        "timestamp": m.timestamp,
                        "avg_response_time": statistics.mean(m.api_response_times) if m.api_response_times else None
                    }
                    for m in self.performance_history
                ],
                "summary": self.get_performance_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"Performance data exported to {filepath}")

# Utility functions for easy integration

def create_adaptive_processor(strategy: str = "adaptive", 
                            initial_batch_size: int = 50) -> AdaptiveBatchProcessor:
    """Create an adaptive batch processor with specified strategy"""
    strategy_enum = BatchStrategy(strategy.lower())
    return AdaptiveBatchProcessor(
        initial_batch_size=initial_batch_size,
        strategy=strategy_enum
    )

async def adaptive_batch_fetch(items: List[Any],
                             fetch_function: Callable,
                             processor: Optional[AdaptiveBatchProcessor] = None) -> List[Any]:
    """Convenience function for adaptive batch processing"""
    if processor is None:
        processor = create_adaptive_processor()
    
    if asyncio.iscoroutinefunction(fetch_function):
        return await processor.process_uniprot_batch(items, fetch_function)
    else:
        # Wrap synchronous function for async processing
        async def async_wrapper(item):
            return fetch_function(item)
        
        return await processor.process_uniprot_batch(items, async_wrapper)