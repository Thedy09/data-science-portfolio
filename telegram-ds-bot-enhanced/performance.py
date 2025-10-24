"""
Improved Performance Management Module
"""
import asyncio
import time
import psutil
import logging
from functools import wraps
from typing import Dict, Any, Optional, List, Tuple, Callable
import json
import os
from datetime import datetime, timedelta
import concurrent.futures
from threading import Lock
import gc

logger = logging.getLogger(__name__)

# Constants for performance tuning
MAX_CONCURRENT_ANALYSES = 3
MEMORY_THRESHOLD_WARNING = 80
MEMORY_THRESHOLD_CRITICAL = 90
CPU_THRESHOLD_WARNING = 85
CPU_THRESHOLD_CRITICAL = 95
ANALYSIS_TIMEOUT = 600  # 10 minutes
CACHE_TTL = 3600  # 1 hour
MAX_RETRIES = 3

class EnhancedPerformanceMonitor:
    """Enhanced performance monitoring with adaptive thresholds"""
    
    def __init__(self):
        self.metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'timeouts': 0,
            'average_processing_time': 0,
            'memory_usage': [],
            'cpu_usage': [],
            'response_times': []
        }
        self.start_time = time.time()
        self._lock = Lock()
        self.adaptive_thresholds = {
            'memory': MEMORY_THRESHOLD_WARNING,
            'cpu': CPU_THRESHOLD_WARNING
        }
    
    def record_analysis(self, success: bool, processing_time: float, error_type: Optional[str] = None):
        """Record analysis metrics with thread safety"""
        with self._lock:
            self.metrics['total_analyses'] += 1
            if success:
                self.metrics['successful_analyses'] += 1
            else:
                self.metrics['failed_analyses'] += 1
                if error_type == 'timeout':
                    self.metrics['timeouts'] += 1
            
            # Update average processing time with moving average
            alpha = 0.1  # Smoothing factor
            current_avg = self.metrics['average_processing_time']
            self.metrics['average_processing_time'] = (alpha * processing_time) + ((1 - alpha) * current_avg)
            
            # Record system metrics
            self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
            self.metrics['cpu_usage'].append(psutil.cpu_percent())
            self.metrics['response_times'].append(processing_time)
            
            # Keep only recent measurements
            max_history = 100
            for metric in ['memory_usage', 'cpu_usage', 'response_times']:
                if len(self.metrics[metric]) > max_history:
                    self.metrics[metric] = self.metrics[metric][-max_history:]
            
            # Adjust thresholds based on performance
            self._adjust_thresholds()
    
    def _adjust_thresholds(self):
        """Dynamically adjust thresholds based on system performance"""
        if len(self.metrics['response_times']) >= 10:
            recent_times = self.metrics['response_times'][-10:]
            if sum(t > ANALYSIS_TIMEOUT for t in recent_times) >= 3:
                # If we see multiple timeouts, lower thresholds
                self.adaptive_thresholds['memory'] = max(70, self.adaptive_thresholds['memory'] - 5)
                self.adaptive_thresholds['cpu'] = max(75, self.adaptive_thresholds['cpu'] - 5)
            elif all(t < ANALYSIS_TIMEOUT/2 for t in recent_times):
                # If performance is good, gradually increase thresholds
                self.adaptive_thresholds['memory'] = min(MEMORY_THRESHOLD_WARNING, self.adaptive_thresholds['memory'] + 1)
                self.adaptive_thresholds['cpu'] = min(CPU_THRESHOLD_WARNING, self.adaptive_thresholds['cpu'] + 1)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status with predictions"""
        with self._lock:
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()
            
            # Predict trend based on recent history
            memory_trend = self._calculate_trend(self.metrics['memory_usage'][-10:])
            cpu_trend = self._calculate_trend(self.metrics['cpu_usage'][-10:])
            
            return {
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'memory_trend': memory_trend,
                'cpu_trend': cpu_trend,
                'disk_percent': psutil.disk_usage('/').percent,
                'uptime': time.time() - self.start_time,
                'total_analyses': self.metrics['total_analyses'],
                'success_rate': self.metrics['successful_analyses'] / max(1, self.metrics['total_analyses']),
                'timeout_rate': self.metrics['timeouts'] / max(1, self.metrics['total_analyses']),
                'avg_processing_time': self.metrics['average_processing_time'],
                'adaptive_thresholds': self.adaptive_thresholds.copy()
            }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from recent values"""
        if len(values) < 2:
            return "stable"
        
        diff = values[-1] - values[0]
        if abs(diff) < 1:
            return "stable"
        return "increasing" if diff > 0 else "decreasing"
    
    def should_throttle(self) -> Tuple[bool, str]:
        """Enhanced throttling decision with reason"""
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent()
        
        if memory_percent > MEMORY_THRESHOLD_CRITICAL:
            return True, "Critical memory usage"
        if cpu_percent > CPU_THRESHOLD_CRITICAL:
            return True, "Critical CPU usage"
        
        # Check adaptive thresholds
        if memory_percent > self.adaptive_thresholds['memory']:
            return True, "High memory usage (adaptive)"
        if cpu_percent > self.adaptive_thresholds['cpu']:
            return True, "High CPU usage (adaptive)"
        
        # Check if too many timeouts recently
        recent_timeouts = sum(1 for t in self.metrics['response_times'][-10:]
                            if t > ANALYSIS_TIMEOUT)
        if recent_timeouts >= 3:
            return True, "Too many recent timeouts"
        
        return False, "OK"

# Global performance monitor
perf_monitor = EnhancedPerformanceMonitor()

class TimeoutManager:
    """Manage timeouts for long-running operations"""
    
    def __init__(self, timeout: int = ANALYSIS_TIMEOUT):
        self.timeout = timeout
    
    async def run_with_timeout(self, func: Callable, *args, timeout: Optional[int] = None, **kwargs) -> Any:
        """Run a sync or async function with timeout and retry logic.

        - If `func` is a coroutine function, it will be awaited directly under a timeout.
        - If `func` is a regular function, it will be executed in a ThreadPoolExecutor and awaited.
        - Supports per-call `timeout` override.
        - Retries up to MAX_RETRIES on timeout.
        """
        effective_timeout = timeout or self.timeout

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    coro = func(*args, **kwargs)
                    return await asyncio.wait_for(coro, timeout=effective_timeout)
                else:
                    # Run synchronous/blocking function in threadpool
                    loop = asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = loop.run_in_executor(pool, lambda: func(*args, **kwargs))
                        return await asyncio.wait_for(future, timeout=effective_timeout)

            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt} for {getattr(func, '__name__', repr(func))}")
                if attempt >= MAX_RETRIES:
                    # Raise a TimeoutError that caller can catch
                    raise
                # brief jittered backoff before retrying
                await asyncio.sleep(min(2 ** attempt, 5))
                continue
            except Exception as e:
                logger.error(f"Error in {getattr(func, '__name__', repr(func))}: {e}")
                raise

# Global timeout manager
timeout_manager = TimeoutManager()

class EnhancedResourceManager:
    """Enhanced resource management with predictive scaling"""
    
    def __init__(self):
        self.active_tasks = {}
        self.max_concurrent = MAX_CONCURRENT_ANALYSES
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self._lock = Lock()
    
    async def can_start_analysis(self, user_id: int) -> Tuple[bool, str]:
        """Check if we can start a new analysis with detailed reason"""
        with self._lock:
            # Check system resources
            should_throttle, reason = perf_monitor.should_throttle()
            if should_throttle:
                return False, f"System load too high: {reason}"
            
            # Clean up old tasks first
            self._cleanup_old_tasks()
            
            # Check concurrent limit
            if len(self.active_tasks) >= self.max_concurrent:
                return False, f"Max concurrent analyses ({self.max_concurrent}) reached"
            
            # Check if user already has active analysis
            if user_id in self.active_tasks:
                task_age = time.time() - self.active_tasks[user_id]['start_time']
                if task_age > ANALYSIS_TIMEOUT:
                    # Auto-cleanup stuck task
                    self.end_analysis(user_id)
                else:
                    return False, f"User already has active analysis ({task_age:.1f}s old)"
            
            return True, "OK"
    
    def start_analysis(self, user_id: int, task_id: str):
        """Register start of analysis with enhanced monitoring"""
        with self._lock:
            self.active_tasks[user_id] = {
                'task_id': task_id,
                'start_time': time.time(),
                'status': 'processing',
                'memory_start': psutil.virtual_memory().percent,
                'cpu_start': psutil.cpu_percent()
            }
    
    def end_analysis(self, user_id: int):
        """Register end of analysis with resource impact tracking"""
        with self._lock:
            if user_id in self.active_tasks:
                task = self.active_tasks[user_id]
                duration = time.time() - task['start_time']
                memory_impact = psutil.virtual_memory().percent - task['memory_start']
                cpu_impact = psutil.cpu_percent() - task['cpu_start']
                
                logger.info(
                    f"Analysis completed - Duration: {duration:.1f}s, "
                    f"Memory impact: {memory_impact:+.1f}%, "
                    f"CPU impact: {cpu_impact:+.1f}%"
                )
                
                del self.active_tasks[user_id]
    
    def _cleanup_old_tasks(self):
        """Clean up old or stuck tasks with improved logging"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        stuck_tasks = []
        
        for user_id, task_info in self.active_tasks.items():
            task_age = current_time - task_info['start_time']
            if task_age > ANALYSIS_TIMEOUT:
                stuck_tasks.append((user_id, task_age))
        
        for user_id, age in stuck_tasks:
            logger.warning(f"Cleaning up stuck task for user {user_id} (age: {age:.1f}s)")
            self.end_analysis(user_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed resource manager status"""
        with self._lock:
            self._cleanup_old_tasks()
            return {
                'active_tasks': len(self.active_tasks),
                'max_concurrent': self.max_concurrent,
                'can_accept_new': len(self.active_tasks) < self.max_concurrent,
                'active_users': list(self.active_tasks.keys()),
                'task_ages': {
                    user_id: time.time() - info['start_time']
                    for user_id, info in self.active_tasks.items()
                }
            }

# Global resource manager
resource_manager = EnhancedResourceManager()

class EnhancedCacheManager:
    """Enhanced caching with size-based and time-based eviction"""
    
    def __init__(self, max_size: int = 100, ttl: int = CACHE_TTL):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with metrics"""
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.hits += 1
                    return value
                else:
                    del self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with intelligent eviction"""
        with self._lock:
            # Clean expired entries first
            self._cleanup_expired()
            
            # If still at capacity, use LRU eviction
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (value, time.time())
    
    def _cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired = [k for k, (_, t) in self.cache.items()
                  if current_time - t > self.ttl]
        for k in expired:
            del self.cache[k]
    
    def _evict_lru(self):
        """Evict least recently used cache entry"""
        if self.cache:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses
            }

    def clear(self):
        """Clear the cache contents safely."""
        with self._lock:
            self.cache.clear()
            # Reset simple metrics
            self.hits = 0
            self.misses = 0

# Global cache manager
cache_manager = EnhancedCacheManager()

async def optimize_memory():
    """Aggressive memory optimization"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear cache if memory is high
        if psutil.virtual_memory().percent > MEMORY_THRESHOLD_WARNING:
            cache_manager.clear()
            gc.collect()
        
        logger.info("Memory optimization completed")
    except Exception as e:
        logger.error(f"Error during memory optimization: {e}")

async def check_system_health() -> Dict[str, Any]:
    """Enhanced system health check with predictions"""
    try:
        status = perf_monitor.get_system_status()
        resource_status = resource_manager.get_status()
        cache_stats = cache_manager.get_stats()
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'performance': status,
            'resources': resource_status,
            'cache': cache_stats,
            'recommendations': []
        }
        
        # Determine health status and generate recommendations
        if status['memory_percent'] > MEMORY_THRESHOLD_CRITICAL or status['cpu_percent'] > CPU_THRESHOLD_CRITICAL:
            health['status'] = 'critical'
            health['recommendations'].append("Consider restarting the service")
            health['recommendations'].append("Reduce max concurrent analyses")
        elif status['memory_percent'] > MEMORY_THRESHOLD_WARNING or status['cpu_percent'] > CPU_THRESHOLD_WARNING:
            health['status'] = 'warning'
            health['recommendations'].append("Monitor system closely")
            
        if status['timeout_rate'] > 0.1:  # More than 10% timeouts
            health['recommendations'].append("Consider increasing analysis timeout")
        
        if cache_stats['hit_rate'] < 0.5:  # Less than 50% cache hits
            health['recommendations'].append("Review cache strategy")
        
        return health
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def monitor_performance(func):
    """Enhanced performance monitoring decorator"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        success = False
        error_type = None

        try:
            # If func is a coroutine function, call it directly (don't run coroutine in threadpool)
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # For regular (sync) functions, use the timeout manager which runs them in a threadpool
                result = await timeout_manager.run_with_timeout(func, *args, **kwargs)

            success = True
            return result
        except asyncio.TimeoutError:
            error_type = 'timeout'
            logger.error(f"Timeout in {func.__name__}")
            raise
        except Exception as e:
            error_type = 'error'
            logger.error(f"Error in {func.__name__}: {e}")
            raise
        finally:
            processing_time = time.time() - start_time
            try:
                perf_monitor.record_analysis(success, processing_time, error_type)
            except Exception:
                logger.exception("Failed recording performance metrics")

            if processing_time > ANALYSIS_TIMEOUT * 0.8:  # Log if close to timeout
                logger.warning(
                    f"Operation {func.__name__} took {processing_time:.2f}s "
                    f"(timeout: {ANALYSIS_TIMEOUT}s)"
                )
    
    return wrapper

def log_performance_metrics():
    """Enhanced performance metrics logging"""
    try:
        status = perf_monitor.get_system_status()
        logger.info("Performance Metrics:")
        logger.info(f"Memory: {status['memory_percent']:.1f}% ({status['memory_trend']})")
        logger.info(f"CPU: {status['cpu_percent']:.1f}% ({status['cpu_trend']})")
        logger.info(f"Success Rate: {status['success_rate']*100:.1f}%")
        logger.info(f"Timeout Rate: {status['timeout_rate']*100:.1f}%")
        logger.info(f"Avg Processing Time: {status['avg_processing_time']:.1f}s")
        logger.info(f"Active Tasks: {resource_manager.get_status()['active_tasks']}")
        logger.info(f"Cache Hit Rate: {cache_manager.get_stats()['hit_rate']*100:.1f}%")
    except Exception as e:
        logger.error(f"Error logging performance metrics: {e}")