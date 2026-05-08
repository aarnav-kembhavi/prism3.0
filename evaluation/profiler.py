import os
import time
import threading
import psutil

class BackgroundProfiler:
    """
    Lightweight resource monitor that tracks CPU and Memory usage
    of the current process asynchronously.
    """
    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.cpu_samples = []
        self.mem_samples = []
        self.start_time = None
        self.end_time = None
        
        self._stop_event = threading.Event()
        self._thread = None

    def _monitor_loop(self):
        # Initial call to psutil.cpu_percent() sets the baseline
        self.process.cpu_percent(interval=None)
        
        while not self._stop_event.is_set():
            try:
                # Get CPU usage since last call (returns float %)
                cpu = self.process.cpu_percent(interval=None)
                # Get RSS memory in MB
                mem_mb = self.process.memory_info().rss / (1024 * 1024)
                
                self.cpu_samples.append(cpu)
                self.mem_samples.append(mem_mb)
            except psutil.NoSuchProcess:
                break
                
            self._stop_event.wait(self.interval)

    def start(self):
        """Starts the background profiling thread."""
        self.cpu_samples.clear()
        self.mem_samples.clear()
        self._stop_event.clear()
        self.start_time = time.perf_counter()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the profiling thread and returns the aggregated metrics."""
        self.end_time = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            
        latency = self.end_time - self.start_time
        
        if not self.cpu_samples:
            self.cpu_samples = [0.0]
        if not self.mem_samples:
            self.mem_samples = [0.0]
            
        metrics = {
            "latency_sec": round(latency, 2),
            "cpu_mean_pct": round(sum(self.cpu_samples) / len(self.cpu_samples), 1),
            "cpu_peak_pct": round(max(self.cpu_samples), 1),
            "mem_mean_mb": round(sum(self.mem_samples) / len(self.mem_samples), 1),
            "mem_peak_mb": round(max(self.mem_samples), 1)
        }
        return metrics
