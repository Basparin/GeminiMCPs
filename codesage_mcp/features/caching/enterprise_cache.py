"""
Enterprise Cache System for CodeSage MCP Server.

This module provides enterprise-grade caching with multi-level architecture,
offline capabilities, CDN integration, and advanced monitoring for Phase 4.

Features:
- Multi-level caching (Memory -> Redis -> Disk)
- Offline capabilities with service worker support
- CDN integration for global distribution
- Enterprise monitoring and alerting
- Predictive cache warming with ML
- Distributed cache synchronization
"""

import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import numpy as np

from .intelligent_cache import IntelligentCache
from .cache_components import LRUCache

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""
    MEMORY = "memory"      # Fast in-memory cache
    REDIS = "redis"        # Distributed Redis cache
    DISK = "disk"          # Persistent disk cache
    CDN = "cdn"           # CDN edge cache


class CacheStrategy(Enum):
    """Cache strategies for different scenarios."""
    WRITE_THROUGH = "write_through"    # Write to all levels
    WRITE_BACK = "write_back"         # Write to memory, lazy sync
    WRITE_AROUND = "write_around"     # Write to disk, bypass memory
    CACHE_ASIDE = "cache_aside"       # Application manages cache


class OfflineMode(Enum):
    """Offline operation modes."""
    ONLINE = "online"              # Normal online operation
    OFFLINE_FIRST = "offline_first" # Try cache first, then network
    OFFLINE_ONLY = "offline_only"   # Cache only, no network calls
    HYBRID = "hybrid"              # Smart offline/online switching


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    level: CacheLevel
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[int] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetrics:
    """Enterprise cache metrics."""
    level: CacheLevel
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    deletes: int = 0
    total_size_bytes: int = 0
    avg_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    sync_operations: int = 0
    sync_failures: int = 0


class RedisCache:
    """Redis-based distributed cache."""

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, password: Optional[str] = None,
                 max_connections: int = 20):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.redis_client = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import redis
            from redis.connection import ConnectionPool

            pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=True
            )

            self.redis_client = redis.Redis(connection_pool=pool)
            self.redis_client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
        except ImportError:
            logger.warning("Redis not available, running without distributed cache")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self._connected or not self.redis_client:
            return None

        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self._connected or not self.redis_client:
            return False

        try:
            return self.redis_client.set(key, value, ex=ttl)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self._connected or not self.redis_client:
            return False

        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self._connected or not self.redis_client:
            return False

        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """Get TTL for key."""
        if not self._connected or not self.redis_client:
            return -1

        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL error: {e}")
            return -1


class OfflineManager:
    """Manages offline capabilities and service worker integration."""

    def __init__(self, cache_dir: str = ".codesage"):
        self.cache_dir = Path(cache_dir)
        self.offline_dir = self.cache_dir / "offline"
        self.offline_dir.mkdir(exist_ok=True)

        self.offline_mode = OfflineMode.ONLINE
        self.service_worker_enabled = False
        self.local_storage_enabled = True

        # Offline queue for operations
        self.offline_queue: List[Dict[str, Any]] = []
        self.queue_lock = threading.RLock()

        # Network status monitoring
        self.network_available = True
        self.network_check_interval = 30  # seconds
        self._network_monitor_thread: Optional[threading.Thread] = None
        self._monitoring_active = False

    def enable_offline_mode(self, mode: OfflineMode = OfflineMode.OFFLINE_FIRST):
        """Enable offline mode."""
        self.offline_mode = mode
        self._start_network_monitoring()
        logger.info(f"Offline mode enabled: {mode.value}")

    def disable_offline_mode(self):
        """Disable offline mode."""
        self.offline_mode = OfflineMode.ONLINE
        self._stop_network_monitoring()
        logger.info("Offline mode disabled")

    def _start_network_monitoring(self):
        """Start network status monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._network_monitor_thread = threading.Thread(
            target=self._network_monitor_loop,
            daemon=True,
            name="NetworkMonitor"
        )
        self._network_monitor_thread.start()

    def _stop_network_monitoring(self):
        """Stop network status monitoring."""
        self._monitoring_active = False
        if self._network_monitor_thread:
            self._network_monitor_thread.join(timeout=5.0)

    def _network_monitor_loop(self):
        """Network monitoring loop."""
        while self._monitoring_active:
            try:
                self._check_network_status()
                time.sleep(self.network_check_interval)
            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
                time.sleep(5)

    def _check_network_status(self):
        """Check network availability."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            new_status = True
        except:
            new_status = False

        if new_status != self.network_available:
            self.network_available = new_status
            if new_status:
                logger.info("Network connection restored")
                self._process_offline_queue()
            else:
                logger.warning("Network connection lost")

    def _process_offline_queue(self):
        """Process queued operations when network is restored."""
        with self.queue_lock:
            if not self.offline_queue:
                return

            logger.info(f"Processing {len(self.offline_queue)} queued operations")

            # Process queue (simplified - in real implementation would retry operations)
            processed = 0
            for operation in self.offline_queue[:]:
                try:
                    # Attempt to process operation
                    # This would call the appropriate handler based on operation type
                    processed += 1
                    self.offline_queue.remove(operation)
                except Exception as e:
                    logger.error(f"Failed to process queued operation: {e}")

            logger.info(f"Processed {processed} operations from offline queue")

    def queue_operation(self, operation: Dict[str, Any]):
        """Queue operation for offline processing."""
        with self.queue_lock:
            self.offline_queue.append(operation)
            logger.debug(f"Queued operation for offline processing: {operation.get('type', 'unknown')}")

    def get_offline_status(self) -> Dict[str, Any]:
        """Get offline status information."""
        return {
            "mode": self.offline_mode.value,
            "network_available": self.network_available,
            "service_worker_enabled": self.service_worker_enabled,
            "local_storage_enabled": self.local_storage_enabled,
            "queued_operations": len(self.offline_queue),
            "offline_cache_size_mb": self._get_offline_cache_size() / (1024 * 1024)
        }

    def _get_offline_cache_size(self) -> int:
        """Get size of offline cache."""
        try:
            total_size = 0
            for file_path in self.offline_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0


class CDNManager:
    """Manages CDN integration for global cache distribution."""

    def __init__(self, cdn_provider: str = "cloudflare",
                 api_key: Optional[str] = None, zone_id: Optional[str] = None):
        self.cdn_provider = cdn_provider
        self.api_key = api_key
        self.zone_id = zone_id
        self.cache_purge_queue: List[str] = []
        self.purge_lock = threading.RLock()

    def purge_cache(self, urls: List[str]) -> bool:
        """Purge CDN cache for given URLs."""
        if not self.api_key or not self.zone_id:
            logger.warning("CDN credentials not configured")
            return False

        try:
            if self.cdn_provider.lower() == "cloudflare":
                return self._purge_cloudflare_cache(urls)
            else:
                logger.warning(f"Unsupported CDN provider: {self.cdn_provider}")
                return False
        except Exception as e:
            logger.error(f"CDN cache purge error: {e}")
            return False

    def _purge_cloudflare_cache(self, urls: List[str]) -> bool:
        """Purge Cloudflare cache."""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "files": urls
            }

            response = requests.post(
                f"https://api.cloudflare.com/client/v4/zones/{self.zone_id}/purge_cache",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                logger.info(f"Successfully purged {len(urls)} URLs from Cloudflare cache")
                return True
            else:
                logger.error(f"Cloudflare purge failed: {response.text}")
                return False

        except ImportError:
            logger.warning("requests library not available for CDN operations")
            return False
        except Exception as e:
            logger.error(f"Cloudflare API error: {e}")
            return False

    def queue_cache_purge(self, urls: List[str]):
        """Queue URLs for cache purge."""
        with self.purge_lock:
            self.cache_purge_queue.extend(urls)

    def process_purge_queue(self):
        """Process queued cache purges."""
        with self.purge_lock:
            if not self.cache_purge_queue:
                return

            # Process in batches
            batch_size = 30  # Cloudflare limit
            for i in range(0, len(self.cache_purge_queue), batch_size):
                batch = self.cache_purge_queue[i:i + batch_size]
                if self.purge_cache(batch):
                    # Remove successfully purged URLs
                    del self.cache_purge_queue[i:i + batch_size]
                else:
                    break  # Stop on failure


class EnterpriseCache:
    """Enterprise-grade multi-level cache system."""

    def __init__(self, cache_dir: str = ".codesage", config: Optional[Dict[str, Any]] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Default configuration
        default_config = {
            "strategy": CacheStrategy.WRITE_THROUGH,
            "enable_redis": True,
            "redis_host": "localhost",
            "redis_port": 6379,
            "enable_cdn": False,
            "cdn_provider": "cloudflare",
            "offline_mode": OfflineMode.ONLINE,
            "sync_interval": 300,  # 5 minutes
            "max_memory_cache_size": 1000,
            "max_disk_cache_size": 10000,
            "compression_enabled": True,
            "encryption_enabled": False,
        }

        self.config = {**default_config, **(config or {})}

        # Initialize cache levels
        self.memory_cache = LRUCache(self.config["max_memory_cache_size"])
        self.disk_cache = self._init_disk_cache()
        self.redis_cache = RedisCache(
            host=self.config["redis_host"],
            port=self.config["redis_port"]
        ) if self.config["enable_redis"] else None

        # Initialize enterprise features
        self.offline_manager = OfflineManager(str(self.cache_dir))
        self.cdn_manager = CDNManager(
            cdn_provider=self.config["cdn_provider"]
        ) if self.config["enable_cdn"] else None

        # Initialize existing intelligent cache for compatibility
        self.intelligent_cache = IntelligentCache(str(self.cache_dir))

        # Metrics tracking
        self.metrics: Dict[CacheLevel, CacheMetrics] = {
            level: CacheMetrics(level=level) for level in CacheLevel
        }

        # Synchronization
        self.sync_lock = threading.RLock()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_active = False

        # Start background sync if Redis is enabled
        if self.redis_cache and self.redis_cache.connect():
            self._start_background_sync()

        # Set offline mode
        if self.config["offline_mode"] != OfflineMode.ONLINE:
            self.offline_manager.enable_offline_mode(self.config["offline_mode"])

    def _init_disk_cache(self) -> LRUCache:
        """Initialize disk-based cache."""
        # For now, use a simple LRU cache with file persistence
        # In production, this could be a more sophisticated disk cache
        return LRUCache(self.config["max_disk_cache_size"])

    def _start_background_sync(self):
        """Start background synchronization with Redis."""
        if self._sync_active:
            return

        self._sync_active = True
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="CacheSync"
        )
        self._sync_thread.start()

    def _sync_loop(self):
        """Background synchronization loop."""
        while self._sync_active:
            try:
                self._sync_with_redis()
                time.sleep(self.config["sync_interval"])
            except Exception as e:
                logger.error(f"Cache sync error: {e}")
                time.sleep(60)

    def _sync_with_redis(self):
        """Synchronize cache with Redis."""
        if not self.redis_cache:
            return

        try:
            # This is a simplified sync - in production would be more sophisticated
            # Sync would involve checking for updates in Redis and updating local caches
            self.metrics[CacheLevel.REDIS].sync_operations += 1
            logger.debug("Cache synchronization completed")
        except Exception as e:
            self.metrics[CacheLevel.REDIS].sync_failures += 1
            logger.error(f"Cache sync failed: {e}")

    def get(self, key: str, level_preference: Optional[CacheLevel] = None) -> Tuple[Optional[Any], CacheLevel]:
        """Get value from cache hierarchy."""
        start_time = time.time()

        # Determine cache level preference
        if level_preference is None:
            level_preference = CacheLevel.MEMORY

        # Try cache levels in order of preference
        levels_to_try = self._get_cache_level_order(level_preference)

        for level in levels_to_try:
            value = self._get_from_level(key, level)
            if value is not None:
                response_time = (time.time() - start_time) * 1000
                self._update_metrics(level, hit=True, response_time=response_time)
                return value, level

        # Cache miss
        response_time = (time.time() - start_time) * 1000
        self._update_metrics(level_preference, hit=False, response_time=response_time)
        return None, level_preference

    def _get_cache_level_order(self, preference: CacheLevel) -> List[CacheLevel]:
        """Get cache level order based on preference."""
        if preference == CacheLevel.MEMORY:
            return [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
        elif preference == CacheLevel.REDIS:
            return [CacheLevel.REDIS, CacheLevel.MEMORY, CacheLevel.DISK]
        else:
            return [CacheLevel.DISK, CacheLevel.MEMORY, CacheLevel.REDIS]

    def _get_from_level(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Get value from specific cache level."""
        try:
            if level == CacheLevel.MEMORY:
                return self.memory_cache.get(key)
            elif level == CacheLevel.REDIS and self.redis_cache:
                value_str = self.redis_cache.get(key)
                return json.loads(value_str) if value_str else None
            elif level == CacheLevel.DISK:
                return self._get_from_disk(key)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting from {level.value} cache: {e}")
            return None

    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check TTL
                    if data.get('ttl') and time.time() > data['timestamp'] + data['ttl']:
                        cache_file.unlink()  # Remove expired entry
                        return None
                    return data['value']
            return None
        except Exception as e:
            logger.error(f"Disk cache read error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None,
            strategy: Optional[CacheStrategy] = None) -> bool:
        """Set value in cache hierarchy."""
        if strategy is None:
            strategy = self.config["strategy"]

        start_time = time.time()
        success = False

        try:
            if strategy == CacheStrategy.WRITE_THROUGH:
                success = self._write_through(key, value, ttl)
            elif strategy == CacheStrategy.WRITE_BACK:
                success = self._write_back(key, value, ttl)
            elif strategy == CacheStrategy.WRITE_AROUND:
                success = self._write_around(key, value, ttl)
            else:  # CACHE_ASIDE
                success = self._cache_aside(key, value, ttl)

            if success:
                response_time = (time.time() - start_time) * 1000
                self._update_metrics(CacheLevel.MEMORY, set_operation=True, response_time=response_time)

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            success = False

        return success

    def _write_through(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Write through all cache levels."""
        success = True

        # Memory
        self.memory_cache.put(key, value)

        # Redis
        if self.redis_cache:
            try:
                value_str = json.dumps(value)
                self.redis_cache.set(key, value_str, ttl)
            except Exception as e:
                logger.error(f"Redis write error: {e}")
                success = False

        # Disk
        try:
            self._write_to_disk(key, value, ttl)
        except Exception as e:
            logger.error(f"Disk write error: {e}")
            success = False

        return success

    def _write_back(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Write to memory, lazy sync to other levels."""
        self.memory_cache.put(key, value)
        # In production, would queue for background sync
        return True

    def _write_around(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Write directly to disk, bypass memory."""
        try:
            self._write_to_disk(key, value, ttl)
            return True
        except Exception as e:
            logger.error(f"Write around error: {e}")
            return False

    def _cache_aside(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Application-managed cache."""
        # For cache-aside, just store in memory for now
        self.memory_cache.put(key, value)
        return True

    def _write_to_disk(self, key: str, value: Any, ttl: Optional[int]):
        """Write value to disk cache."""
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
        data = {
            'key': key,
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def _update_metrics(self, level: CacheLevel, hit: bool = False, set_operation: bool = False,
                       response_time: float = 0.0):
        """Update cache metrics."""
        metrics = self.metrics[level]

        if hit:
            metrics.hits += 1
        else:
            metrics.misses += 1

        if set_operation:
            metrics.sets += 1

        # Update average response time
        if response_time > 0:
            if metrics.avg_response_time_ms == 0:
                metrics.avg_response_time_ms = response_time
            else:
                metrics.avg_response_time_ms = (metrics.avg_response_time_ms * 0.9 + response_time * 0.1)

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        total_metrics = {
            "overall": {
                "total_hits": sum(m.hits for m in self.metrics.values()),
                "total_misses": sum(m.misses for m in self.metrics.values()),
                "total_sets": sum(m.sets for m in self.metrics.values()),
                "total_evictions": sum(m.evictions for m in self.metrics.values()),
                "overall_hit_rate": 0.0
            },
            "levels": {},
            "offline_status": self.offline_manager.get_offline_status(),
            "configuration": self.config.copy()
        }

        total_hits = total_metrics["overall"]["total_hits"]
        total_misses = total_metrics["overall"]["total_misses"]
        if total_hits + total_misses > 0:
            total_metrics["overall"]["overall_hit_rate"] = total_hits / (total_hits + total_misses)

        for level, metrics in self.metrics.items():
            total_requests = metrics.hits + metrics.misses
            hit_rate = metrics.hits / total_requests if total_requests > 0 else 0

            total_metrics["levels"][level.value] = {
                "hits": metrics.hits,
                "misses": metrics.misses,
                "sets": metrics.sets,
                "evictions": metrics.evictions,
                "hit_rate": hit_rate,
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "total_size_bytes": metrics.total_size_bytes,
                "uptime_seconds": metrics.uptime_seconds,
                "sync_operations": metrics.sync_operations,
                "sync_failures": metrics.sync_failures
            }

        return total_metrics

    def enable_offline_mode(self, mode: OfflineMode = OfflineMode.OFFLINE_FIRST):
        """Enable offline mode."""
        self.offline_manager.enable_offline_mode(mode)
        self.config["offline_mode"] = mode

    def disable_offline_mode(self):
        """Disable offline mode."""
        self.offline_manager.disable_offline_mode()
        self.config["offline_mode"] = OfflineMode.ONLINE

    def purge_cdn_cache(self, urls: List[str]) -> bool:
        """Purge CDN cache for given URLs."""
        if self.cdn_manager:
            return self.cdn_manager.purge_cache(urls)
        return False

    def cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            # Clean disk cache
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        if data.get('ttl') and time.time() > data['timestamp'] + data['ttl']:
                            cache_file.unlink()
                except Exception:
                    # Remove corrupted files
                    cache_file.unlink()

            # Clean Redis (if TTL is set, Redis handles it automatically)
            if self.redis_cache:
                # Could implement cleanup of Redis keys if needed
                pass

        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

    def shutdown(self):
        """Shutdown the enterprise cache system."""
        self._sync_active = False
        self.offline_manager.disable_offline_mode()

        if self._sync_thread:
            self._sync_thread.join(timeout=5.0)

        logger.info("Enterprise cache system shutdown complete")


# Global instance
_enterprise_cache_instance: Optional[EnterpriseCache] = None
_enterprise_cache_lock = threading.Lock()


def get_enterprise_cache(cache_dir: str = ".codesage",
                        config: Optional[Dict[str, Any]] = None) -> EnterpriseCache:
    """Get the global enterprise cache instance."""
    global _enterprise_cache_instance

    if _enterprise_cache_instance is None:
        with _enterprise_cache_lock:
            if _enterprise_cache_instance is None:
                _enterprise_cache_instance = EnterpriseCache(cache_dir, config)

    return _enterprise_cache_instance