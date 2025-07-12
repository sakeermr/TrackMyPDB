"""
TrackMyPDB Intelligent Caching System
@author Anu Gamage

Multi-level caching system with memory cache, Redis cache, and persistent storage
for molecular data with automatic cleanup and cache invalidation strategies.
Licensed under MIT License - Open Source Project
"""

import sqlite3
import hashlib
import json
import time
import pickle
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import OrderedDict
import streamlit as st
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    st.warning("⚠️ Redis not available - using memory and disk cache only")

class CacheLevel(Enum):
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"

@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(key)
                entry.access_count += 1
                entry.last_accessed = time.time()
                self.cache[key] = entry
                return entry
            return None
    
    def put(self, key: str, entry: CacheEntry):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            entry.last_accessed = time.time()
            self.cache[key] = entry
    
    def remove(self, key: str):
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        return len(self.cache)

class SqliteCache:
    """Persistent SQLite cache for molecular data"""
    
    def __init__(self, db_path: str = "trackmypdb_cache.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp REAL,
                    ttl REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    size_bytes INTEGER,
                    cache_type TEXT
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
            ''')
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT data, timestamp, ttl, access_count, last_accessed, size_bytes FROM cache_entries WHERE key = ?',
                        (key,)
                    )
                    row = cursor.fetchone()
                    if row:
                        data, timestamp, ttl, access_count, last_accessed, size_bytes = row
                        entry = CacheEntry(
                            data=pickle.loads(data),
                            timestamp=timestamp,
                            ttl=ttl,
                            access_count=access_count,
                            last_accessed=last_accessed,
                            size_bytes=size_bytes
                        )
                        
                        # Update access statistics
                        entry.access_count += 1
                        entry.last_accessed = time.time()
                        self._update_access_stats(key, entry.access_count, entry.last_accessed)
                        
                        return entry
            except Exception as e:
                logging.error(f"Error reading from SQLite cache: {e}")
            return None
    
    def put(self, key: str, entry: CacheEntry, cache_type: str = "molecular"):
        with self.lock:
            try:
                serialized_data = pickle.dumps(entry.data)
                entry.size_bytes = len(serialized_data)
                entry.last_accessed = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (key, data, timestamp, ttl, access_count, last_accessed, size_bytes, cache_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (key, serialized_data, entry.timestamp, entry.ttl, entry.access_count, 
                         entry.last_accessed, entry.size_bytes, cache_type))
            except Exception as e:
                logging.error(f"Error writing to SQLite cache: {e}")
    
    def _update_access_stats(self, key: str, access_count: int, last_accessed: float):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE cache_entries SET access_count = ?, last_accessed = ? WHERE key = ?',
                    (access_count, last_accessed, key)
                )
        except Exception as e:
            logging.error(f"Error updating access stats: {e}")
    
    def remove(self, key: str):
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
            except Exception as e:
                logging.error(f"Error removing from SQLite cache: {e}")
    
    def cleanup_expired(self):
        """Remove expired entries"""
        with self.lock:
            try:
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'DELETE FROM cache_entries WHERE timestamp + ttl < ?',
                        (current_time,)
                    )
            except Exception as e:
                logging.error(f"Error cleaning up expired entries: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size_bytes,
                        AVG(access_count) as avg_access_count,
                        cache_type,
                        COUNT(*) as count_by_type
                    FROM cache_entries 
                    GROUP BY cache_type
                ''')
                results = cursor.fetchall()
                
                stats = {
                    'total_entries': 0,
                    'total_size_bytes': 0,
                    'by_type': {}
                }
                
                for row in results:
                    total, size_bytes, avg_access, cache_type, count = row
                    stats['total_entries'] += total
                    stats['total_size_bytes'] += size_bytes or 0
                    stats['by_type'][cache_type] = {
                        'count': count,
                        'avg_access_count': avg_access
                    }
                
                return stats
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return {}

class MolecularDataCache:
    """
    Multi-level caching system for molecular data with intelligent cache management
    
    Cache hierarchy:
    1. Memory cache (fastest, limited size)
    2. Redis cache (fast, shared across processes)
    3. SQLite cache (persistent, unlimited size)
    """
    
    def __init__(self, 
                 memory_size: int = 1000,
                 redis_config: Optional[Dict] = None,
                 db_path: str = "trackmypdb_cache.db",
                 default_ttl: float = 3600):  # 1 hour default TTL
        
        self.default_ttl = default_ttl
        
        # Initialize memory cache
        self.memory_cache = LRUCache(memory_size)
        
        # Initialize Redis cache if available
        self.redis_cache = None
        if REDIS_AVAILABLE and redis_config:
            try:
                self.redis_cache = redis.Redis(**redis_config)
                self.redis_cache.ping()  # Test connection
            except Exception as e:
                st.warning(f"Redis connection failed: {e}")
                self.redis_cache = None
        
        # Initialize SQLite cache
        self.db_cache = SqliteCache(db_path)
        
        # Cache statistics
        self.stats = {
            'hits': {'memory': 0, 'redis': 0, 'disk': 0},
            'misses': {'memory': 0, 'redis': 0, 'disk': 0},
            'puts': {'memory': 0, 'redis': 0, 'disk': 0}
        }
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a consistent cache key"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        hash_obj = hashlib.md5(f"{prefix}_{data_str}".encode())
        return hash_obj.hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return time.time() > (entry.timestamp + entry.ttl)
    
    def get_cached_similarity(self, smiles_hash: str, target_smiles: str) -> Optional[Any]:
        """Get cached similarity results"""
        cache_key = self._generate_key("similarity", {"smiles_hash": smiles_hash, "target": target_smiles})
        return self._get_from_cache(cache_key, "similarity")
    
    def store_similarity(self, smiles_hash: str, target_smiles: str, results: Any, ttl: Optional[float] = None):
        """Store similarity results in cache"""
        cache_key = self._generate_key("similarity", {"smiles_hash": smiles_hash, "target": target_smiles})
        self._put_to_cache(cache_key, results, "similarity", ttl or self.default_ttl)
    
    def get_cached_heteroatoms(self, uniprot_id: str) -> Optional[Any]:
        """Get cached heteroatom data"""
        cache_key = self._generate_key("heteroatoms", uniprot_id)
        return self._get_from_cache(cache_key, "heteroatoms")
    
    def store_heteroatoms(self, uniprot_id: str, data: Any, ttl: Optional[float] = None):
        """Store heteroatom data in cache"""
        cache_key = self._generate_key("heteroatoms", uniprot_id)
        self._put_to_cache(cache_key, data, "heteroatoms", ttl or self.default_ttl)
    
    def get_cached_pdb_structure(self, pdb_id: str) -> Optional[Any]:
        """Get cached PDB structure"""
        cache_key = self._generate_key("pdb_structure", pdb_id)
        return self._get_from_cache(cache_key, "pdb_structure")
    
    def store_pdb_structure(self, pdb_id: str, structure_data: Any, ttl: Optional[float] = None):
        """Store PDB structure in cache"""
        cache_key = self._generate_key("pdb_structure", pdb_id)
        # PDB structures can be large, use longer TTL
        self._put_to_cache(cache_key, structure_data, "pdb_structure", ttl or (self.default_ttl * 24))
    
    def get_cached_chemical_data(self, heteroatom_code: str) -> Optional[Any]:
        """Get cached chemical data"""
        cache_key = self._generate_key("chemical_data", heteroatom_code)
        return self._get_from_cache(cache_key, "chemical_data")
    
    def store_chemical_data(self, heteroatom_code: str, data: Any, ttl: Optional[float] = None):
        """Store chemical data in cache"""
        cache_key = self._generate_key("chemical_data", heteroatom_code)
        # Chemical data is relatively stable, use longer TTL
        self._put_to_cache(cache_key, data, "chemical_data", ttl or (self.default_ttl * 7))
    
    def _get_from_cache(self, key: str, cache_type: str) -> Optional[Any]:
        """Get data from cache hierarchy"""
        # Try memory cache first
        entry = self.memory_cache.get(key)
        if entry and not self._is_expired(entry):
            self.stats['hits']['memory'] += 1
            return entry.data
        else:
            self.stats['misses']['memory'] += 1
        
        # Try Redis cache
        if self.redis_cache:
            try:
                cached_data = self.redis_cache.get(key)
                if cached_data:
                    entry = pickle.loads(cached_data)
                    if not self._is_expired(entry):
                        self.stats['hits']['redis'] += 1
                        # Promote to memory cache
                        self.memory_cache.put(key, entry)
                        return entry.data
                    else:
                        # Remove expired entry
                        self.redis_cache.delete(key)
                self.stats['misses']['redis'] += 1
            except Exception as e:
                logging.error(f"Redis cache error: {e}")
        
        # Try SQLite cache
        entry = self.db_cache.get(key)
        if entry and not self._is_expired(entry):
            self.stats['hits']['disk'] += 1
            # Promote to higher level caches
            self.memory_cache.put(key, entry)
            if self.redis_cache:
                try:
                    self.redis_cache.setex(key, int(entry.ttl), pickle.dumps(entry))
                except Exception as e:
                    logging.error(f"Redis promotion error: {e}")
            return entry.data
        else:
            self.stats['misses']['disk'] += 1
        
        return None
    
    def _put_to_cache(self, key: str, data: Any, cache_type: str, ttl: float):
        """Store data in all cache levels"""
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=ttl,
            access_count=1
        )
        
        # Store in memory cache
        self.memory_cache.put(key, entry)
        self.stats['puts']['memory'] += 1
        
        # Store in Redis cache
        if self.redis_cache:
            try:
                self.redis_cache.setex(key, int(ttl), pickle.dumps(entry))
                self.stats['puts']['redis'] += 1
            except Exception as e:
                logging.error(f"Redis store error: {e}")
        
        # Store in SQLite cache
        self.db_cache.put(key, entry, cache_type)
        self.stats['puts']['disk'] += 1
    
    def invalidate_cache(self, pattern: Optional[str] = None, cache_type: Optional[str] = None):
        """Invalidate cache entries based on pattern or type"""
        if pattern:
            # Clear memory cache entries matching pattern
            keys_to_remove = [k for k in self.memory_cache.cache.keys() if pattern in k]
            for key in keys_to_remove:
                self.memory_cache.remove(key)
            
            # Clear Redis cache entries matching pattern
            if self.redis_cache:
                try:
                    keys = self.redis_cache.keys(f"*{pattern}*")
                    if keys:
                        self.redis_cache.delete(*keys)
                except Exception as e:
                    logging.error(f"Redis invalidation error: {e}")
        
        # Clear SQLite cache by type
        if cache_type:
            try:
                with sqlite3.connect(self.db_cache.db_path) as conn:
                    conn.execute('DELETE FROM cache_entries WHERE cache_type = ?', (cache_type,))
            except Exception as e:
                logging.error(f"SQLite invalidation error: {e}")
    
    def cleanup_expired_entries(self):
        """Clean up expired entries from all cache levels"""
        current_time = time.time()
        
        # Clean memory cache
        expired_keys = []
        for key, entry in self.memory_cache.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        for key in expired_keys:
            self.memory_cache.remove(key)
        
        # Clean Redis cache (Redis handles TTL automatically, but we can clean manually)
        if self.redis_cache:
            try:
                # Redis automatically expires keys, but we can scan for any remaining expired ones
                pass
            except Exception as e:
                logging.error(f"Redis cleanup error: {e}")
        
        # Clean SQLite cache
        self.db_cache.cleanup_expired()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'performance': self.stats,
            'memory_cache_size': self.memory_cache.size(),
            'sqlite_stats': self.db_cache.get_cache_stats()
        }
        
        # Add Redis stats if available
        if self.redis_cache:
            try:
                redis_info = self.redis_cache.info()
                stats['redis_stats'] = {
                    'used_memory': redis_info.get('used_memory_human'),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'connected_clients': redis_info.get('connected_clients', 0)
                }
            except Exception as e:
                stats['redis_stats'] = {'error': str(e)}
        
        # Calculate hit rates
        for level in ['memory', 'redis', 'disk']:
            hits = self.stats['hits'][level]
            misses = self.stats['misses'][level]
            total = hits + misses
            if total > 0:
                stats['performance'][f'{level}_hit_rate'] = hits / total
            else:
                stats['performance'][f'{level}_hit_rate'] = 0
        
        return stats
    
    def clear_all_caches(self):
        """Clear all cache levels"""
        self.memory_cache.clear()
        
        if self.redis_cache:
            try:
                self.redis_cache.flushdb()
            except Exception as e:
                logging.error(f"Redis clear error: {e}")
        
        try:
            with sqlite3.connect(self.db_cache.db_path) as conn:
                conn.execute('DELETE FROM cache_entries')
        except Exception as e:
            logging.error(f"SQLite clear error: {e}")


# Convenience functions for easy integration

def create_default_cache() -> MolecularDataCache:
    """Create a default cache configuration"""
    redis_config = None
    if REDIS_AVAILABLE:
        redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'decode_responses': False
        }
    
    return MolecularDataCache(
        memory_size=1000,
        redis_config=redis_config,
        db_path="trackmypdb_cache.db",
        default_ttl=3600
    )