"""
TrackMyPDB Async API Handler
@author Anu Gamage

High-performance async API handler with connection pooling, retry logic,
rate limiting, and comprehensive error handling for molecular data APIs.
Licensed under MIT License - Open Source Project
"""

import asyncio
import aiohttp
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import streamlit as st
import logging
from collections import defaultdict, deque

class APIEndpoint(Enum):
    PDBE = "pdbe"
    RCSB = "rcsb"
    PUBCHEM = "pubchem"

@dataclass
class APIResponse:
    data: Any
    status_code: int
    headers: Dict[str, str]
    response_time: float
    endpoint: APIEndpoint
    url: str

class RateLimiter:
    """Token bucket rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = self.requests[0]
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            self.requests.append(now)
            return True

class AsyncAPIHandler:
    """
    High-performance async API handler for TrackMyPDB with advanced features:
    - Connection pooling and reuse
    - Automatic retry with exponential backoff
    - Rate limiting per API endpoint
    - Comprehensive error handling
    - Performance monitoring
    - Request/response caching
    """
    
    def __init__(self, 
                 max_concurrent: int = 15,
                 retry_attempts: int = 3,
                 timeout: int = 30,
                 rate_limits: Optional[Dict[APIEndpoint, int]] = None):
        
        # Connection management
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.retry_attempts = retry_attempts
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
        # Rate limiting setup
        default_limits = {
            APIEndpoint.PDBE: 100,    # 100 requests per minute
            APIEndpoint.RCSB: 50,     # 50 requests per minute
            APIEndpoint.PUBCHEM: 200  # 200 requests per minute
        }
        self.rate_limits = rate_limits or default_limits
        self.rate_limiters = {
            endpoint: RateLimiter(limit, 60) 
            for endpoint, limit in self.rate_limits.items()
        }
        
        # Performance monitoring
        self.performance_metrics = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self._connector = None
        
        # API endpoints configuration
        self.endpoints = {
            APIEndpoint.PDBE: "https://www.ebi.ac.uk/pdbe/api",
            APIEndpoint.RCSB: "https://data.rcsb.org/rest/v1",
            APIEndpoint.PUBCHEM: "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        }
        
        # Request cache
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def initialize(self):
        """Initialize the HTTP session with optimized settings"""
        if self.session is None:
            # Create optimized connector
            self._connector = aiohttp.TCPConnector(
                limit=self.max_concurrent * 2,  # Total connection pool size
                limit_per_host=self.max_concurrent,  # Max connections per host
                keepalive_timeout=60,  # Keep connections alive for 60s
                enable_cleanup_closed=True,
                use_dns_cache=True,
                ttl_dns_cache=300,  # DNS cache for 5 minutes
            )
            
            # Create session with optimized settings
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'TrackMyPDB/1.0 (Protein Analysis Tool)',
                    'Accept': 'application/json',
                    'Connection': 'keep-alive'
                },
                raise_for_status=False  # Handle status codes manually
            )

    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        if self._connector:
            await self._connector.close()
            self._connector = None

    def _generate_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key for the request"""
        cache_data = f"{url}_{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _is_cache_valid(self, cached_time: float) -> bool:
        """Check if cached data is still valid"""
        return time.time() - cached_time < self.cache_ttl

    async def fetch_with_retry(self, 
                              url: str, 
                              endpoint: APIEndpoint,
                              params: Optional[Dict] = None,
                              headers: Optional[Dict] = None,
                              method: str = 'GET',
                              data: Optional[Dict] = None) -> APIResponse:
        """
        Fetch data with automatic retry, rate limiting, and caching
        
        Args:
            url: The URL to fetch
            endpoint: API endpoint type for rate limiting
            params: Query parameters
            headers: Additional headers
            method: HTTP method
            data: Request body data
            
        Returns:
            APIResponse object with data and metadata
        """
        await self.initialize()
        
        # Check cache first
        cache_key = self._generate_cache_key(url, params)
        if cache_key in self.cache:
            cached_response, cached_time = self.cache[cache_key]
            if self._is_cache_valid(cached_time):
                return cached_response

        # Rate limiting
        await self.rate_limiters[endpoint].acquire()
        
        async with self.semaphore:
            for attempt in range(self.retry_attempts + 1):
                try:
                    start_time = time.time()
                    
                    # Make the request
                    async with self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        headers=headers,
                        json=data if method.upper() in ['POST', 'PUT'] else None
                    ) as response:
                        
                        response_time = time.time() - start_time
                        response_data = await response.text()
                        
                        # Try to parse as JSON
                        try:
                            response_data = json.loads(response_data)
                        except json.JSONDecodeError:
                            pass  # Keep as text if not valid JSON
                        
                        api_response = APIResponse(
                            data=response_data,
                            status_code=response.status,
                            headers=dict(response.headers),
                            response_time=response_time,
                            endpoint=endpoint,
                            url=url
                        )
                        
                        # Record metrics
                        self.performance_metrics[endpoint].append(response_time)
                        
                        if response.status < 400:
                            self.success_counts[endpoint] += 1
                            # Cache successful responses
                            self.cache[cache_key] = (api_response, time.time())
                            return api_response
                        else:
                            self.error_counts[endpoint] += 1
                            if response.status < 500 or attempt == self.retry_attempts:
                                # Don't retry client errors (4xx) or final attempt
                                return api_response
                            
                except asyncio.TimeoutError:
                    self.error_counts[endpoint] += 1
                    if attempt == self.retry_attempts:
                        raise
                except Exception as e:
                    self.error_counts[endpoint] += 1
                    if attempt == self.retry_attempts:
                        raise
                
                # Exponential backoff
                if attempt < self.retry_attempts:
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)
            
            raise Exception(f"Max retries exceeded for {url}")

    # Specialized methods for each API

    async def fetch_pdbe_mappings(self, uniprot_id: str) -> APIResponse:
        """Fetch PDB mappings from PDBe API"""
        url = f"{self.endpoints[APIEndpoint.PDBE]}/mappings/best_structures/{uniprot_id}"
        return await self.fetch_with_retry(url, APIEndpoint.PDBE)

    async def fetch_pdb_structure(self, pdb_id: str) -> APIResponse:
        """Fetch PDB structure file from RCSB"""
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        return await self.fetch_with_retry(url, APIEndpoint.RCSB)

    async def fetch_rcsb_chemical_data(self, heteroatom_code: str) -> APIResponse:
        """Fetch chemical data from RCSB API"""
        url = f"{self.endpoints[APIEndpoint.RCSB]}/core/chemcomp/{heteroatom_code}"
        return await self.fetch_with_retry(url, APIEndpoint.RCSB)

    async def fetch_pubchem_smiles(self, compound_name: str) -> APIResponse:
        """Fetch SMILES from PubChem API"""
        url = f"{self.endpoints[APIEndpoint.PUBCHEM]}/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
        return await self.fetch_with_retry(url, APIEndpoint.PUBCHEM)

    async def batch_fetch(self, requests: List[Dict[str, Any]]) -> List[APIResponse]:
        """
        Execute multiple requests concurrently
        
        Args:
            requests: List of request dictionaries with keys: url, endpoint, params, etc.
            
        Returns:
            List of APIResponse objects
        """
        tasks = []
        for req in requests:
            task = self.fetch_with_retry(
                url=req['url'],
                endpoint=req['endpoint'],
                params=req.get('params'),
                headers=req.get('headers'),
                method=req.get('method', 'GET'),
                data=req.get('data')
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        stats = {}
        for endpoint in APIEndpoint:
            metrics = self.performance_metrics[endpoint]
            if metrics:
                stats[endpoint.value] = {
                    'total_requests': len(metrics),
                    'avg_response_time': sum(metrics) / len(metrics),
                    'min_response_time': min(metrics),
                    'max_response_time': max(metrics),
                    'success_count': self.success_counts[endpoint],
                    'error_count': self.error_counts[endpoint],
                    'success_rate': self.success_counts[endpoint] / (self.success_counts[endpoint] + self.error_counts[endpoint]) if (self.success_counts[endpoint] + self.error_counts[endpoint]) > 0 else 0
                }
        return stats

    def clear_cache(self):
        """Clear the request cache"""
        self.cache.clear()

    def clear_metrics(self):
        """Clear performance metrics"""
        self.performance_metrics.clear()
        self.error_counts.clear()
        self.success_counts.clear()


# Convenience functions for easy integration

async def fetch_multiple_structures(uniprot_ids: List[str]) -> List[APIResponse]:
    """Fetch PDB structures for multiple UniProt IDs"""
    async with AsyncAPIHandler(max_concurrent=15) as handler:
        # First get PDB mappings
        mapping_tasks = [handler.fetch_pdbe_mappings(uid) for uid in uniprot_ids]
        mapping_responses = await asyncio.gather(*mapping_tasks)
        
        # Extract PDB IDs and fetch structures
        pdb_ids = []
        for response in mapping_responses:
            if response.status_code == 200 and isinstance(response.data, dict):
                for uniprot_id, data in response.data.items():
                    if isinstance(data, dict) and 'best_structures' in data:
                        for struct in data['best_structures']:
                            if 'pdb_id' in struct:
                                pdb_ids.append(struct['pdb_id'].upper())
        
        # Fetch PDB files
        structure_tasks = [handler.fetch_pdb_structure(pdb_id) for pdb_id in set(pdb_ids)]
        return await asyncio.gather(*structure_tasks)

async def fetch_chemical_data_batch(heteroatom_codes: List[str]) -> List[APIResponse]:
    """Fetch chemical data for multiple heteroatom codes"""
    async with AsyncAPIHandler(max_concurrent=10) as handler:
        tasks = [handler.fetch_rcsb_chemical_data(code) for code in heteroatom_codes]
        return await asyncio.gather(*tasks)