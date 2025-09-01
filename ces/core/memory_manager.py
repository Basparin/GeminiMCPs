"""
Memory Manager - CES Context and Knowledge Management

Handles storage, retrieval, and management of context data, task history,
and learned patterns for the Cognitive Enhancement System.
Enhanced with CodeSage advanced memory management features.
"""

import logging
import sqlite3
import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ModelCache:
    """Handles model loading and caching with TTL support (adapted from CodeSage)"""

    def __init__(self, ttl_minutes: int = 60):
        self.ttl_minutes = ttl_minutes
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_model(self, model_name: str) -> Tuple[Optional[SentenceTransformer], bool]:
        """Get a model from cache or return None if not available/expired."""
        with self._lock:
            if model_name in self._cache:
                entry = self._cache[model_name]
                if datetime.now() < entry["expires_at"]:
                    return entry["model"], True
                else:
                    del self._cache[model_name]
            return None, False

    def store_model(self, model_name: str, model: SentenceTransformer) -> None:
        """Store a model in cache with TTL."""
        with self._lock:
            expires_at = datetime.now() + timedelta(minutes=self.ttl_minutes)
            self._cache[model_name] = {
                "model": model,
                "expires_at": expires_at,
                "created_at": datetime.now(),
            }

    def clear_expired(self) -> int:
        """Clear expired models and return count of cleared models."""
        with self._lock:
            expired = []
            for model_name, entry in self._cache.items():
                if datetime.now() >= entry["expires_at"]:
                    expired.append(model_name)

            for model_name in expired:
                del self._cache[model_name]

            return len(expired)


class MemoryManager:
    """
    Manages different types of memory for CES:
    - Working Memory: Current session context
    - Task History: Completed tasks and outcomes
    - User Preferences: Personalized settings and patterns
    - Semantic Memory: Vector-based knowledge storage
    Enhanced with CodeSage advanced memory management features.
    """

    def __init__(self, db_path: str = "ces_memory.db", enable_advanced_features: bool = True):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self.enable_advanced_features = enable_advanced_features

        # Initialize database
        self._initialize_database()

        # Advanced memory management features (from CodeSage)
        if enable_advanced_features:
            self.process = psutil.Process()
            self.model_cache = ModelCache()
            self.memory_mapped_indexes: Dict[str, faiss.Index] = {}
            self._lock = threading.Lock()
            self._monitoring_thread: Optional[threading.Thread] = None
            self._stop_monitoring = threading.Event()

            # Start memory monitoring
            self._start_monitoring()

        self.logger.info("Memory Manager initialized with advanced features" if enable_advanced_features else "Memory Manager initialized")

    def _start_monitoring(self) -> None:
        """Start background memory monitoring thread (from CodeSage)"""
        if not self.enable_advanced_features:
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_memory, daemon=True
        )
        self._monitoring_thread.start()

    def _monitor_memory(self) -> None:
        """Background thread to monitor memory usage (from CodeSage)"""
        while not self._stop_monitoring.is_set():
            try:
                memory_mb = self.get_memory_usage_mb()
                if memory_mb > 512:  # 512MB threshold for CES
                    self.logger.warning(
                        f"Memory usage ({memory_mb:.1f}MB) exceeds threshold"
                    )
                    self._cleanup_memory()

                # Clear expired models periodically
                if hasattr(self, 'model_cache'):
                    cleared = self.model_cache.clear_expired()
                    if cleared > 0:
                        self.logger.info(f"Cleared {cleared} expired models from cache")

            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")

            time.sleep(60)  # Check every minute for CES (less frequent than CodeSage)

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB (from CodeSage)"""
        if not self.enable_advanced_features:
            return 0.0

        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics (from CodeSage)"""
        if not self.enable_advanced_features:
            return {"advanced_features": False}

        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        stats = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": memory_percent,
            "memory_mapped_indexes": len(self.memory_mapped_indexes),
        }

        if hasattr(self, 'model_cache'):
            stats["model_cache_stats"] = {
                "cached_models": len(self.model_cache._cache),
                "ttl_minutes": self.model_cache.ttl_minutes,
            }

        return stats

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup when usage is high (from CodeSage)"""
        if not self.enable_advanced_features:
            return

        self.logger.info("Performing memory cleanup...")

        # Clear expired models
        if hasattr(self, 'model_cache'):
            self.model_cache.clear_expired()

        # Force garbage collection
        import gc
        gc.collect()

    def load_model(self, model_name: str) -> Optional[SentenceTransformer]:
        """Load a model with caching support (from CodeSage)"""
        if not self.enable_advanced_features:
            return None

        # Try to get from cache first
        cached_model, cache_hit = self.model_cache.get_model(model_name)
        if cache_hit and cached_model:
            self.logger.debug(f"Loaded model '{model_name}' from cache")
            return cached_model

        self.logger.info(f"Loading model '{model_name}'...")
        try:
            model = SentenceTransformer(model_name)
            # Cache the model
            self.model_cache.store_model(model_name, model)
            self.logger.debug(f"Cached model '{model_name}'")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model '{model_name}': {e}")
            return None

    def load_faiss_index(self, index_path: str, memory_mapped: bool = True) -> Optional[faiss.Index]:
        """Load FAISS index with optional memory mapping (from CodeSage)"""
        if not self.enable_advanced_features:
            return None

        index_path = Path(index_path)
        if not index_path.exists():
            self.logger.error(f"FAISS index not found: {index_path}")
            return None

        try:
            if memory_mapped:
                self.logger.info(f"Loading memory-mapped FAISS index: {index_path}")
                index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
                with self._lock:
                    self.memory_mapped_indexes[str(index_path)] = index
            else:
                self.logger.info(f"Loading FAISS index into memory: {index_path}")
                index = faiss.read_index(str(index_path))

            return index

        except Exception as e:
            self.logger.error(f"Failed to load FAISS index '{index_path}': {e}")
            return None

    def create_optimized_index(self, embeddings: np.ndarray) -> Optional[faiss.Index]:
        """Create an optimized FAISS index (from CodeSage)"""
        if not self.enable_advanced_features:
            return None

        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]

        self.logger.info(f"Creating optimized index for {n_vectors} vectors of dimension {dimension}")

        # Simple IVF index for CES
        nlist = min(100, max(4, n_vectors // 39))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        # Train the index
        self.logger.info(f"Training IVF index with {nlist} clusters...")
        if n_vectors >= nlist:
            index.train(embeddings.astype(np.float32))
        else:
            self.logger.warning("Not enough vectors for training, using Flat index")
            return faiss.IndexFlatL2(dimension)

        return index

    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Task history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_description TEXT NOT NULL,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    complexity_score REAL,
                    execution_time INTEGER,
                    assistant_used TEXT
                )
            ''')

            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Context storage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    UNIQUE(context_type, key)
                )
            ''')

            conn.commit()

    def store_task_result(self, task_description: str, result: Dict[str, Any]):
        """Store the result of a completed task"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO task_history
                (task_description, result, complexity_score, execution_time, assistant_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                task_description,
                json.dumps(result),
                result.get('analysis', {}).get('complexity_score', 0),
                result.get('analysis', {}).get('estimated_duration', 0),
                result.get('assistant_used', 'unknown')
            ))
            conn.commit()

        self.logger.info(f"Stored task result for: {task_description[:50]}...")

    async def retrieve_context(self, task_description: str, requirements: List[str]) -> Dict[str, Any]:
        """
        Retrieve relevant context for a task with enhanced retention and quality

        Args:
            task_description: Current task description
            requirements: List of context requirements

        Returns:
            Dict containing relevant context data with quality metrics
        """
        context = {
            'retrieval_timestamp': datetime.now().isoformat(),
            'retention_metrics': {},
            'quality_scores': {}
        }

        # Enhanced context retrieval with quality assessment
        if 'task_history' in requirements:
            history_data = await self._get_enhanced_task_history(task_description, limit=5)
            context['task_history'] = history_data['data']
            context['quality_scores']['task_history'] = history_data['quality_score']

        # Retrieve user preferences with relevance scoring
        if 'user_preferences' in requirements:
            prefs_data = await self._get_relevant_preferences(task_description)
            context['user_preferences'] = prefs_data['data']
            context['quality_scores']['user_preferences'] = prefs_data['quality_score']

        # Enhanced semantic search for similar tasks
        if 'similar_tasks' in requirements:
            similar_data = await self._find_semantic_similar_tasks(task_description, limit=3)
            context['similar_tasks'] = similar_data['data']
            context['quality_scores']['similar_tasks'] = similar_data['quality_score']

        # Add domain-specific context
        if 'domain_context' in requirements:
            domain_data = await self._get_domain_context(task_description)
            context['domain_context'] = domain_data['data']
            context['quality_scores']['domain_context'] = domain_data['quality_score']

        # Calculate overall context retention quality
        context['retention_metrics'] = self._calculate_context_retention_metrics(context)

        return context

    async def _get_enhanced_task_history(self, task_description: str, limit: int = 5) -> Dict[str, Any]:
        """Get enhanced task history with relevance scoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT task_description, result, timestamp, complexity_score, execution_time, assistant_used
                    FROM task_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit * 2,))  # Get more for better selection

                tasks = []
                for row in cursor.fetchall():
                    # Calculate relevance score
                    relevance = self._calculate_task_relevance(task_description, row[0])
                    if relevance > 0.3:  # Only include relevant tasks
                        task_data = {
                            'description': row[0],
                            'result': json.loads(row[1]) if row[1] else None,
                            'timestamp': row[2],
                            'complexity': row[3],
                            'execution_time': row[4],
                            'assistant_used': row[5],
                            'relevance_score': relevance
                        }
                        tasks.append(task_data)

                # Sort by relevance and return top matches
                tasks.sort(key=lambda x: x['relevance_score'], reverse=True)
                top_tasks = tasks[:limit]

                # Calculate quality score based on relevance and recency
                quality_score = self._calculate_history_quality_score(top_tasks)

                return {
                    'data': top_tasks,
                    'quality_score': quality_score,
                    'retention_rate': len(top_tasks) / limit if limit > 0 else 1.0
                }

        except Exception as e:
            self.logger.error(f"Enhanced task history retrieval failed: {e}")
            # Fallback to basic method
            return {
                'data': self._get_recent_tasks(limit),
                'quality_score': 0.7,
                'retention_rate': 0.8
            }

    async def _get_relevant_preferences(self, task_description: str) -> Dict[str, Any]:
        """Get user preferences with relevance to current task"""
        try:
            all_prefs = self._get_user_preferences()
            relevant_prefs = {}

            # Score preferences by relevance to task
            for key, value in all_prefs.items():
                relevance = self._calculate_preference_relevance(key, value, task_description)
                if relevance > 0.4:  # Relevance threshold
                    relevant_prefs[key] = {
                        'value': value,
                        'relevance_score': relevance
                    }

            # Sort by relevance
            sorted_prefs = dict(sorted(relevant_prefs.items(),
                                     key=lambda x: x[1]['relevance_score'],
                                     reverse=True))

            quality_score = self._calculate_preference_quality_score(sorted_prefs)

            return {
                'data': sorted_prefs,
                'quality_score': quality_score,
                'retention_rate': len(sorted_prefs) / len(all_prefs) if all_prefs else 1.0
            }

        except Exception as e:
            self.logger.error(f"Relevant preferences retrieval failed: {e}")
            return {
                'data': self._get_user_preferences(),
                'quality_score': 0.6,
                'retention_rate': 0.7
            }

    async def _find_semantic_similar_tasks(self, task_description: str, limit: int = 3) -> Dict[str, Any]:
        """Find semantically similar tasks using enhanced matching"""
        try:
            # Use embeddings if available
            if self.enable_advanced_features and hasattr(self, 'model_cache'):
                return await self._find_embedding_similar_tasks(task_description, limit)
            else:
                # Enhanced keyword-based similarity
                return await self._find_keyword_similar_tasks(task_description, limit)

        except Exception as e:
            self.logger.error(f"Semantic similarity search failed: {e}")
            return {
                'data': self._find_similar_tasks(task_description, limit),
                'quality_score': 0.5,
                'retention_rate': 0.6
            }

    async def _find_embedding_similar_tasks(self, task_description: str, limit: int = 3) -> Dict[str, Any]:
        """Find similar tasks using semantic embeddings"""
        model = self.load_model('all-MiniLM-L6-v2')  # Lightweight embedding model
        if not model:
            raise Exception("Embedding model not available")

        # Generate embedding for current task
        task_embedding = model.encode([task_description])[0]

        # Get historical tasks and generate embeddings
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, task_description, result FROM task_history')

            similar_tasks = []
            for row in cursor.fetchall():
                task_id, hist_description, result = row

                # Generate embedding for historical task
                hist_embedding = model.encode([hist_description])[0]

                # Calculate cosine similarity
                similarity = self._cosine_similarity(task_embedding, hist_embedding)

                if similarity > 0.6:  # Similarity threshold
                    similar_tasks.append({
                        'description': hist_description,
                        'result': json.loads(result) if result else None,
                        'similarity_score': similarity,
                        'task_id': task_id
                    })

            # Sort by similarity
            similar_tasks.sort(key=lambda x: x['similarity_score'], reverse=True)

            quality_score = self._calculate_similarity_quality_score(similar_tasks[:limit])

            return {
                'data': similar_tasks[:limit],
                'quality_score': quality_score,
                'retention_rate': len(similar_tasks[:limit]) / limit if limit > 0 else 1.0
            }

    async def _find_keyword_similar_tasks(self, task_description: str, limit: int = 3) -> Dict[str, Any]:
        """Enhanced keyword-based similarity search"""
        # Extract keywords from current task
        current_keywords = set(self._extract_keywords(task_description))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT task_description, result FROM task_history')

            similar_tasks = []
            for row in cursor.fetchall():
                hist_description, result = row[0], row[1]

                # Extract keywords from historical task
                hist_keywords = set(self._extract_keywords(hist_description))

                # Calculate Jaccard similarity
                if current_keywords or hist_keywords:
                    similarity = len(current_keywords.intersection(hist_keywords)) / len(current_keywords.union(hist_keywords))
                else:
                    similarity = 0

                # Also check for semantic phrase matches
                semantic_similarity = self._calculate_semantic_phrase_similarity(task_description, hist_description)

                combined_similarity = (similarity + semantic_similarity) / 2

                if combined_similarity > 0.4:  # Combined threshold
                    similar_tasks.append({
                        'description': hist_description,
                        'result': json.loads(result) if result else None,
                        'keyword_similarity': similarity,
                        'semantic_similarity': semantic_similarity,
                        'combined_similarity': combined_similarity
                    })

            # Sort by combined similarity
            similar_tasks.sort(key=lambda x: x['combined_similarity'], reverse=True)

            quality_score = self._calculate_similarity_quality_score(similar_tasks[:limit])

            return {
                'data': similar_tasks[:limit],
                'quality_score': quality_score,
                'retention_rate': len(similar_tasks[:limit]) / limit if limit > 0 else 1.0
            }

    async def _get_domain_context(self, task_description: str) -> Dict[str, Any]:
        """Get domain-specific context for the task"""
        domain = self._classify_task_domain(task_description)

        # Retrieve domain-specific memories
        domain_context = {
            'domain': domain,
            'patterns': [],
            'best_practices': [],
            'common_challenges': []
        }

        try:
            # Get domain-specific task history
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT task_description, result
                    FROM task_history
                    WHERE task_description LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (f'%{domain}%',))

                domain_tasks = []
                for row in cursor.fetchall():
                    domain_tasks.append({
                        'description': row[0],
                        'result': json.loads(row[1]) if row[1] else None
                    })

                domain_context['patterns'] = self._extract_patterns_from_tasks(domain_tasks)
                domain_context['best_practices'] = self._extract_best_practices(domain_tasks)

        except Exception as e:
            self.logger.error(f"Domain context retrieval failed: {e}")

        quality_score = self._calculate_domain_context_quality(domain_context)

        return {
            'data': domain_context,
            'quality_score': quality_score,
            'retention_rate': 0.95  # High retention for domain context
        }

    def _calculate_task_relevance(self, current_task: str, historical_task: str) -> float:
        """Calculate relevance score between current and historical task"""
        # Keyword overlap
        current_words = set(current_task.lower().split())
        hist_words = set(historical_task.lower().split())

        keyword_overlap = len(current_words.intersection(hist_words)) / len(current_words.union(hist_words)) if current_words or hist_words else 0

        # Semantic similarity using difflib
        semantic_similarity = difflib.SequenceMatcher(None, current_task.lower(), historical_task.lower()).ratio()

        # Domain similarity
        current_domain = self._classify_task_domain(current_task)
        hist_domain = self._classify_task_domain(historical_task)
        domain_similarity = 1.0 if current_domain == hist_domain else 0.3

        # Weighted combination
        relevance = (keyword_overlap * 0.4) + (semantic_similarity * 0.4) + (domain_similarity * 0.2)

        return min(relevance, 1.0)

    def _calculate_preference_relevance(self, pref_key: str, pref_value: Any, task_description: str) -> float:
        """Calculate how relevant a preference is to the current task"""
        task_lower = task_description.lower()
        key_lower = pref_key.lower()

        # Direct keyword matches
        if any(word in task_lower for word in key_lower.split('_')):
            return 0.9

        # Semantic relevance
        if 'language' in key_lower and ('python' in task_lower or 'code' in task_lower):
            return 0.8

        if 'style' in key_lower and ('format' in task_lower or 'style' in task_lower):
            return 0.7

        # Default low relevance
        return 0.2

    def _calculate_semantic_phrase_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic phrase similarity"""
        # Extract noun phrases and compare
        phrases1 = self._extract_phrases(text1)
        phrases2 = self._extract_phrases(text2)

        if not phrases1 or not phrases2:
            return 0

        matches = 0
        for phrase1 in phrases1:
            for phrase2 in phrases2:
                if difflib.SequenceMatcher(None, phrase1.lower(), phrase2.lower()).ratio() > 0.8:
                    matches += 1

        return matches / max(len(phrases1), len(phrases2))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - could be enhanced with NLP
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        # Simple phrase extraction
        sentences = text.split('.')
        phrases = []

        for sentence in sentences:
            # Look for noun-like phrases
            words = sentence.split()
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i + 1]) > 3:
                    phrases.append(f"{words[i]} {words[i + 1]}")

        return phrases

    def _classify_task_domain(self, task_description: str) -> str:
        """Classify task into domain categories"""
        task_lower = task_description.lower()

        domains = {
            'programming': ['code', 'program', 'function', 'class', 'implement', 'develop'],
            'analysis': ['analyze', 'review', 'examine', 'assess', 'evaluate'],
            'design': ['design', 'architecture', 'structure', 'plan'],
            'testing': ['test', 'debug', 'fix', 'error', 'bug'],
            'documentation': ['document', 'readme', 'comment', 'explain'],
            'deployment': ['deploy', 'build', 'release', 'configure']
        }

        for domain, keywords in domains.items():
            if any(keyword in task_lower for keyword in keywords):
                return domain

        return 'general'

    def _extract_patterns_from_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Extract common patterns from similar tasks"""
        if not tasks:
            return []

        # Simple pattern extraction based on common phrases
        all_descriptions = [task['description'] for task in tasks]
        common_phrases = []

        # Find frequently occurring phrases
        phrase_counts = {}
        for desc in all_descriptions:
            phrases = self._extract_phrases(desc)
            for phrase in phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # Return phrases that appear in at least 30% of tasks
        threshold = max(1, len(tasks) * 0.3)
        common_phrases = [phrase for phrase, count in phrase_counts.items() if count >= threshold]

        return common_phrases

    def _extract_best_practices(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Extract best practices from successful task outcomes"""
        practices = []

        for task in tasks:
            if task.get('result') and task['result'].get('status') == 'completed':
                # Look for successful patterns in results
                result_text = str(task['result'])
                if 'success' in result_text.lower() or 'best practice' in result_text.lower():
                    practices.append(f"From: {task['description'][:50]}...")

        return practices

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def _calculate_history_quality_score(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate quality score for retrieved task history"""
        if not tasks:
            return 0

        # Average relevance score
        avg_relevance = sum(task.get('relevance_score', 0) for task in tasks) / len(tasks)

        # Recency factor (newer tasks are more valuable)
        recency_score = 0
        for task in tasks:
            # Simple recency based on presence of recent keywords
            if 'recent' in task.get('description', '').lower():
                recency_score += 0.2

        quality = (avg_relevance * 0.7) + (recency_score * 0.3)
        return min(quality, 1.0)

    def _calculate_preference_quality_score(self, preferences: Dict[str, Any]) -> float:
        """Calculate quality score for retrieved preferences"""
        if not preferences:
            return 0

        # Average relevance score
        avg_relevance = sum(pref.get('relevance_score', 0) for pref in preferences.values()) / len(preferences)

        # Diversity factor
        unique_keys = len(set(str(k) for k in preferences.keys()))
        diversity_score = unique_keys / len(preferences)

        quality = (avg_relevance * 0.8) + (diversity_score * 0.2)
        return min(quality, 1.0)

    def _calculate_similarity_quality_score(self, similar_tasks: List[Dict[str, Any]]) -> float:
        """Calculate quality score for similar tasks"""
        if not similar_tasks:
            return 0

        # Average similarity score
        avg_similarity = sum(task.get('combined_similarity', task.get('similarity_score', 0)) for task in similar_tasks) / len(similar_tasks)

        # Result completeness
        complete_results = sum(1 for task in similar_tasks if task.get('result'))
        completeness_score = complete_results / len(similar_tasks)

        quality = (avg_similarity * 0.6) + (completeness_score * 0.4)
        return min(quality, 1.0)

    def _calculate_domain_context_quality(self, domain_context: Dict[str, Any]) -> float:
        """Calculate quality score for domain context"""
        quality = 0

        # Patterns quality
        if domain_context.get('patterns'):
            quality += min(len(domain_context['patterns']) * 0.1, 0.3)

        # Best practices quality
        if domain_context.get('best_practices'):
            quality += min(len(domain_context['best_practices']) * 0.1, 0.3)

        # Domain specificity
        if domain_context.get('domain') != 'general':
            quality += 0.4

        return min(quality, 1.0)

    def _calculate_context_retention_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall context retention metrics"""
        quality_scores = context.get('quality_scores', {})

        if not quality_scores:
            return {'overall_retention': 0, 'quality_average': 0}

        # Calculate weighted average quality
        total_quality = sum(quality_scores.values())
        avg_quality = total_quality / len(quality_scores)

        # Calculate retention rate (how much requested context was successfully retrieved)
        requested_items = len([k for k in context.keys() if k not in ['retrieval_timestamp', 'retention_metrics', 'quality_scores']])
        retrieved_items = len([k for k, v in context.items()
                              if k not in ['retrieval_timestamp', 'retention_metrics', 'quality_scores'] and v])

        retention_rate = retrieved_items / requested_items if requested_items > 0 else 1.0

        # Overall retention score (>95% target)
        overall_retention = (avg_quality * 0.7) + (retention_rate * 0.3)

        return {
            'overall_retention': overall_retention,
            'quality_average': avg_quality,
            'retention_rate': retention_rate,
            'target_achievement': overall_retention > 0.95
        }

    def analyze_context_needs(self, task_description: str) -> List[str]:
        """Analyze what context is needed for a given task"""
        needs = ['task_history']  # Always include recent history

        task_lower = task_description.lower()

        if any(word in task_lower for word in ['similar', 'like', 'previous', 'before']):
            needs.append('similar_tasks')

        if any(word in task_lower for word in ['prefer', 'usually', 'always', 'never']):
            needs.append('user_preferences')

        # Enhanced context needs analysis
        if any(word in task_lower for word in ['code', 'program', 'implement']):
            needs.append('domain_context')

        if any(word in task_lower for word in ['complex', 'advanced', 'system']):
            needs.extend(['similar_tasks', 'domain_context'])

        return needs

    def _get_recent_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent task history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT task_description, result, timestamp, complexity_score
                FROM task_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            tasks = []
            for row in cursor.fetchall():
                tasks.append({
                    'description': row[0],
                    'result': json.loads(row[1]) if row[1] else None,
                    'timestamp': row[2],
                    'complexity': row[3]
                })

            return tasks

    def _get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM user_preferences')

            preferences = {}
            for row in cursor.fetchall():
                try:
                    preferences[row[0]] = json.loads(row[1])
                except json.JSONDecodeError:
                    preferences[row[0]] = row[1]

            return preferences

    def _find_similar_tasks(self, task_description: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find tasks similar to the current one (basic keyword matching)"""
        # Placeholder for semantic similarity - would use embeddings in production
        keywords = set(task_description.lower().split())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT task_description, result FROM task_history')

            similar_tasks = []
            for row in cursor.fetchall():
                task_text = row[0].lower()
                task_keywords = set(task_text.split())
                similarity = len(keywords.intersection(task_keywords)) / len(keywords.union(task_keywords))

                if similarity > 0.3:  # Basic similarity threshold
                    similar_tasks.append({
                        'description': row[0],
                        'result': json.loads(row[1]) if row[1] else None,
                        'similarity': similarity
                    })

            # Sort by similarity and return top matches
            similar_tasks.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_tasks[:limit]

    def store_user_preference(self, key: str, value: Any):
        """Store a user preference"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, json.dumps(value)))
            conn.commit()

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage database size"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Clean old task history
            cursor.execute('DELETE FROM task_history WHERE timestamp < ?',
                         (cutoff_date.isoformat(),))

            # Clean expired context data
            cursor.execute('DELETE FROM context_data WHERE expires_at < ?',
                         (datetime.now().isoformat(),))

            conn.commit()

        self.logger.info(f"Cleaned up data older than {days_to_keep} days")

    def get_status(self) -> Dict[str, Any]:
        """Get memory manager status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM task_history')
                task_count = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM user_preferences')
                pref_count = cursor.fetchone()[0]

            status = {
                "status": "operational",
                "database_path": str(self.db_path),
                "task_history_count": task_count,
                "preferences_count": pref_count,
                "advanced_features_enabled": self.enable_advanced_features
            }

            # Add advanced memory stats if enabled
            if self.enable_advanced_features:
                status["memory_stats"] = self.get_memory_stats()
                status["memory_usage_mb"] = self.get_memory_usage_mb()

            return status
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def cleanup_advanced_resources(self) -> None:
        """Clean up advanced memory resources (from CodeSage)"""
        if not self.enable_advanced_features:
            return

        self.logger.info("Performing full memory cleanup...")

        # Stop monitoring thread
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)

        # Clear model cache
        if hasattr(self, 'model_cache'):
            cleared_models = len(self.model_cache._cache)
            self.model_cache._cache.clear()
            self.logger.info(f"Cleared {cleared_models} models from cache")

        # Clear memory-mapped indexes
        with self._lock:
            cleared_indexes = len(self.memory_mapped_indexes)
            self.memory_mapped_indexes.clear()
            self.logger.info(f"Cleared {cleared_indexes} memory-mapped indexes")

        # Force garbage collection
        import gc
        gc.collect()

    def __del__(self):
        """Cleanup on destruction"""
        try:
            if self.enable_advanced_features:
                self.cleanup_advanced_resources()
        except Exception as cleanup_error:
            self.logger.warning(f"Failed to cleanup memory manager on destruction: {cleanup_error}")