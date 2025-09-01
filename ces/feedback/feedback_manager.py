"""
CES Feedback Manager

Comprehensive feedback collection, analysis, and insights system
for continuous improvement of the Cognitive Enhancement System.
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics
import os
from pathlib import Path


@dataclass
class FeedbackEntry:
    """Represents a user feedback entry"""
    id: str
    user_id: str
    feedback_type: str  # bug, feature, improvement, general, rating
    title: str
    message: str
    rating: Optional[int] = None  # 1-5 scale for satisfaction
    category: Optional[str] = None  # ai_assistant, ui, performance, etc.
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "new"  # new, reviewed, addressed, closed
    priority: str = "medium"  # low, medium, high, critical
    created_at: datetime = None
    updated_at: datetime = None
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    resolution: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class FeedbackAnalysis:
    """Analysis results for feedback data"""
    total_feedback: int = 0
    average_rating: float = 0.0
    feedback_types: Dict[str, int] = field(default_factory=dict)
    categories: Dict[str, int] = field(default_factory=dict)
    priority_distribution: Dict[str, int] = field(default_factory=dict)
    status_distribution: Dict[str, int] = field(default_factory=dict)
    trending_topics: List[str] = field(default_factory=list)
    sentiment_analysis: Dict[str, float] = field(default_factory=dict)
    common_suggestions: List[str] = field(default_factory=list)
    urgent_issues: List[str] = field(default_factory=list)
    generated_at: datetime = None

    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()


class FeedbackManager:
    """
    Manages user feedback collection, storage, and analysis
    """

    def __init__(self, storage_path: str = "./data/feedback"):
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

        # Feedback storage
        self.feedback_entries: Dict[str, FeedbackEntry] = {}

        # Analysis cache
        self.last_analysis: Optional[FeedbackAnalysis] = None
        self.analysis_cache_duration = timedelta(hours=1)  # Cache analysis for 1 hour

        # Initialize storage
        self._initialized = False

    async def ensure_initialized(self):
        """Ensure the manager is initialized"""
        if not self._initialized:
            await self._initialize_storage()
            self._initialized = True

    async def _initialize_storage(self):
        """Initialize feedback storage"""
        os.makedirs(self.storage_path, exist_ok=True)

        # Load existing feedback
        await self._load_feedback_entries()

    async def _load_feedback_entries(self):
        """Load feedback entries from storage"""
        feedback_file = f"{self.storage_path}/feedback_entries.json"
        if os.path.exists(feedback_file):
            try:
                # Use synchronous file operations for now
                with open(feedback_file, 'r') as f:
                    data = json.loads(f.read())
                    for entry_data in data.get('entries', []):
                        # Convert datetime strings back to datetime objects
                        entry_data['created_at'] = datetime.fromisoformat(entry_data['created_at'])
                        entry_data['updated_at'] = datetime.fromisoformat(entry_data['updated_at'])
                        entry = FeedbackEntry(**entry_data)
                        self.feedback_entries[entry.id] = entry
                self.logger.info(f"Loaded {len(self.feedback_entries)} feedback entries")
            except Exception as e:
                self.logger.error(f"Error loading feedback entries: {e}")

    async def _save_feedback_entries(self):
        """Save feedback entries to storage"""
        await self.ensure_initialized()
        try:
            data = {'entries': []}
            for entry in self.feedback_entries.values():
                entry_dict = {
                    **entry.__dict__,
                    'created_at': entry.created_at.isoformat(),
                    'updated_at': entry.updated_at.isoformat()
                }
                data['entries'].append(entry_dict)

            # Use synchronous file operations for now
            with open(f"{self.storage_path}/feedback_entries.json", 'w') as f:
                f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving feedback entries: {e}")

    def submit_feedback(self, user_id: str, feedback_type: str, title: str,
                       message: str, rating: Optional[int] = None,
                       category: Optional[str] = None, tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit new user feedback

        Args:
            user_id: User identifier
            feedback_type: Type of feedback (bug, feature, improvement, general, rating)
            title: Feedback title
            message: Feedback message
            rating: Optional rating (1-5)
            category: Optional category
            tags: Optional tags
            metadata: Optional additional metadata

        Returns:
            Feedback entry ID
        """
        feedback_id = f"feedback_{int(datetime.now().timestamp())}_{user_id}"

        # Determine priority based on type and content
        priority = self._determine_priority(feedback_type, message, rating)

        entry = FeedbackEntry(
            id=feedback_id,
            user_id=user_id,
            feedback_type=feedback_type,
            title=title,
            message=message,
            rating=rating,
            category=category or self._categorize_feedback(message),
            tags=tags or [],
            metadata=metadata or {},
            priority=priority
        )

        self.feedback_entries[feedback_id] = entry

        # Save asynchronously
        asyncio.create_task(self._save_feedback_entries())

        # Invalidate analysis cache
        self.last_analysis = None

        self.logger.info(f"Feedback submitted: {feedback_id} by {user_id}")
        return feedback_id

    def _determine_priority(self, feedback_type: str, message: str,
                          rating: Optional[int]) -> str:
        """
        Determine feedback priority based on type, content, and rating

        Args:
            feedback_type: Type of feedback
            message: Feedback message
            rating: Optional rating

        Returns:
            Priority level (low, medium, high, critical)
        """
        # Critical priority
        if feedback_type == "bug" and any(word in message.lower() for word in
                                         ["crash", "error", "fail", "broken", "not working"]):
            return "critical"

        if rating and rating <= 2:
            return "high"

        # High priority
        if feedback_type == "bug":
            return "high"

        if any(word in message.lower() for word in
               ["urgent", "critical", "emergency", "security"]):
            return "high"

        # Medium priority (default)
        if feedback_type in ["feature", "improvement"]:
            return "medium"

        # Low priority
        if feedback_type == "general" or (rating and rating >= 4):
            return "low"

        return "medium"

    def _categorize_feedback(self, message: str) -> str:
        """
        Automatically categorize feedback based on content

        Args:
            message: Feedback message

        Returns:
            Category name
        """
        message_lower = message.lower()

        # AI-related feedback
        if any(word in message_lower for word in ["ai", "assistant", "bot", "response", "answer"]):
            return "ai_assistant"

        # UI/UX feedback
        if any(word in message_lower for word in ["ui", "interface", "display", "screen", "button", "layout"]):
            return "user_interface"

        # Performance feedback
        if any(word in message_lower for word in ["slow", "fast", "performance", "speed", "loading", "lag"]):
            return "performance"

        # Feature feedback
        if any(word in message_lower for word in ["feature", "functionality", "capability", "tool"]):
            return "features"

        # Bug reports
        if any(word in message_lower for word in ["bug", "error", "issue", "problem", "crash", "fail"]):
            return "bugs"

        # Documentation feedback
        if any(word in message_lower for word in ["documentation", "docs", "help", "tutorial", "guide"]):
            return "documentation"

        return "general"

    def update_feedback_status(self, feedback_id: str, status: str,
                             reviewed_by: Optional[str] = None,
                             review_notes: Optional[str] = None,
                             resolution: Optional[str] = None) -> bool:
        """
        Update feedback status and add review information

        Args:
            feedback_id: Feedback entry ID
            status: New status
            reviewed_by: Reviewer identifier
            review_notes: Review notes
            resolution: Resolution description

        Returns:
            True if updated successfully, False otherwise
        """
        if feedback_id not in self.feedback_entries:
            return False

        entry = self.feedback_entries[feedback_id]
        entry.status = status
        entry.updated_at = datetime.now()

        if reviewed_by:
            entry.reviewed_by = reviewed_by
        if review_notes:
            entry.review_notes = review_notes
        if resolution:
            entry.resolution = resolution

        # Save asynchronously
        asyncio.create_task(self._save_feedback_entries())

        self.logger.info(f"Feedback {feedback_id} status updated to {status}")
        return True

    def get_feedback_entries(self, status: Optional[str] = None,
                           feedback_type: Optional[str] = None,
                           category: Optional[str] = None,
                           priority: Optional[str] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get feedback entries with optional filtering

        Args:
            status: Filter by status
            feedback_type: Filter by feedback type
            category: Filter by category
            priority: Filter by priority
            limit: Maximum number of entries to return

        Returns:
            List of feedback entries
        """
        entries = list(self.feedback_entries.values())

        # Apply filters
        if status:
            entries = [e for e in entries if e.status == status]
        if feedback_type:
            entries = [e for e in entries if e.feedback_type == feedback_type]
        if category:
            entries = [e for e in entries if e.category == category]
        if priority:
            entries = [e for e in entries if e.priority == priority]

        # Sort by creation date (newest first)
        entries.sort(key=lambda x: x.created_at, reverse=True)

        # Convert to dict and limit
        result = []
        for entry in entries[:limit]:
            entry_dict = {
                **entry.__dict__,
                'created_at': entry.created_at.isoformat(),
                'updated_at': entry.updated_at.isoformat()
            }
            result.append(entry_dict)

        return result

    def analyze_feedback(self, days: int = 30) -> FeedbackAnalysis:
        """
        Analyze feedback data and generate insights

        Args:
            days: Number of days to analyze

        Returns:
            Feedback analysis results
        """
        # Check cache
        if (self.last_analysis and
            datetime.now() - self.last_analysis.generated_at < self.analysis_cache_duration):
            return self.last_analysis

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_entries = [
            entry for entry in self.feedback_entries.values()
            if entry.created_at > cutoff_date
        ]

        analysis = FeedbackAnalysis()
        analysis.total_feedback = len(recent_entries)

        # Calculate average rating
        ratings = [entry.rating for entry in recent_entries if entry.rating is not None]
        if ratings:
            analysis.average_rating = statistics.mean(ratings)

        # Count distributions
        analysis.feedback_types = dict(Counter(entry.feedback_type for entry in recent_entries))
        analysis.categories = dict(Counter(entry.category for entry in recent_entries))
        analysis.priority_distribution = dict(Counter(entry.priority for entry in recent_entries))
        analysis.status_distribution = dict(Counter(entry.status for entry in recent_entries))

        # Identify trending topics
        analysis.trending_topics = self._identify_trending_topics(recent_entries)

        # Sentiment analysis (simplified)
        analysis.sentiment_analysis = self._analyze_sentiment(recent_entries)

        # Common suggestions
        analysis.common_suggestions = self._extract_common_suggestions(recent_entries)

        # Urgent issues
        analysis.urgent_issues = self._identify_urgent_issues(recent_entries)

        # Cache the analysis
        self.last_analysis = analysis

        return analysis

    def _identify_trending_topics(self, entries: List[FeedbackEntry]) -> List[str]:
        """Identify trending topics from feedback"""
        all_tags = []
        for entry in entries:
            all_tags.extend(entry.tags)
            # Extract keywords from title and message
            words = (entry.title + " " + entry.message).lower().split()
            # Simple keyword extraction (could be enhanced with NLP)
            keywords = [word for word in words if len(word) > 3 and word not in
                       ['that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 'their']]
            all_tags.extend(keywords[:3])  # Take first 3 keywords

        # Count most common tags/keywords
        tag_counts = Counter(all_tags)
        trending = [tag for tag, count in tag_counts.most_common(10) if count > 1]

        return trending

    def _analyze_sentiment(self, entries: List[FeedbackEntry]) -> Dict[str, float]:
        """Perform basic sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'awesome', 'perfect', 'helpful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'useless', 'broken', 'slow', 'confusing']

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for entry in entries:
            text = (entry.title + " " + entry.message).lower()
            pos_words = sum(1 for word in positive_words if word in text)
            neg_words = sum(1 for word in negative_words if word in text)

            if pos_words > neg_words:
                positive_count += 1
            elif neg_words > pos_words:
                negative_count += 1
            else:
                neutral_count += 1

        total = positive_count + negative_count + neutral_count
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

        return {
            'positive': positive_count / total,
            'negative': negative_count / total,
            'neutral': neutral_count / total
        }

    def _extract_common_suggestions(self, entries: List[FeedbackEntry]) -> List[str]:
        """Extract common suggestions from feedback"""
        suggestions = []
        suggestion_keywords = ['should', 'could', 'would', 'please', 'add', 'implement', 'improve', 'fix']

        for entry in entries:
            if entry.feedback_type in ['feature', 'improvement']:
                text = entry.message.lower()
                if any(keyword in text for keyword in suggestion_keywords):
                    # Extract first sentence containing suggestion keywords
                    sentences = text.split('.')
                    for sentence in sentences:
                        if any(keyword in sentence for keyword in suggestion_keywords):
                            suggestions.append(sentence.strip())
                            break

        # Return most common suggestions (limit to 5)
        return list(set(suggestions))[:5]

    def _identify_urgent_issues(self, entries: List[FeedbackEntry]) -> List[str]:
        """Identify urgent issues that need immediate attention"""
        urgent_issues = []

        # High priority items
        high_priority = [entry for entry in entries if entry.priority in ['high', 'critical']]

        for entry in high_priority:
            if entry.status == 'new':
                urgent_issues.append(f"{entry.feedback_type.upper()}: {entry.title}")

        # Very low ratings
        low_ratings = [entry for entry in entries if entry.rating and entry.rating <= 2]
        for entry in low_ratings[:3]:  # Top 3 lowest ratings
            urgent_issues.append(f"Low Rating ({entry.rating}/5): {entry.title}")

        return urgent_issues[:5]  # Limit to 5 urgent issues

    def get_feedback_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get a summary of feedback statistics

        Args:
            days: Number of days to summarize

        Returns:
            Feedback summary
        """
        analysis = self.analyze_feedback(days)

        return {
            'period_days': days,
            'total_feedback': analysis.total_feedback,
            'average_rating': analysis.average_rating,
            'feedback_types': analysis.feedback_types,
            'categories': analysis.categories,
            'priority_distribution': analysis.priority_distribution,
            'status_distribution': analysis.status_distribution,
            'trending_topics': analysis.trending_topics[:5],  # Top 5
            'sentiment': analysis.sentiment_analysis,
            'urgent_issues': analysis.urgent_issues,
            'generated_at': analysis.generated_at.isoformat()
        }

    def export_feedback_data(self, filepath: str, format: str = "json") -> bool:
        """
        Export feedback data to file

        Args:
            filepath: Export file path
            format: Export format (json, csv)

        Returns:
            True if export successful, False otherwise
        """
        try:
            if format == "json":
                data = {
                    'exported_at': datetime.now().isoformat(),
                    'entries': [
                        {
                            **entry.__dict__,
                            'created_at': entry.created_at.isoformat(),
                            'updated_at': entry.updated_at.isoformat()
                        }
                        for entry in self.feedback_entries.values()
                    ]
                }

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

            elif format == "csv":
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow([
                        'id', 'user_id', 'feedback_type', 'title', 'message',
                        'rating', 'category', 'status', 'priority', 'created_at'
                    ])

                    # Write data
                    for entry in self.feedback_entries.values():
                        writer.writerow([
                            entry.id, entry.user_id, entry.feedback_type,
                            entry.title, entry.message, entry.rating,
                            entry.category, entry.status, entry.priority,
                            entry.created_at.isoformat()
                        ])

            self.logger.info(f"Feedback data exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting feedback data: {e}")
            return False

    async def cleanup_old_feedback(self, days_to_keep: int = 365):
        """
        Clean up old feedback entries

        Args:
            days_to_keep: Number of days of feedback to retain
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Identify old entries
        old_entries = [
            entry_id for entry_id, entry in self.feedback_entries.items()
            if entry.created_at < cutoff_date and entry.status in ['closed', 'addressed']
        ]

        # Remove old entries
        for entry_id in old_entries:
            del self.feedback_entries[entry_id]

        if old_entries:
            await self._save_feedback_entries()
            self.logger.info(f"Cleaned up {len(old_entries)} old feedback entries")


# Global feedback manager instance
feedback_manager = FeedbackManager()