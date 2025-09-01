"""
CES Feedback Collector - Phase 5 Launch

Comprehensive feedback collection system for community beta program and public launch.
Supports multiple feedback channels, real-time collection, and integration with analytics.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class FeedbackChannel(Enum):
    """Feedback collection channels"""
    WEB_SURVEY = "web_survey"
    IN_APP_FEEDBACK = "in_app_feedback"
    EMAIL_SURVEY = "email_survey"
    SOCIAL_MEDIA = "social_media"
    SUPPORT_TICKET = "support_ticket"
    USER_INTERVIEW = "user_interview"
    BETA_FORUM = "beta_forum"
    API_FEEDBACK = "api_feedback"


class FeedbackPriority(Enum):
    """Feedback priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeedbackStatus(Enum):
    """Feedback processing status"""
    RECEIVED = "received"
    ANALYZING = "analyzing"
    PROCESSED = "processed"
    ACTIONED = "actioned"
    CLOSED = "closed"


@dataclass
class FeedbackItem:
    """Individual feedback item"""
    feedback_id: str
    user_id: str
    channel: FeedbackChannel
    category: str
    priority: FeedbackPriority
    status: FeedbackStatus
    title: str
    description: str
    rating: Optional[int] = None  # 1-5 scale
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    sentiment_score: Optional[float] = None
    urgency_score: Optional[float] = None
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None


@dataclass
class FeedbackCampaign:
    """Feedback collection campaign"""
    campaign_id: str
    name: str
    description: str
    channel: FeedbackChannel
    target_users: List[str]
    questions: List[Dict[str, Any]]
    start_date: datetime
    end_date: Optional[datetime] = None
    is_active: bool = True
    responses: List[Dict[str, Any]] = field(default_factory=list)
    analytics: Dict[str, Any] = field(default_factory=dict)


class FeedbackCollector:
    """
    CES Feedback Collection System - Phase 5

    Features:
    - Multi-channel feedback collection
    - Real-time feedback processing
    - Automated prioritization and routing
    - Integration with sentiment analysis
    - Campaign management for targeted feedback
    - Comprehensive analytics and reporting
    """

    def __init__(self):
        self.feedback_items: Dict[str, FeedbackItem] = {}
        self.feedback_campaigns: Dict[str, FeedbackCampaign] = {}
        self.feedback_handlers: Dict[FeedbackChannel, Callable] = {}
        self.analytics_cache: Dict[str, Any] = {}

        # Initialize default handlers
        self._setup_default_handlers()

        logger.info("CES Feedback Collector initialized for Phase 5 Launch")

    def _setup_default_handlers(self):
        """Setup default feedback channel handlers"""
        self.feedback_handlers = {
            FeedbackChannel.WEB_SURVEY: self._handle_web_survey,
            FeedbackChannel.IN_APP_FEEDBACK: self._handle_in_app_feedback,
            FeedbackChannel.EMAIL_SURVEY: self._handle_email_survey,
            FeedbackChannel.SOCIAL_MEDIA: self._handle_social_media,
            FeedbackChannel.SUPPORT_TICKET: self._handle_support_ticket,
            FeedbackChannel.USER_INTERVIEW: self._handle_user_interview,
            FeedbackChannel.BETA_FORUM: self._handle_beta_forum,
            FeedbackChannel.API_FEEDBACK: self._handle_api_feedback
        }

    async def collect_feedback(self, user_id: str, channel: FeedbackChannel,
                             category: str, title: str, description: str,
                             rating: int = None, metadata: Dict[str, Any] = None,
                             tags: List[str] = None) -> str:
        """
        Collect feedback from various channels

        Args:
            user_id: User providing feedback
            channel: Feedback collection channel
            category: Feedback category
            title: Feedback title
            description: Detailed feedback description
            rating: Optional rating (1-5)
            metadata: Additional metadata
            tags: Feedback tags

        Returns:
            Feedback ID
        """
        feedback_id = f"feedback_{int(datetime.now().timestamp())}_{user_id}"

        # Determine priority based on content and rating
        priority = self._calculate_priority(rating, description, category)

        feedback_item = FeedbackItem(
            feedback_id=feedback_id,
            user_id=user_id,
            channel=channel,
            category=category,
            priority=priority,
            status=FeedbackStatus.RECEIVED,
            title=title,
            description=description,
            rating=rating,
            metadata=metadata or {},
            tags=tags or []
        )

        self.feedback_items[feedback_id] = feedback_item

        # Process feedback asynchronously
        asyncio.create_task(self._process_feedback(feedback_item))

        logger.info(f"Collected feedback: {feedback_id} from user {user_id} via {channel.value}")
        return feedback_id

    async def create_feedback_campaign(self, name: str, description: str,
                                     channel: FeedbackChannel, target_users: List[str],
                                     questions: List[Dict[str, Any]],
                                     duration_days: int = 30) -> str:
        """
        Create a targeted feedback collection campaign

        Args:
            name: Campaign name
            description: Campaign description
            channel: Feedback channel
            target_users: Target user list
            questions: Survey questions
            duration_days: Campaign duration

        Returns:
            Campaign ID
        """
        campaign_id = f"campaign_{int(datetime.now().timestamp())}"

        campaign = FeedbackCampaign(
            campaign_id=campaign_id,
            name=name,
            description=description,
            channel=channel,
            target_users=target_users,
            questions=questions,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days)
        )

        self.feedback_campaigns[campaign_id] = campaign

        # Send campaign invitations
        await self._send_campaign_invitations(campaign)

        logger.info(f"Created feedback campaign: {campaign_id} - {name}")
        return campaign_id

    async def submit_campaign_response(self, campaign_id: str, user_id: str,
                                     responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit response to a feedback campaign

        Args:
            campaign_id: Campaign ID
            user_id: Responding user ID
            responses: Question responses

        Returns:
            Submission result
        """
        if campaign_id not in self.feedback_campaigns:
            return {"status": "error", "error": "Campaign not found"}

        campaign = self.feedback_campaigns[campaign_id]

        if not campaign.is_active:
            return {"status": "error", "error": "Campaign is not active"}

        if user_id not in campaign.target_users:
            return {"status": "error", "error": "User not eligible for this campaign"}

        # Record response
        response_data = {
            "user_id": user_id,
            "responses": responses,
            "submitted_at": datetime.now().isoformat(),
            "campaign_id": campaign_id
        }

        campaign.responses.append(response_data)

        # Update campaign analytics
        await self._update_campaign_analytics(campaign)

        logger.info(f"Campaign response submitted: {campaign_id} by user {user_id}")
        return {"status": "submitted", "response_id": len(campaign.responses) - 1}

    async def get_feedback_analytics(self, time_range_days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive feedback analytics

        Args:
            time_range_days: Analysis time range in days

        Returns:
            Feedback analytics data
        """
        cutoff_date = datetime.now() - timedelta(days=time_range_days)

        # Filter recent feedback
        recent_feedback = [
            f for f in self.feedback_items.values()
            if f.created_at >= cutoff_date
        ]

        if not recent_feedback:
            return {"status": "no_data", "message": "No feedback data available"}

        # Calculate analytics
        analytics = {
            "total_feedback": len(recent_feedback),
            "average_rating": self._calculate_average_rating(recent_feedback),
            "category_breakdown": self._calculate_category_breakdown(recent_feedback),
            "priority_distribution": self._calculate_priority_distribution(recent_feedback),
            "channel_usage": self._calculate_channel_usage(recent_feedback),
            "sentiment_analysis": self._calculate_sentiment_summary(recent_feedback),
            "trends": self._calculate_feedback_trends(recent_feedback),
            "top_issues": self._identify_top_issues(recent_feedback),
            "satisfaction_score": self._calculate_satisfaction_score(recent_feedback),
            "generated_at": datetime.now().isoformat()
        }

        return analytics

    async def get_campaign_analytics(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific feedback campaign

        Args:
            campaign_id: Campaign ID

        Returns:
            Campaign analytics
        """
        if campaign_id not in self.feedback_campaigns:
            return {"status": "error", "error": "Campaign not found"}

        campaign = self.feedback_campaigns[campaign_id]
        return campaign.analytics

    def _calculate_priority(self, rating: int, description: str, category: str) -> FeedbackPriority:
        """Calculate feedback priority based on content"""
        # Critical priority for very low ratings or urgent keywords
        urgent_keywords = ['crash', 'error', 'broken', 'urgent', 'critical', 'security']
        if rating and rating <= 2:
            return FeedbackPriority.CRITICAL
        elif any(keyword in description.lower() for keyword in urgent_keywords):
            return FeedbackPriority.HIGH
        elif rating and rating <= 3:
            return FeedbackPriority.MEDIUM
        else:
            return FeedbackPriority.LOW

    async def _process_feedback(self, feedback: FeedbackItem):
        """Process feedback item asynchronously"""
        try:
            # Update status to analyzing
            feedback.status = FeedbackStatus.ANALYZING
            feedback.updated_at = datetime.now()

            # Perform sentiment analysis (placeholder)
            feedback.sentiment_score = self._analyze_sentiment(feedback.description)

            # Calculate urgency score
            feedback.urgency_score = self._calculate_urgency(feedback)

            # Auto-assign based on priority and category
            if feedback.priority in [FeedbackPriority.HIGH, FeedbackPriority.CRITICAL]:
                feedback.assigned_to = self._auto_assign_feedback(feedback)

            # Update status to processed
            feedback.status = FeedbackStatus.PROCESSED
            feedback.updated_at = datetime.now()

            logger.info(f"Processed feedback: {feedback.feedback_id}")

        except Exception as e:
            logger.error(f"Error processing feedback {feedback.feedback_id}: {e}")
            feedback.status = FeedbackStatus.RECEIVED  # Reset status on error

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of feedback text (placeholder)"""
        # Simple keyword-based sentiment analysis
        positive_words = ['great', 'excellent', 'amazing', 'love', 'awesome', 'good', 'helpful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'useless']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return 0.8
        elif negative_count > positive_count:
            return 0.2
        else:
            return 0.5

    def _calculate_urgency(self, feedback: FeedbackItem) -> float:
        """Calculate urgency score for feedback"""
        urgency_score = 0.0

        # Priority contribution
        priority_weights = {
            FeedbackPriority.CRITICAL: 1.0,
            FeedbackPriority.HIGH: 0.7,
            FeedbackPriority.MEDIUM: 0.4,
            FeedbackPriority.LOW: 0.1
        }
        urgency_score += priority_weights.get(feedback.priority, 0.1)

        # Rating contribution (lower rating = higher urgency)
        if feedback.rating:
            urgency_score += (6 - feedback.rating) * 0.1

        # Sentiment contribution
        if feedback.sentiment_score is not None:
            urgency_score += (1 - feedback.sentiment_score) * 0.3

        return min(1.0, urgency_score)

    def _auto_assign_feedback(self, feedback: FeedbackItem) -> str:
        """Auto-assign feedback to appropriate team member"""
        # Simple assignment logic based on category
        category_assignments = {
            'bug': 'development_team',
            'feature_request': 'product_team',
            'performance': 'devops_team',
            'usability': 'design_team',
            'security': 'security_team'
        }

        return category_assignments.get(feedback.category, 'support_team')

    async def _send_campaign_invitations(self, campaign: FeedbackCampaign):
        """Send campaign invitations to target users"""
        # Placeholder for sending invitations
        logger.info(f"Sending campaign invitations for: {campaign.campaign_id}")

    async def _update_campaign_analytics(self, campaign: FeedbackCampaign):
        """Update campaign analytics"""
        total_responses = len(campaign.responses)
        total_targeted = len(campaign.target_users)
        response_rate = total_responses / total_targeted if total_targeted > 0 else 0

        campaign.analytics = {
            "total_responses": total_responses,
            "response_rate": response_rate,
            "completion_rate": response_rate,  # Simplified
            "average_completion_time": "N/A",  # Would need timestamps
            "last_updated": datetime.now().isoformat()
        }

    def _calculate_average_rating(self, feedback_items: List[FeedbackItem]) -> float:
        """Calculate average rating from feedback items"""
        ratings = [f.rating for f in feedback_items if f.rating is not None]
        return sum(ratings) / len(ratings) if ratings else 0.0

    def _calculate_category_breakdown(self, feedback_items: List[FeedbackItem]) -> Dict[str, int]:
        """Calculate feedback count by category"""
        breakdown = {}
        for item in feedback_items:
            breakdown[item.category] = breakdown.get(item.category, 0) + 1
        return breakdown

    def _calculate_priority_distribution(self, feedback_items: List[FeedbackItem]) -> Dict[str, int]:
        """Calculate feedback count by priority"""
        distribution = {}
        for item in feedback_items:
            distribution[item.priority.value] = distribution.get(item.priority.value, 0) + 1
        return distribution

    def _calculate_channel_usage(self, feedback_items: List[FeedbackItem]) -> Dict[str, int]:
        """Calculate feedback count by channel"""
        usage = {}
        for item in feedback_items:
            usage[item.channel.value] = usage.get(item.channel.value, 0) + 1
        return usage

    def _calculate_sentiment_summary(self, feedback_items: List[FeedbackItem]) -> Dict[str, Any]:
        """Calculate sentiment analysis summary"""
        sentiments = [f.sentiment_score for f in feedback_items if f.sentiment_score is not None]

        if not sentiments:
            return {"average_sentiment": 0.0, "sentiment_distribution": {}}

        return {
            "average_sentiment": sum(sentiments) / len(sentiments),
            "sentiment_distribution": {
                "positive": len([s for s in sentiments if s >= 0.7]),
                "neutral": len([s for s in sentiments if 0.3 <= s < 0.7]),
                "negative": len([s for s in sentiments if s < 0.3])
            }
        }

    def _calculate_feedback_trends(self, feedback_items: List[FeedbackItem]) -> Dict[str, Any]:
        """Calculate feedback trends over time"""
        # Group by day
        daily_counts = {}
        for item in feedback_items:
            day = item.created_at.date()
            daily_counts[day] = daily_counts.get(day, 0) + 1

        return {
            "daily_volume": dict(sorted(daily_counts.items())),
            "trend": "increasing" if len(daily_counts) > 1 and
                   list(daily_counts.values())[-1] > list(daily_counts.values())[0] else "stable"
        }

    def _identify_top_issues(self, feedback_items: List[FeedbackItem]) -> List[Dict[str, Any]]:
        """Identify top issues from feedback"""
        issue_counts = {}
        for item in feedback_items:
            # Simple keyword extraction for issues
            words = item.description.lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    issue_counts[word] = issue_counts.get(word, 0) + 1

        # Return top 10 issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"issue": issue, "count": count} for issue, count in sorted_issues[:10]]

    def _calculate_satisfaction_score(self, feedback_items: List[FeedbackItem]) -> float:
        """Calculate overall satisfaction score"""
        ratings = [f.rating for f in feedback_items if f.rating is not None]
        sentiments = [f.sentiment_score for f in feedback_items if f.sentiment_score is not None]

        if not ratings and not sentiments:
            return 0.0

        rating_score = sum(ratings) / len(ratings) if ratings else 0.0
        sentiment_score = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Weighted average
        return (rating_score * 0.6 + sentiment_score * 0.4) if ratings or sentiments else 0.0

    # Channel-specific handlers
    async def _handle_web_survey(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web survey feedback"""
        return {"status": "processed", "channel": "web_survey"}

    async def _handle_in_app_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle in-app feedback"""
        return {"status": "processed", "channel": "in_app"}

    async def _handle_email_survey(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email survey feedback"""
        return {"status": "processed", "channel": "email"}

    async def _handle_social_media(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle social media feedback"""
        return {"status": "processed", "channel": "social_media"}

    async def _handle_support_ticket(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle support ticket feedback"""
        return {"status": "processed", "channel": "support"}

    async def _handle_user_interview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user interview feedback"""
        return {"status": "processed", "channel": "interview"}

    async def _handle_beta_forum(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle beta forum feedback"""
        return {"status": "processed", "channel": "forum"}

    async def _handle_api_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API feedback"""
        return {"status": "processed", "channel": "api"}