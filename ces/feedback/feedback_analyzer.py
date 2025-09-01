"""
CES Feedback Analyzer - Phase 5 Launch

Advanced feedback analysis system with AI-powered insights, trend detection,
and automated recommendations for the CES community beta program.
"""

import asyncio
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class AnalysisType(Enum):
    """Types of feedback analysis"""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TREND_ANALYSIS = "trend_analysis"
    CATEGORY_ANALYSIS = "category_analysis"
    PRIORITY_ANALYSIS = "priority_analysis"
    USER_SEGMENTATION = "user_segmentation"
    FEATURE_IMPACT = "feature_impact"
    COMPETITIVE_ANALYSIS = "competitive_analysis"


class InsightPriority(Enum):
    """Priority levels for insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeedbackInsight:
    """AI-generated insight from feedback analysis"""
    insight_id: str
    analysis_type: AnalysisType
    title: str
    description: str
    priority: InsightPriority
    confidence_score: float
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    affected_features: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    implemented: bool = False


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    trend_id: str
    metric: str
    direction: str  # increasing, decreasing, stable
    magnitude: float
    time_period: str
    significance: float
    description: str
    data_points: List[Tuple[datetime, float]]


@dataclass
class UserSegment:
    """User segmentation analysis"""
    segment_id: str
    name: str
    description: str
    user_count: int
    characteristics: Dict[str, Any]
    satisfaction_score: float
    key_feedback_themes: List[str]


class FeedbackAnalyzer:
    """
    CES Feedback Analysis Engine - Phase 5

    Features:
    - AI-powered sentiment analysis and trend detection
    - Automated insight generation and prioritization
    - User segmentation and behavior analysis
    - Feature impact assessment
    - Predictive analytics for user satisfaction
    - Real-time feedback processing and alerting
    """

    def __init__(self):
        self.insights: Dict[str, FeedbackInsight] = {}
        self.trends: Dict[str, TrendAnalysis] = {}
        self.user_segments: Dict[str, UserSegment] = {}
        self.analysis_cache: Dict[str, Any] = {}

        # Analysis thresholds
        self.sentiment_thresholds = {
            'positive': 0.7,
            'neutral': 0.3,
            'negative': 0.0
        }

        self.priority_keywords = {
            InsightPriority.CRITICAL: ['crash', 'security', 'data loss', 'unusable', 'broken'],
            InsightPriority.HIGH: ['error', 'bug', 'slow', 'confusing', 'missing feature'],
            InsightPriority.MEDIUM: ['improvement', 'suggestion', 'enhancement'],
            InsightPriority.LOW: ['nice to have', 'minor', 'cosmetic']
        }

        logger.info("CES Feedback Analyzer initialized for Phase 5 Launch")

    async def analyze_feedback_batch(self, feedback_items: List[Dict[str, Any]],
                                   analysis_types: List[AnalysisType] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a batch of feedback items

        Args:
            feedback_items: List of feedback items to analyze
            analysis_types: Types of analysis to perform

        Returns:
            Comprehensive analysis results
        """
        if not analysis_types:
            analysis_types = list(AnalysisType)

        results = {
            'timestamp': datetime.now().isoformat(),
            'total_feedback': len(feedback_items),
            'analysis_types': [t.value for t in analysis_types],
            'insights': [],
            'trends': [],
            'segments': [],
            'recommendations': [],
            'summary': {}
        }

        # Perform each analysis type
        for analysis_type in analysis_types:
            try:
                if analysis_type == AnalysisType.SENTIMENT_ANALYSIS:
                    sentiment_results = await self._perform_sentiment_analysis(feedback_items)
                    results['insights'].extend(sentiment_results.get('insights', []))

                elif analysis_type == AnalysisType.TREND_ANALYSIS:
                    trend_results = await self._perform_trend_analysis(feedback_items)
                    results['trends'].extend(trend_results.get('trends', []))

                elif analysis_type == AnalysisType.CATEGORY_ANALYSIS:
                    category_results = await self._perform_category_analysis(feedback_items)
                    results['insights'].extend(category_results.get('insights', []))

                elif analysis_type == AnalysisType.USER_SEGMENTATION:
                    segment_results = await self._perform_user_segmentation(feedback_items)
                    results['segments'].extend(segment_results.get('segments', []))

                elif analysis_type == AnalysisType.FEATURE_IMPACT:
                    impact_results = await self._perform_feature_impact_analysis(feedback_items)
                    results['insights'].extend(impact_results.get('insights', []))

            except Exception as e:
                logger.error(f"Error in {analysis_type.value} analysis: {e}")

        # Generate overall recommendations
        results['recommendations'] = await self._generate_overall_recommendations(results)

        # Create summary
        results['summary'] = self._create_analysis_summary(results)

        return results

    async def generate_real_time_insights(self, feedback_stream: asyncio.Queue) -> asyncio.Queue:
        """
        Generate real-time insights from continuous feedback stream

        Args:
            feedback_stream: Queue of incoming feedback items

        Returns:
            Queue of generated insights
        """
        insight_queue = asyncio.Queue()

        async def insight_generator():
            buffer = []
            buffer_size = 10  # Process every 10 feedback items

            while True:
                try:
                    feedback_item = await feedback_stream.get()

                    buffer.append(feedback_item)

                    if len(buffer) >= buffer_size:
                        # Analyze buffer and generate insights
                        insights = await self._generate_buffer_insights(buffer)

                        for insight in insights:
                            await insight_queue.put(insight)

                        buffer.clear()

                except Exception as e:
                    logger.error(f"Error in real-time insight generation: {e}")

        # Start the insight generator
        asyncio.create_task(insight_generator())

        return insight_queue

    async def _perform_sentiment_analysis(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform sentiment analysis on feedback items"""
        sentiments = []

        for item in feedback_items:
            sentiment_score = self._analyze_sentiment(item.get('description', ''))
            sentiments.append(sentiment_score)

        # Calculate sentiment distribution
        positive = len([s for s in sentiments if s >= self.sentiment_thresholds['positive']])
        neutral = len([s for s in sentiments if self.sentiment_thresholds['neutral'] <= s < self.sentiment_thresholds['positive']])
        negative = len([s for s in sentiments if s < self.sentiment_thresholds['neutral']])

        # Generate insights based on sentiment
        insights = []

        if negative > positive * 2:
            insights.append(FeedbackInsight(
                insight_id=f"sentiment_{int(datetime.now().timestamp())}",
                analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
                title="Critical Sentiment Alert",
                description=f"Negative sentiment is {negative/positive:.1f}x higher than positive sentiment",
                priority=InsightPriority.CRITICAL,
                confidence_score=0.9,
                supporting_data={
                    'positive_count': positive,
                    'negative_count': negative,
                    'ratio': negative/positive if positive > 0 else float('inf')
                },
                recommendations=[
                    "Immediate investigation of negative feedback sources",
                    "Urgent customer outreach to affected users",
                    "Priority bug fixes for reported issues"
                ],
                affected_features=[]
            ))

        return {
            'sentiment_distribution': {
                'positive': positive,
                'neutral': neutral,
                'negative': negative
            },
            'average_sentiment': statistics.mean(sentiments) if sentiments else 0.0,
            'insights': insights
        }

    async def _perform_trend_analysis(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform trend analysis on feedback data"""
        # Group feedback by time periods
        daily_feedback = {}
        for item in feedback_items:
            date = item.get('created_at', datetime.now()).date()
            if date not in daily_feedback:
                daily_feedback[date] = []
            daily_feedback[date].append(item)

        # Analyze volume trends
        dates = sorted(daily_feedback.keys())
        volumes = [len(daily_feedback[date]) for date in dates]

        if len(volumes) >= 3:
            # Simple trend detection
            recent_avg = statistics.mean(volumes[-3:])
            earlier_avg = statistics.mean(volumes[:-3]) if len(volumes) > 3 else statistics.mean(volumes)

            if recent_avg > earlier_avg * 1.5:
                direction = "increasing"
                magnitude = (recent_avg - earlier_avg) / earlier_avg
            elif recent_avg < earlier_avg * 0.7:
                direction = "decreasing"
                magnitude = (earlier_avg - recent_avg) / earlier_avg
            else:
                direction = "stable"
                magnitude = 0.0

            trend = TrendAnalysis(
                trend_id=f"volume_trend_{int(datetime.now().timestamp())}",
                metric="feedback_volume",
                direction=direction,
                magnitude=magnitude,
                time_period="daily",
                significance=0.8,
                description=f"Feedback volume is {direction} by {magnitude:.1%}",
                data_points=[(dates[i], volumes[i]) for i in range(len(dates))]
            )

            return {'trends': [trend]}
        else:
            return {'trends': []}

    async def _perform_category_analysis(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback by categories"""
        category_counts = {}
        category_sentiments = {}

        for item in feedback_items:
            category = item.get('category', 'uncategorized')
            sentiment = self._analyze_sentiment(item.get('description', ''))

            if category not in category_counts:
                category_counts[category] = 0
                category_sentiments[category] = []

            category_counts[category] += 1
            category_sentiments[category].append(sentiment)

        # Find categories with issues
        insights = []
        for category, count in category_counts.items():
            sentiments = category_sentiments[category]
            avg_sentiment = statistics.mean(sentiments) if sentiments else 0.0

            if avg_sentiment < 0.4 and count >= 3:
                insights.append(FeedbackInsight(
                    insight_id=f"category_{category}_{int(datetime.now().timestamp())}",
                    analysis_type=AnalysisType.CATEGORY_ANALYSIS,
                    title=f"Issues in {category.replace('_', ' ').title()}",
                    description=f"Low satisfaction in {category} category ({avg_sentiment:.2f} average sentiment)",
                    priority=InsightPriority.HIGH,
                    confidence_score=0.8,
                    supporting_data={
                        'category': category,
                        'feedback_count': count,
                        'average_sentiment': avg_sentiment,
                        'sentiment_distribution': self._calculate_sentiment_distribution(sentiments)
                    },
                    recommendations=[
                        f"Review and improve {category} functionality",
                        "Gather more detailed feedback on specific issues",
                        "Prioritize fixes for this category"
                    ],
                    affected_features=[category]
                ))

        return {'insights': insights}

    async def _perform_user_segmentation(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform user segmentation analysis"""
        # Simple segmentation based on feedback patterns
        segments = {}

        for item in feedback_items:
            user_id = item.get('user_id', 'unknown')
            rating = item.get('rating', 3)
            category = item.get('category', 'general')

            # Create segment key based on behavior
            if rating >= 4:
                segment_key = "satisfied_users"
            elif rating <= 2:
                segment_key = "dissatisfied_users"
            else:
                segment_key = "neutral_users"

            if segment_key not in segments:
                segments[segment_key] = {
                    'users': set(),
                    'ratings': [],
                    'categories': set(),
                    'feedback_count': 0
                }

            segments[segment_key]['users'].add(user_id)
            segments[segment_key]['ratings'].append(rating)
            segments[segment_key]['categories'].add(category)
            segments[segment_key]['feedback_count'] += 1

        # Convert to UserSegment objects
        user_segments = []
        for segment_key, data in segments.items():
            avg_rating = statistics.mean(data['ratings']) if data['ratings'] else 0.0

            segment = UserSegment(
                segment_id=f"{segment_key}_{int(datetime.now().timestamp())}",
                name=segment_key.replace('_', ' ').title(),
                description=f"Users with {segment_key.replace('_', ' ')} feedback patterns",
                user_count=len(data['users']),
                characteristics={
                    'average_rating': avg_rating,
                    'feedback_count': data['feedback_count'],
                    'categories': list(data['categories'])
                },
                satisfaction_score=avg_rating / 5.0,  # Normalize to 0-1
                key_feedback_themes=list(data['categories'])
            )

            user_segments.append(segment)

        return {'segments': user_segments}

    async def _perform_feature_impact_analysis(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze impact of features on user satisfaction"""
        feature_feedback = {}

        for item in feedback_items:
            features = item.get('affected_features', [])
            rating = item.get('rating', 3)
            sentiment = self._analyze_sentiment(item.get('description', ''))

            for feature in features:
                if feature not in feature_feedback:
                    feature_feedback[feature] = {
                        'ratings': [],
                        'sentiments': [],
                        'feedback_count': 0
                    }

                feature_feedback[feature]['ratings'].append(rating)
                feature_feedback[feature]['sentiments'].append(sentiment)
                feature_feedback[feature]['feedback_count'] += 1

        # Generate insights for features with significant feedback
        insights = []
        for feature, data in feature_feedback.items():
            if data['feedback_count'] >= 5:  # Minimum feedback threshold
                avg_rating = statistics.mean(data['ratings'])
                avg_sentiment = statistics.mean(data['sentiments'])

                if avg_rating < 3.0 or avg_sentiment < 0.4:
                    insights.append(FeedbackInsight(
                        insight_id=f"feature_{feature}_{int(datetime.now().timestamp())}",
                        analysis_type=AnalysisType.FEATURE_IMPACT,
                        title=f"Feature Impact: {feature}",
                        description=f"Feature '{feature}' has low user satisfaction ({avg_rating:.1f} rating, {avg_sentiment:.2f} sentiment)",
                        priority=InsightPriority.HIGH,
                        confidence_score=0.85,
                        supporting_data={
                            'feature': feature,
                            'average_rating': avg_rating,
                            'average_sentiment': avg_sentiment,
                            'feedback_count': data['feedback_count']
                        },
                        recommendations=[
                            f"Review and improve {feature} functionality",
                            "Consider user testing for feature improvements",
                            "Gather specific feedback on feature pain points"
                        ],
                        affected_features=[feature]
                    ))

        return {'insights': insights}

    async def _generate_buffer_insights(self, feedback_buffer: List[Dict[str, Any]]) -> List[FeedbackInsight]:
        """Generate insights from a buffer of feedback items"""
        insights = []

        # Check for urgent patterns
        urgent_feedback = [
            item for item in feedback_buffer
            if self._is_urgent_feedback(item)
        ]

        if len(urgent_feedback) >= 3:
            insights.append(FeedbackInsight(
                insight_id=f"urgent_pattern_{int(datetime.now().timestamp())}",
                analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
                title="Urgent Feedback Pattern Detected",
                description=f"Multiple urgent feedback items detected ({len(urgent_feedback)} in recent batch)",
                priority=InsightPriority.CRITICAL,
                confidence_score=0.9,
                supporting_data={'urgent_count': len(urgent_feedback)},
                recommendations=[
                    "Immediate review of urgent feedback items",
                    "Escalate to development team if needed",
                    "Consider emergency response procedures"
                ],
                affected_features=[]
            ))

        return insights

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (simplified implementation)"""
        if not text:
            return 0.5

        text_lower = text.lower()

        positive_words = ['great', 'excellent', 'amazing', 'love', 'awesome', 'good', 'helpful', 'perfect', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'useless', 'broken', 'horrible', 'disappointed']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return 0.5

        sentiment_score = (positive_count - negative_count) / max(total_words * 0.1, 1)
        return max(0.0, min(1.0, 0.5 + sentiment_score))

    def _is_urgent_feedback(self, feedback_item: Dict[str, Any]) -> bool:
        """Check if feedback item is urgent"""
        text = feedback_item.get('description', '').lower()
        rating = feedback_item.get('rating', 5)

        urgent_indicators = [
            'crash', 'broken', 'unusable', 'emergency', 'urgent',
            'security issue', 'data loss', 'cannot access'
        ]

        has_urgent_keyword = any(indicator in text for indicator in urgent_indicators)
        has_low_rating = rating <= 2

        return has_urgent_keyword or has_low_rating

    def _calculate_sentiment_distribution(self, sentiments: List[float]) -> Dict[str, int]:
        """Calculate sentiment distribution"""
        return {
            'positive': len([s for s in sentiments if s >= self.sentiment_thresholds['positive']]),
            'neutral': len([s for s in sentiments if self.sentiment_thresholds['neutral'] <= s < self.sentiment_thresholds['positive']]),
            'negative': len([s for s in sentiments if s < self.sentiment_thresholds['neutral']])
        }

    async def _generate_overall_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations from analysis results"""
        recommendations = []

        insights = analysis_results.get('insights', [])
        trends = analysis_results.get('trends', [])

        # Recommendations based on insights
        critical_insights = [i for i in insights if i.get('priority') == InsightPriority.CRITICAL]
        if critical_insights:
            recommendations.append("Address critical insights immediately - these require urgent attention")

        high_priority_insights = [i for i in insights if i.get('priority') == InsightPriority.HIGH]
        if high_priority_insights:
            recommendations.append(f"Review {len(high_priority_insights)} high-priority insights for quick wins")

        # Recommendations based on trends
        for trend in trends:
            if trend.direction == "decreasing" and trend.metric == "feedback_volume":
                recommendations.append("Feedback volume is decreasing - investigate user engagement")
            elif trend.direction == "increasing" and trend.metric == "feedback_volume":
                recommendations.append("Feedback volume is increasing - ensure adequate support resources")

        # General recommendations
        if not recommendations:
            recommendations.append("Continue monitoring feedback patterns and user satisfaction")
            recommendations.append("Consider expanding feedback collection channels")

        return recommendations

    def _create_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of analysis results"""
        insights = analysis_results.get('insights', [])
        trends = analysis_results.get('trends', [])
        segments = analysis_results.get('segments', [])

        return {
            'total_insights': len(insights),
            'critical_insights': len([i for i in insights if i.get('priority') == InsightPriority.CRITICAL]),
            'high_priority_insights': len([i for i in insights if i.get('priority') == InsightPriority.HIGH]),
            'total_trends': len(trends),
            'total_segments': len(segments),
            'overall_health_score': self._calculate_health_score(insights, trends),
            'key_takeaways': self._extract_key_takeaways(insights, trends, segments)
        }

    def _calculate_health_score(self, insights: List[Dict[str, Any]], trends: List[Dict[str, Any]]) -> float:
        """Calculate overall health score from analysis"""
        if not insights:
            return 0.8  # Neutral score if no insights

        critical_count = len([i for i in insights if i.get('priority') == InsightPriority.CRITICAL])
        high_count = len([i for i in insights if i.get('priority') == InsightPriority.HIGH])

        # Penalize for critical and high priority issues
        penalty = (critical_count * 0.3) + (high_count * 0.1)

        base_score = 0.9  # Start with good score
        health_score = max(0.0, base_score - penalty)

        return health_score

    def _extract_key_takeaways(self, insights: List[Dict[str, Any]], trends: List[Dict[str, Any]],
                             segments: List[Dict[str, Any]]) -> List[str]:
        """Extract key takeaways from analysis"""
        takeaways = []

        # Insights takeaways
        if insights:
            critical_insights = [i for i in insights if i.get('priority') == InsightPriority.CRITICAL]
            if critical_insights:
                takeaways.append(f"{len(critical_insights)} critical issues require immediate attention")

        # Trends takeaways
        if trends:
            increasing_trends = [t for t in trends if t.direction == "increasing"]
            if increasing_trends:
                takeaways.append(f"{len(increasing_trends)} positive trends identified")

        # Segments takeaways
        if segments:
            dissatisfied_segments = [s for s in segments if s.satisfaction_score < 0.6]
            if dissatisfied_segments:
                takeaways.append(f"{len(dissatisfied_segments)} user segments show low satisfaction")

        if not takeaways:
            takeaways.append("Analysis shows generally positive user feedback patterns")

        return takeaways