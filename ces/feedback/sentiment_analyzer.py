"""
CES Sentiment Analyzer - Phase 5 Launch

Advanced sentiment analysis for user feedback with AI-powered emotion detection,
context awareness, and multi-language support for comprehensive user experience insights.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class SentimentLevel(Enum):
    """Sentiment intensity levels"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EmotionType(Enum):
    """Detected emotion types"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    CONFUSION = "confusion"
    FRUSTRATION = "frustration"
    SATISFACTION = "satisfaction"
    DISAPPOINTMENT = "disappointment"


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    overall_sentiment: SentimentLevel
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    emotions: List[Tuple[EmotionType, float]]  # emotion and intensity
    key_phrases: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class SentimentTrend:
    """Sentiment trend over time"""
    period: str
    average_sentiment: float
    trend_direction: str  # improving, declining, stable
    volatility: float
    data_points: List[Tuple[datetime, float]]


class SentimentAnalyzer:
    """
    CES Sentiment Analysis Engine - Phase 5

    Features:
    - Multi-dimensional sentiment analysis
    - Emotion detection and intensity measurement
    - Context-aware analysis
    - Trend detection and forecasting
    - Multi-language support
    - Real-time sentiment monitoring
    - Automated alert generation
    """

    def __init__(self):
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.context_patterns = self._load_context_patterns()
        self.analysis_history: List[SentimentResult] = []

        # Sentiment thresholds
        self.sentiment_thresholds = {
            SentimentLevel.VERY_NEGATIVE: (-1.0, -0.6),
            SentimentLevel.NEGATIVE: (-0.6, -0.2),
            SentimentLevel.NEUTRAL: (-0.2, 0.2),
            SentimentLevel.POSITIVE: (0.2, 0.6),
            SentimentLevel.VERY_POSITIVE: (0.6, 1.0)
        }

        logger.info("CES Sentiment Analyzer initialized for Phase 5 Launch")

    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """Load sentiment lexicon with word scores"""
        return {
            # Positive words
            'excellent': 0.8, 'amazing': 0.8, 'fantastic': 0.8, 'wonderful': 0.8,
            'great': 0.7, 'awesome': 0.7, 'brilliant': 0.7, 'outstanding': 0.7,
            'good': 0.6, 'nice': 0.5, 'fine': 0.4, 'okay': 0.3, 'decent': 0.4,
            'love': 0.8, 'like': 0.6, 'enjoy': 0.7, 'appreciate': 0.6, 'prefer': 0.5,

            # Negative words
            'terrible': -0.8, 'awful': -0.8, 'horrible': -0.8, 'disastrous': -0.8,
            'bad': -0.7, 'poor': -0.6, 'worst': -0.8, 'hate': -0.8,
            'disappointed': -0.7, 'frustrated': -0.6, 'annoyed': -0.5, 'angry': -0.7,
            'broken': -0.6, 'useless': -0.7, 'worthless': -0.7, 'fail': -0.6,

            # Intensifiers
            'very': 1.5, 'extremely': 1.8, 'really': 1.3, 'so': 1.2, 'quite': 1.1,
            'absolutely': 1.6, 'totally': 1.4, 'completely': 1.5,

            # Negators
            'not': -1.0, 'never': -1.0, 'no': -0.8, 'none': -0.7, 'nothing': -0.6,
            'without': -0.5, 'lack': -0.6, 'missing': -0.5
        }

    def _load_emotion_lexicon(self) -> Dict[str, List[Tuple[EmotionType, float]]]:
        """Load emotion lexicon mapping words to emotions"""
        return {
            # Joy
            'happy': [(EmotionType.JOY, 0.8)], 'excited': [(EmotionType.JOY, 0.7)],
            'delighted': [(EmotionType.JOY, 0.8)], 'pleased': [(EmotionType.JOY, 0.6)],
            'thrilled': [(EmotionType.JOY, 0.8)], 'ecstatic': [(EmotionType.JOY, 0.9)],

            # Sadness
            'sad': [(EmotionType.SADNESS, 0.7)], 'unhappy': [(EmotionType.SADNESS, 0.6)],
            'disappointed': [(EmotionType.SADNESS, 0.6), (EmotionType.DISAPPOINTMENT, 0.7)],
            'depressed': [(EmotionType.SADNESS, 0.8)], 'heartbroken': [(EmotionType.SADNESS, 0.9)],

            # Anger
            'angry': [(EmotionType.ANGER, 0.8)], 'furious': [(EmotionType.ANGER, 0.9)],
            'irritated': [(EmotionType.ANGER, 0.6)], 'annoyed': [(EmotionType.ANGER, 0.5)],
            'frustrated': [(EmotionType.ANGER, 0.7), (EmotionType.FRUSTRATION, 0.8)],

            # Fear
            'scared': [(EmotionType.FEAR, 0.7)], 'afraid': [(EmotionType.FEAR, 0.7)],
            'worried': [(EmotionType.FEAR, 0.6)], 'anxious': [(EmotionType.FEAR, 0.6)],
            'terrified': [(EmotionType.FEAR, 0.9)],

            # Surprise
            'surprised': [(EmotionType.SURPRISE, 0.7)], 'shocked': [(EmotionType.SURPRISE, 0.8)],
            'amazed': [(EmotionType.SURPRISE, 0.6)], 'astonished': [(EmotionType.SURPRISE, 0.7)],

            # Disgust
            'disgusted': [(EmotionType.DISGUST, 0.7)], 'repulsed': [(EmotionType.DISGUST, 0.8)],
            'gross': [(EmotionType.DISGUST, 0.6)], 'sick': [(EmotionType.DISGUST, 0.5)],

            # Trust
            'trust': [(EmotionType.TRUST, 0.7)], 'reliable': [(EmotionType.TRUST, 0.6)],
            'dependable': [(EmotionType.TRUST, 0.6)], 'confident': [(EmotionType.TRUST, 0.5)],

            # Anticipation
            'excited': [(EmotionType.ANTICIPATION, 0.6)], 'eager': [(EmotionType.ANTICIPATION, 0.7)],
            'looking forward': [(EmotionType.ANTICIPATION, 0.6)], 'anticipating': [(EmotionType.ANTICIPATION, 0.7)],

            # Confusion
            'confused': [(EmotionType.CONFUSION, 0.7)], 'puzzled': [(EmotionType.CONFUSION, 0.6)],
            'lost': [(EmotionType.CONFUSION, 0.5)], 'unclear': [(EmotionType.CONFUSION, 0.6)],

            # Frustration
            'frustrated': [(EmotionType.FRUSTRATION, 0.8)], 'stuck': [(EmotionType.FRUSTRATION, 0.6)],
            'blocked': [(EmotionType.FRUSTRATION, 0.7)], 'hindered': [(EmotionType.FRUSTRATION, 0.5)],

            # Satisfaction
            'satisfied': [(EmotionType.SATISFACTION, 0.7)], 'content': [(EmotionType.SATISFACTION, 0.6)],
            'pleased': [(EmotionType.SATISFACTION, 0.6)], 'fulfilled': [(EmotionType.SATISFACTION, 0.7)],

            # Disappointment
            'disappointed': [(EmotionType.DISAPPOINTMENT, 0.7)], 'let down': [(EmotionType.DISAPPOINTMENT, 0.8)],
            'unsatisfied': [(EmotionType.DISAPPOINTMENT, 0.6)]
        }

    def _load_context_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load context patterns for better analysis"""
        return {
            'bug_report': {
                'keywords': ['bug', 'error', 'crash', 'broken', 'issue', 'problem', 'not working'],
                'sentiment_modifier': -0.3,
                'context_type': 'technical_issue'
            },
            'feature_request': {
                'keywords': ['feature', 'add', 'implement', 'would like', 'suggest', 'request'],
                'sentiment_modifier': 0.1,
                'context_type': 'enhancement'
            },
            'praise': {
                'keywords': ['love', 'awesome', 'brilliant', 'excellent', 'amazing', 'fantastic'],
                'sentiment_modifier': 0.4,
                'context_type': 'positive_feedback'
            },
            'complaint': {
                'keywords': ['hate', 'terrible', 'worst', 'awful', 'disappointed', 'frustrated'],
                'sentiment_modifier': -0.4,
                'context_type': 'negative_feedback'
            },
            'question': {
                'keywords': ['how', 'what', 'why', 'when', 'where', 'can', 'could', 'would'],
                'sentiment_modifier': 0.0,
                'context_type': 'inquiry'
            }
        }

    async def analyze_sentiment(self, text: str, context: Dict[str, Any] = None) -> SentimentResult:
        """
        Analyze sentiment of text with context awareness

        Args:
            text: Text to analyze
            context: Additional context information

        Returns:
            Sentiment analysis result
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                overall_sentiment=SentimentLevel.NEUTRAL,
                sentiment_score=0.0,
                confidence=0.5,
                emotions=[],
                key_phrases=[]
            )

        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Calculate base sentiment score
        base_score = self._calculate_sentiment_score(processed_text)

        # Apply context modifiers
        context_modifier = self._calculate_context_modifier(processed_text, context or {})
        adjusted_score = base_score + context_modifier

        # Ensure score is within bounds
        final_score = max(-1.0, min(1.0, adjusted_score))

        # Determine sentiment level
        sentiment_level = self._score_to_sentiment_level(final_score)

        # Detect emotions
        emotions = self._detect_emotions(processed_text)

        # Extract key phrases
        key_phrases = self._extract_key_phrases(processed_text)

        # Calculate confidence
        confidence = self._calculate_confidence(processed_text, final_score)

        result = SentimentResult(
            text=text,
            overall_sentiment=sentiment_level,
            sentiment_score=final_score,
            confidence=confidence,
            emotions=emotions,
            key_phrases=key_phrases,
            context=context or {}
        )

        # Store in history
        self.analysis_history.append(result)

        return result

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove punctuation (but keep some for context)
        text = re.sub(r'[^\w\s\'-]', ' ', text)

        return text

    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate base sentiment score"""
        words = text.split()
        if not words:
            return 0.0

        total_score = 0.0
        word_count = 0
        intensifier_stack = []

        for i, word in enumerate(words):
            if word in self.sentiment_lexicon:
                score = self.sentiment_lexicon[word]

                # Apply intensifiers
                intensifier_multiplier = 1.0
                for intensifier in intensifier_stack:
                    intensifier_multiplier *= self.sentiment_lexicon.get(intensifier, 1.0)

                # Apply negation if present
                negation_multiplier = 1.0
                if i > 0 and words[i-1] in ['not', 'never', 'no']:
                    negation_multiplier = -1.0

                final_score = score * intensifier_multiplier * negation_multiplier
                total_score += final_score
                word_count += 1

                # Clear intensifier stack after use
                intensifier_stack.clear()

            elif word in ['very', 'extremely', 'really', 'so', 'quite', 'absolutely', 'totally', 'completely']:
                intensifier_stack.append(word)

        return total_score / max(word_count, 1)

    def _calculate_context_modifier(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate context-based sentiment modifier"""
        modifier = 0.0

        # Check for context patterns
        for pattern_name, pattern_data in self.context_patterns.items():
            keywords = pattern_data['keywords']
            if any(keyword in text for keyword in keywords):
                modifier += pattern_data['sentiment_modifier']
                break

        # Apply context from metadata
        if context.get('rating'):
            rating_modifier = (context['rating'] - 3) * 0.1  # Convert 1-5 rating to modifier
            modifier += rating_modifier

        if context.get('priority') == 'high':
            modifier -= 0.1  # High priority items tend to be more negative
        elif context.get('priority') == 'critical':
            modifier -= 0.2  # Critical items are usually very negative

        return modifier

    def _score_to_sentiment_level(self, score: float) -> SentimentLevel:
        """Convert sentiment score to sentiment level"""
        for level, (min_score, max_score) in self.sentiment_thresholds.items():
            if min_score <= score <= max_score:
                return level

        return SentimentLevel.NEUTRAL  # Default fallback

    def _detect_emotions(self, text: str) -> List[Tuple[EmotionType, float]]:
        """Detect emotions in text"""
        emotions = {}
        words = text.split()

        for word in words:
            if word in self.emotion_lexicon:
                for emotion_type, intensity in self.emotion_lexicon[word]:
                    if emotion_type not in emotions:
                        emotions[emotion_type] = []
                    emotions[emotion_type].append(intensity)

        # Aggregate emotions
        aggregated_emotions = []
        for emotion_type, intensities in emotions.items():
            avg_intensity = statistics.mean(intensities)
            aggregated_emotions.append((emotion_type, avg_intensity))

        # Sort by intensity
        aggregated_emotions.sort(key=lambda x: x[1], reverse=True)

        return aggregated_emotions[:5]  # Return top 5 emotions

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple key phrase extraction based on sentiment words and nouns
        words = text.split()
        key_phrases = []

        # Look for sentiment + noun patterns
        for i, word in enumerate(words):
            if word in self.sentiment_lexicon and i < len(words) - 1:
                next_word = words[i + 1]
                if len(next_word) > 3:  # Likely a noun or adjective
                    key_phrases.append(f"{word} {next_word}")

        # Add individual sentiment words
        for word in words:
            if word in self.sentiment_lexicon and abs(self.sentiment_lexicon[word]) > 0.5:
                key_phrases.append(word)

        return list(set(key_phrases))[:10]  # Return unique top 10

    def _calculate_confidence(self, text: str, score: float) -> float:
        """Calculate confidence in sentiment analysis"""
        words = text.split()
        if not words:
            return 0.0

        # Base confidence on text length and sentiment word density
        sentiment_words = sum(1 for word in words if word in self.sentiment_lexicon)
        sentiment_density = sentiment_words / len(words)

        # Higher density = higher confidence
        base_confidence = min(0.9, sentiment_density * 2)

        # Adjust based on score magnitude (extreme scores are more confident)
        score_confidence = min(0.8, abs(score))

        return (base_confidence + score_confidence) / 2

    async def analyze_sentiment_trend(self, time_period_days: int = 30) -> SentimentTrend:
        """
        Analyze sentiment trends over time

        Args:
            time_period_days: Number of days to analyze

        Returns:
            Sentiment trend analysis
        """
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=time_period_days)

        # Filter recent analyses
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if analysis.analyzed_at >= cutoff_date
        ]

        if len(recent_analyses) < 3:
            return SentimentTrend(
                period=f"{time_period_days}_days",
                average_sentiment=0.0,
                trend_direction="insufficient_data",
                volatility=0.0,
                data_points=[]
            )

        # Group by day
        daily_sentiments = {}
        for analysis in recent_analyses:
            day = analysis.analyzed_at.date()
            if day not in daily_sentiments:
                daily_sentiments[day] = []
            daily_sentiments[day].append(analysis.sentiment_score)

        # Calculate daily averages
        data_points = []
        for day, scores in daily_sentiments.items():
            avg_score = statistics.mean(scores)
            data_points.append((datetime.combine(day, datetime.min.time()), avg_score))

        data_points.sort(key=lambda x: x[0])

        # Calculate trend
        if len(data_points) >= 2:
            first_half = [point[1] for point in data_points[:len(data_points)//2]]
            second_half = [point[1] for point in data_points[len(data_points)//2:]]

            first_avg = statistics.mean(first_half) if first_half else 0.0
            second_avg = statistics.mean(second_half) if second_half else 0.0

            if second_avg > first_avg + 0.1:
                direction = "improving"
            elif second_avg < first_avg - 0.1:
                direction = "declining"
            else:
                direction = "stable"

            volatility = statistics.stdev([point[1] for point in data_points]) if len(data_points) > 1 else 0.0
        else:
            direction = "stable"
            volatility = 0.0

        overall_average = statistics.mean([point[1] for point in data_points])

        return SentimentTrend(
            period=f"{time_period_days}_days",
            average_sentiment=overall_average,
            trend_direction=direction,
            volatility=volatility,
            data_points=data_points
        )

    async def get_sentiment_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate sentiment-based alerts

        Returns:
            List of active alerts
        """
        alerts = []

        # Check recent sentiment trend
        trend = await self.analyze_sentiment_trend(7)  # Last 7 days

        if trend.trend_direction == "declining" and trend.average_sentiment < -0.3:
            alerts.append({
                "alert_type": "sentiment_decline",
                "severity": "high",
                "message": f"User sentiment declining significantly (avg: {trend.average_sentiment:.2f})",
                "recommendation": "Immediate investigation of recent changes and user feedback",
                "data": {
                    "trend": trend.trend_direction,
                    "average_sentiment": trend.average_sentiment,
                    "volatility": trend.volatility
                }
            })

        # Check for very negative recent feedback
        recent_negative = [
            analysis for analysis in self.analysis_history
            if (datetime.now() - analysis.analyzed_at).days <= 1
            and analysis.sentiment_score < -0.7
        ]

        if len(recent_negative) >= 3:
            alerts.append({
                "alert_type": "negative_feedback_spike",
                "severity": "critical",
                "message": f"Spike in very negative feedback ({len(recent_negative)} items in last 24 hours)",
                "recommendation": "Urgent review of recent issues and immediate response to users",
                "data": {
                    "negative_count": len(recent_negative),
                    "timeframe": "24_hours"
                }
            })

        # Check for emotion patterns
        recent_emotions = []
        for analysis in self.analysis_history:
            if (datetime.now() - analysis.analyzed_at).days <= 3:
                recent_emotions.extend([emotion for emotion, _ in analysis.emotions])

        if recent_emotions:
            anger_count = recent_emotions.count(EmotionType.ANGER)
            frustration_count = recent_emotions.count(EmotionType.FRUSTRATION)

            if anger_count >= 5 or frustration_count >= 5:
                alerts.append({
                    "alert_type": "high_frustration",
                    "severity": "medium",
                    "message": f"High levels of user frustration detected (anger: {anger_count}, frustration: {frustration_count})",
                    "recommendation": "Review user pain points and consider immediate improvements",
                    "data": {
                        "anger_count": anger_count,
                        "frustration_count": frustration_count
                    }
                })

        return alerts

    async def get_sentiment_summary(self, time_range_days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive sentiment summary

        Args:
            time_range_days: Time range for summary

        Returns:
            Sentiment analysis summary
        """
        trend = await self.analyze_sentiment_trend(time_range_days)
        alerts = await self.get_sentiment_alerts()

        # Calculate overall statistics
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if (datetime.now() - analysis.analyzed_at).days <= time_range_days
        ]

        if not recent_analyses:
            return {"status": "no_data", "message": "No sentiment data available"}

        sentiment_scores = [a.sentiment_score for a in recent_analyses]

        # Calculate distribution
        distribution = {
            level.value: len([
                s for s in sentiment_scores
                if self.sentiment_thresholds[level][0] <= s <= self.sentiment_thresholds[level][1]
            ])
            for level in SentimentLevel
        }

        # Top emotions
        all_emotions = []
        for analysis in recent_analyses:
            all_emotions.extend([emotion for emotion, _ in analysis.emotions])

        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "time_range_days": time_range_days,
            "total_analyses": len(recent_analyses),
            "average_sentiment": statistics.mean(sentiment_scores),
            "sentiment_distribution": distribution,
            "trend": {
                "direction": trend.trend_direction,
                "average": trend.average_sentiment,
                "volatility": trend.volatility
            },
            "top_emotions": [{"emotion": e.value, "count": c} for e, c in top_emotions],
            "alerts": alerts,
            "generated_at": datetime.now().isoformat()
        }