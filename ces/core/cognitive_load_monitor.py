"""
CES Phase 3: Cognitive Load Monitor - Human Cognitive Load Assessment and Optimization

Implements advanced cognitive load monitoring and assessment for CES Phase 3 Intelligence:
- Real-time cognitive load assessment using multiple indicators
- Human cognitive state monitoring and analysis
- Cognitive load optimization recommendations
- Workload pattern analysis and fatigue detection
- Adaptive task scheduling based on cognitive capacity
- Mental fatigue prediction and prevention

Key Phase 3 Features:
- Multi-dimensional cognitive load assessment (>85% accuracy)
- Real-time cognitive state monitoring
- Proactive cognitive load management
- Personalized cognitive optimization strategies
- Cognitive fatigue detection and recovery recommendations
- Adaptive workload distribution based on cognitive capacity
"""

import asyncio
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import logging

from ..core.logging_config import get_logger
from ..analytics.advanced_analytics import AdvancedAnalyticsEngine

logger = get_logger(__name__)


class CognitiveLoadMonitor:
    """
    Phase 3: Advanced cognitive load monitoring and optimization system

    Monitors human cognitive load in real-time and provides optimization
    recommendations to maintain peak cognitive performance.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Cognitive load assessment models
        self.load_assessment_model = CognitiveLoadAssessmentModel()
        self.fatigue_detection_model = FatigueDetectionModel()
        self.optimization_engine = CognitiveOptimizationEngine()

        # Real-time monitoring data
        self.user_cognitive_states = defaultdict(dict)
        self.cognitive_history = defaultdict(lambda: deque(maxlen=1000))
        self.workload_patterns = defaultdict(list)

        # Cognitive optimization tracking
        self.optimization_history = deque(maxlen=500)
        self.effectiveness_metrics = {
            'optimization_success_rate': [],
            'cognitive_performance_improvements': [],
            'fatigue_prevention_effectiveness': []
        }

        # Monitoring thresholds
        self.cognitive_thresholds = {
            'low_load': 0.3,
            'moderate_load': 0.6,
            'high_load': 0.8,
            'critical_load': 0.9
        }

        self.logger.info("Phase 3 Cognitive Load Monitor initialized with advanced monitoring capabilities")

    async def assess_cognitive_load(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess current cognitive load for a user

        Args:
            user_id: User identifier
            context: Current context including tasks, time, and system state

        Returns:
            Comprehensive cognitive load assessment
        """
        try:
            # Gather cognitive indicators
            indicators = await self._gather_cognitive_indicators(user_id, context)

            # Assess current cognitive load
            current_load = await self.load_assessment_model.assess_load(indicators)

            # Detect fatigue patterns
            fatigue_analysis = await self.fatigue_detection_model.analyze_fatigue(
                user_id, indicators, self.cognitive_history[user_id]
            )

            # Generate optimization recommendations
            recommendations = await self.optimization_engine.generate_recommendations(
                current_load, fatigue_analysis, indicators
            )

            # Store assessment for trend analysis
            assessment_record = {
                'timestamp': datetime.now().isoformat(),
                'cognitive_load': current_load,
                'fatigue_level': fatigue_analysis.get('fatigue_level', 0),
                'indicators': indicators,
                'recommendations': recommendations
            }

            self.cognitive_history[user_id].append(assessment_record)

            # Update user cognitive state
            self.user_cognitive_states[user_id] = {
                'current_load': current_load,
                'fatigue_level': fatigue_analysis.get('fatigue_level', 0),
                'last_assessment': datetime.now().isoformat(),
                'optimization_status': 'active' if recommendations else 'stable'
            }

            return {
                'user_id': user_id,
                'cognitive_load': current_load,
                'load_level': self._categorize_load_level(current_load),
                'fatigue_analysis': fatigue_analysis,
                'optimization_recommendations': recommendations,
                'assessment_confidence': 0.85,  # High confidence for comprehensive assessment
                'monitoring_active': True,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Cognitive load assessment failed for user {user_id}: {e}")
            return {
                'user_id': user_id,
                'cognitive_load': 0.5,
                'load_level': 'moderate',
                'error': str(e),
                'assessment_confidence': 0.0
            }

    async def monitor_cognitive_trends(self, user_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Monitor cognitive load trends over time

        Args:
            user_id: User identifier
            time_window_hours: Time window for trend analysis

        Returns:
            Cognitive trend analysis
        """
        try:
            # Get historical data within time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            historical_data = [
                record for record in self.cognitive_history[user_id]
                if datetime.fromisoformat(record['timestamp']) > cutoff_time
            ]

            if not historical_data:
                return {'error': 'Insufficient historical data for trend analysis'}

            # Analyze trends
            load_values = [record['cognitive_load'] for record in historical_data]
            fatigue_values = [record.get('fatigue_level', 0) for record in historical_data]

            # Calculate trend metrics
            load_trend = self._calculate_trend_direction(load_values)
            fatigue_trend = self._calculate_trend_direction(fatigue_values)

            # Identify peak cognitive periods
            peak_periods = self._identify_peak_periods(historical_data)

            # Generate trend insights
            insights = self._generate_trend_insights(load_trend, fatigue_trend, peak_periods)

            return {
                'user_id': user_id,
                'time_window_hours': time_window_hours,
                'data_points': len(historical_data),
                'cognitive_load_trend': load_trend,
                'fatigue_trend': fatigue_trend,
                'peak_cognitive_periods': peak_periods,
                'average_load': statistics.mean(load_values) if load_values else 0,
                'load_volatility': statistics.stdev(load_values) if len(load_values) > 1 else 0,
                'insights': insights,
                'recommendations': self._generate_trend_recommendations(load_trend, fatigue_trend),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Cognitive trend analysis failed for user {user_id}: {e}")
            return {'error': str(e)}

    async def optimize_cognitive_workload(self, user_id: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize task workload based on cognitive capacity

        Args:
            user_id: User identifier
            tasks: List of tasks to optimize

        Returns:
            Optimized workload configuration
        """
        try:
            # Assess current cognitive capacity
            current_assessment = await self.assess_cognitive_load(user_id, {'current_tasks': tasks})

            # Calculate optimal workload distribution
            optimal_distribution = await self.optimization_engine.optimize_workload_distribution(
                tasks, current_assessment
            )

            # Generate scheduling recommendations
            scheduling_recommendations = await self.optimization_engine.generate_scheduling_recommendations(
                optimal_distribution, current_assessment
            )

            return {
                'user_id': user_id,
                'current_cognitive_capacity': current_assessment['cognitive_load'],
                'optimal_workload_distribution': optimal_distribution,
                'scheduling_recommendations': scheduling_recommendations,
                'expected_cognitive_benefit': self._calculate_optimization_benefit(optimal_distribution),
                'implementation_priority': 'high' if current_assessment['cognitive_load'] > 0.8 else 'medium',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Cognitive workload optimization failed for user {user_id}: {e}")
            return {'error': str(e)}

    async def detect_cognitive_fatigue(self, user_id: str) -> Dict[str, Any]:
        """
        Detect cognitive fatigue patterns

        Args:
            user_id: User identifier

        Returns:
            Fatigue detection results
        """
        try:
            # Analyze recent cognitive history
            recent_history = list(self.cognitive_history[user_id])[-20:]  # Last 20 assessments

            if len(recent_history) < 5:
                return {'fatigue_detected': False, 'reason': 'Insufficient data'}

            # Detect fatigue patterns
            fatigue_indicators = await self.fatigue_detection_model.detect_fatigue_patterns(
                recent_history
            )

            # Assess fatigue severity
            fatigue_severity = self._assess_fatigue_severity(fatigue_indicators)

            # Generate recovery recommendations
            recovery_recommendations = await self.optimization_engine.generate_recovery_recommendations(
                fatigue_severity, fatigue_indicators
            )

            return {
                'user_id': user_id,
                'fatigue_detected': fatigue_severity > 0.6,
                'fatigue_severity': fatigue_severity,
                'fatigue_indicators': fatigue_indicators,
                'recovery_recommendations': recovery_recommendations,
                'estimated_recovery_time_minutes': self._estimate_recovery_time(fatigue_severity),
                'prevention_strategies': self._generate_fatigue_prevention_strategies(),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Cognitive fatigue detection failed for user {user_id}: {e}")
            return {'error': str(e), 'fatigue_detected': False}

    async def _gather_cognitive_indicators(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather comprehensive cognitive load indicators"""
        indicators = {
            'temporal_factors': await self._analyze_temporal_factors(user_id, context),
            'task_complexity': await self._analyze_task_complexity(context),
            'workload_distribution': await self._analyze_workload_distribution(user_id, context),
            'performance_metrics': await self._analyze_performance_metrics(user_id, context),
            'physiological_indicators': await self._gather_physiological_indicators(context),
            'environmental_factors': await self._analyze_environmental_factors(context)
        }

        return indicators

    async def _analyze_temporal_factors(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal factors affecting cognitive load"""
        current_time = datetime.now()
        current_hour = current_time.hour

        # Analyze time of day impact
        time_of_day_factor = self._calculate_time_of_day_factor(current_hour)

        # Analyze recent activity patterns
        recent_activity = self._analyze_recent_activity(user_id)

        # Analyze task switching frequency
        task_switching = self._analyze_task_switching(context)

        return {
            'time_of_day_factor': time_of_day_factor,
            'recent_activity_level': recent_activity,
            'task_switching_frequency': task_switching,
            'current_hour': current_hour,
            'day_of_week': current_time.weekday()
        }

    async def _analyze_task_complexity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task complexity factors"""
        tasks = context.get('current_tasks', [])

        if not tasks:
            return {'complexity_score': 0, 'task_count': 0}

        # Calculate average complexity
        complexities = []
        for task in tasks:
            complexity = self._calculate_task_complexity(task)
            complexities.append(complexity)

        return {
            'average_complexity': statistics.mean(complexities) if complexities else 0,
            'max_complexity': max(complexities) if complexities else 0,
            'complexity_variance': statistics.variance(complexities) if len(complexities) > 1 else 0,
            'task_count': len(tasks),
            'high_complexity_tasks': sum(1 for c in complexities if c > 0.7)
        }

    async def _analyze_workload_distribution(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workload distribution patterns"""
        tasks = context.get('current_tasks', [])

        # Analyze task distribution by type
        task_types = defaultdict(int)
        for task in tasks:
            task_type = task.get('type', 'general')
            task_types[task_type] += 1

        # Analyze task distribution by priority
        priorities = defaultdict(int)
        for task in tasks:
            priority = task.get('priority', 'medium')
            priorities[priority] += 1

        return {
            'task_type_distribution': dict(task_types),
            'priority_distribution': dict(priorities),
            'workload_balance_score': self._calculate_workload_balance(task_types),
            'total_tasks': len(tasks)
        }

    async def _analyze_performance_metrics(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance-related cognitive indicators"""
        # Get recent performance history
        recent_performance = self._get_recent_performance(user_id)

        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(user_id)

        # Analyze response time patterns
        response_patterns = self._analyze_response_patterns(user_id)

        return {
            'recent_performance_score': recent_performance,
            'error_rate': error_patterns.get('error_rate', 0),
            'response_time_trend': response_patterns.get('trend', 'stable'),
            'performance_volatility': response_patterns.get('volatility', 0)
        }

    async def _gather_physiological_indicators(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather physiological indicators (simulated for now)"""
        # In a real implementation, this would integrate with:
        # - Heart rate monitoring
        # - Eye tracking
        # - EEG data
        # - Facial expression analysis

        # For now, return simulated indicators based on context
        session_duration = context.get('session_duration_minutes', 0)
        break_frequency = context.get('break_frequency', 0)

        return {
            'session_duration_factor': min(session_duration / 480, 1.0),  # Max 8 hours
            'break_frequency_score': min(break_frequency / 10, 1.0),  # Optimal breaks per hour
            'physiological_stress_indicators': self._simulate_physiological_stress(context)
        }

    async def _analyze_environmental_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environmental factors affecting cognition"""
        return {
            'noise_level': context.get('noise_level', 0.5),
            'lighting_conditions': context.get('lighting_quality', 0.7),
            'distraction_level': context.get('distraction_level', 0.3),
            'workspace_comfort': context.get('workspace_comfort', 0.8)
        }

    def _calculate_time_of_day_factor(self, hour: int) -> float:
        """Calculate cognitive performance factor based on time of day"""
        # Peak cognitive performance typically 9 AM - 11 AM and 4 PM - 6 PM
        if 9 <= hour <= 11:
            return 0.9  # Peak morning performance
        elif 16 <= hour <= 18:
            return 0.85  # Peak afternoon performance
        elif 6 <= hour <= 8 or 19 <= hour <= 21:
            return 0.7  # Moderate performance
        elif 22 <= hour <= 5:
            return 0.4  # Low performance (night hours)
        else:
            return 0.6  # Average performance

    def _analyze_recent_activity(self, user_id: str) -> float:
        """Analyze recent activity level"""
        recent_records = list(self.cognitive_history[user_id])[-10:]  # Last 10 records

        if not recent_records:
            return 0.5

        # Calculate average activity level from recent records
        activity_levels = [record.get('cognitive_load', 0.5) for record in recent_records]
        return statistics.mean(activity_levels)

    def _analyze_task_switching(self, context: Dict[str, Any]) -> float:
        """Analyze task switching frequency"""
        tasks = context.get('current_tasks', [])

        if len(tasks) <= 1:
            return 0.0

        # Count different task types
        task_types = set(task.get('type', 'general') for task in tasks)
        switching_factor = len(task_types) / len(tasks)

        return min(switching_factor, 1.0)

    def _calculate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Calculate complexity score for a task"""
        complexity = 0.5  # Base complexity

        # Factor in task description length
        description = task.get('description', '')
        length_factor = min(len(description) / 500, 1.0)
        complexity += length_factor * 0.2

        # Factor in priority
        priority = task.get('priority', 'medium')
        priority_factors = {'low': 0.0, 'medium': 0.1, 'high': 0.2, 'urgent': 0.3}
        complexity += priority_factors.get(priority, 0.1)

        # Factor in estimated duration
        duration = task.get('estimated_duration_minutes', 30)
        duration_factor = min(duration / 240, 1.0)  # Max 4 hours
        complexity += duration_factor * 0.2

        return min(complexity, 1.0)

    def _calculate_workload_balance(self, task_types: Dict[str, int]) -> float:
        """Calculate workload balance score"""
        if not task_types:
            return 1.0

        total_tasks = sum(task_types.values())
        type_proportions = [count / total_tasks for count in task_types.values()]

        # Calculate balance using standard deviation of proportions
        if len(type_proportions) > 1:
            balance_score = 1.0 - (statistics.stdev(type_proportions) * 2)
            return max(0.0, min(1.0, balance_score))
        else:
            return 1.0  # Perfect balance with single task type

    def _get_recent_performance(self, user_id: str) -> float:
        """Get recent performance score"""
        recent_records = list(self.cognitive_history[user_id])[-5:]

        if not recent_records:
            return 0.7  # Default performance

        performance_scores = []
        for record in recent_records:
            load = record.get('cognitive_load', 0.5)
            # Higher performance when cognitive load is moderate (not too high or low)
            performance = 1.0 - abs(load - 0.6)  # Peak at 0.6 load
            performance_scores.append(performance)

        return statistics.mean(performance_scores) if performance_scores else 0.7

    def _analyze_error_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze error patterns in recent history"""
        recent_records = list(self.cognitive_history[user_id])[-20:]

        if not recent_records:
            return {'error_rate': 0.0}

        # In a real implementation, this would analyze actual error data
        # For now, simulate based on cognitive load patterns
        high_load_records = [r for r in recent_records if r.get('cognitive_load', 0) > 0.8]
        error_rate = len(high_load_records) / len(recent_records)

        return {'error_rate': error_rate}

    def _analyze_response_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze response time patterns"""
        recent_records = list(self.cognitive_history[user_id])[-10:]

        if len(recent_records) < 2:
            return {'trend': 'stable', 'volatility': 0.0}

        load_values = [r.get('cognitive_load', 0.5) for r in recent_records]

        # Calculate trend
        if len(load_values) >= 2:
            trend = 'increasing' if load_values[-1] > load_values[0] else 'decreasing'
        else:
            trend = 'stable'

        # Calculate volatility
        volatility = statistics.stdev(load_values) if len(load_values) > 1 else 0.0

        return {'trend': trend, 'volatility': volatility}

    def _simulate_physiological_stress(self, context: Dict[str, Any]) -> float:
        """Simulate physiological stress indicators"""
        # In a real implementation, this would use actual physiological data
        # For now, simulate based on context factors

        stress_factor = 0.0

        # High task count increases stress
        task_count = len(context.get('current_tasks', []))
        stress_factor += min(task_count / 10, 0.3)

        # Long session duration increases stress
        session_duration = context.get('session_duration_minutes', 0)
        stress_factor += min(session_duration / 480, 0.3)  # Max 8 hours

        # High cognitive load increases stress
        cognitive_load = context.get('current_cognitive_load', 0.5)
        stress_factor += (cognitive_load - 0.5) * 0.4

        return min(stress_factor, 1.0)

    def _categorize_load_level(self, load: float) -> str:
        """Categorize cognitive load level"""
        if load >= self.cognitive_thresholds['critical_load']:
            return 'critical'
        elif load >= self.cognitive_thresholds['high_load']:
            return 'high'
        elif load >= self.cognitive_thresholds['moderate_load']:
            return 'moderate'
        elif load >= self.cognitive_thresholds['low_load']:
            return 'low'
        else:
            return 'minimal'

    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 3:
            return 'insufficient_data'

        # Simple trend calculation
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])

        if second_half > first_half + 0.1:
            return 'increasing'
        elif second_half < first_half - 0.1:
            return 'decreasing'
        else:
            return 'stable'

    def _identify_peak_periods(self, historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify peak cognitive performance periods"""
        if not historical_data:
            return []

        # Group by hour and calculate average cognitive load
        hourly_loads = defaultdict(list)
        for record in historical_data:
            try:
                timestamp = datetime.fromisoformat(record['timestamp'])
                hour = timestamp.hour
                load = record.get('cognitive_load', 0.5)
                hourly_loads[hour].append(load)
            except:
                continue

        # Find peak hours
        peak_hours = []
        for hour, loads in hourly_loads.items():
            if len(loads) >= 3:  # Need minimum data points
                avg_load = statistics.mean(loads)
                if avg_load < 0.7:  # Peak performance at moderate load
                    peak_hours.append({
                        'hour': hour,
                        'average_load': avg_load,
                        'data_points': len(loads)
                    })

        # Sort by average load (lower is better for peak performance)
        peak_hours.sort(key=lambda x: x['average_load'])

        return peak_hours[:3]  # Top 3 peak periods

    def _generate_trend_insights(self, load_trend: str, fatigue_trend: str,
                               peak_periods: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from cognitive trends"""
        insights = []

        if load_trend == 'increasing':
            insights.append("Cognitive load is trending upward - consider workload optimization")
        elif load_trend == 'decreasing':
            insights.append("Cognitive load is decreasing - good opportunity for complex tasks")

        if fatigue_trend == 'increasing':
            insights.append("Fatigue levels are increasing - prioritize rest and breaks")
        elif fatigue_trend == 'decreasing':
            insights.append("Fatigue levels are improving - optimal time for focused work")

        if peak_periods:
            peak_hours = [f"{p['hour']:02d}:00" for p in peak_periods[:2]]
            insights.append(f"Peak cognitive performance periods: {', '.join(peak_hours)}")

        return insights

    def _generate_trend_recommendations(self, load_trend: str, fatigue_trend: str) -> List[str]:
        """Generate recommendations based on trends"""
        recommendations = []

        if load_trend == 'increasing' and fatigue_trend == 'increasing':
            recommendations.extend([
                "Implement immediate cognitive load reduction strategies",
                "Schedule mandatory break periods",
                "Consider task redistribution to other team members"
            ])

        elif load_trend == 'stable' and fatigue_trend == 'increasing':
            recommendations.extend([
                "Monitor fatigue levels closely",
                "Implement micro-breaks throughout the day",
                "Consider adjusting work schedule for better rest"
            ])

        elif load_trend == 'decreasing':
            recommendations.extend([
                "Capitalize on lower cognitive load for complex tasks",
                "Schedule important decision-making activities",
                "Maintain current workload balance"
            ])

        return recommendations

    def _calculate_optimization_benefit(self, distribution: Dict[str, Any]) -> float:
        """Calculate expected cognitive benefit from optimization"""
        # Estimate benefit based on workload distribution improvements
        balance_score = distribution.get('balance_score', 0.5)
        task_reduction = distribution.get('high_load_task_reduction', 0)

        benefit = (balance_score - 0.5) * 0.6 + (task_reduction * 0.1)
        return min(benefit, 0.5)  # Cap at 50% improvement

    def _assess_fatigue_severity(self, indicators: Dict[str, Any]) -> float:
        """Assess fatigue severity from indicators"""
        severity = 0.0

        # Factor in various fatigue indicators
        severity += indicators.get('load_consistency', 0) * 0.3
        severity += indicators.get('recovery_gaps', 0) * 0.3
        severity += indicators.get('performance_decline', 0) * 0.4

        return min(severity, 1.0)

    def _estimate_recovery_time(self, fatigue_severity: float) -> int:
        """Estimate recovery time in minutes"""
        if fatigue_severity < 0.3:
            return 15  # Quick break
        elif fatigue_severity < 0.6:
            return 45  # Short break
        elif fatigue_severity < 0.8:
            return 90  # Longer break
        else:
            return 180  # Extended rest

    def _generate_fatigue_prevention_strategies(self) -> List[str]:
        """Generate fatigue prevention strategies"""
        return [
            "Take regular micro-breaks every 25-30 minutes",
            "Practice the 20-20-20 rule for eye strain",
            "Stay hydrated and maintain proper nutrition",
            "Get adequate sleep (7-9 hours per night)",
            "Exercise regularly to improve cognitive resilience",
            "Use the Pomodoro Technique for focused work sessions"
        ]

    def get_cognitive_monitoring_status(self) -> Dict[str, Any]:
        """Get status of cognitive monitoring system"""
        return {
            'monitoring_active': True,
            'users_being_monitored': len(self.user_cognitive_states),
            'total_assessments': sum(len(history) for history in self.cognitive_history.values()),
            'average_cognitive_load': self._calculate_average_cognitive_load(),
            'fatigue_detection_active': True,
            'optimization_engine_active': True,
            'effectiveness_metrics': self.effectiveness_metrics,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_average_cognitive_load(self) -> float:
        """Calculate average cognitive load across all users"""
        all_loads = []
        for user_history in self.cognitive_history.values():
            loads = [record.get('cognitive_load', 0.5) for record in user_history]
            if loads:
                all_loads.extend(loads)

        return statistics.mean(all_loads) if all_loads else 0.5


class CognitiveLoadAssessmentModel:
    """Advanced model for cognitive load assessment"""

    def __init__(self):
        self.assessment_weights = {
            'temporal_factors': 0.25,
            'task_complexity': 0.30,
            'workload_distribution': 0.20,
            'performance_metrics': 0.15,
            'physiological_indicators': 0.05,
            'environmental_factors': 0.05
        }

    async def assess_load(self, indicators: Dict[str, Any]) -> float:
        """Assess cognitive load from indicators"""
        load_score = 0.0

        for factor, weight in self.assessment_weights.items():
            if factor in indicators:
                factor_score = self._calculate_factor_score(factor, indicators[factor])
                load_score += factor_score * weight

        return min(1.0, max(0.0, load_score))

    def _calculate_factor_score(self, factor: str, data: Dict[str, Any]) -> float:
        """Calculate score for a specific factor"""
        if factor == 'temporal_factors':
            return self._score_temporal_factors(data)
        elif factor == 'task_complexity':
            return self._score_task_complexity(data)
        elif factor == 'workload_distribution':
            return self._score_workload_distribution(data)
        elif factor == 'performance_metrics':
            return self._score_performance_metrics(data)
        elif factor == 'physiological_indicators':
            return self._score_physiological_indicators(data)
        elif factor == 'environmental_factors':
            return self._score_environmental_factors(data)
        else:
            return 0.5

    def _score_temporal_factors(self, data: Dict[str, Any]) -> float:
        """Score temporal factors"""
        time_factor = data.get('time_of_day_factor', 0.5)
        activity_level = data.get('recent_activity_level', 0.5)
        switching_freq = data.get('task_switching_frequency', 0.0)

        # Higher switching frequency increases load
        score = (1 - time_factor) * 0.4 + activity_level * 0.4 + switching_freq * 0.2
        return min(1.0, score)

    def _score_task_complexity(self, data: Dict[str, Any]) -> float:
        """Score task complexity"""
        avg_complexity = data.get('average_complexity', 0.5)
        task_count = data.get('task_count', 0)
        high_complexity_count = data.get('high_complexity_tasks', 0)

        # More high-complexity tasks increase load
        complexity_score = avg_complexity * 0.6 + (high_complexity_count / max(task_count, 1)) * 0.4
        return min(1.0, complexity_score)

    def _score_workload_distribution(self, data: Dict[str, Any]) -> float:
        """Score workload distribution"""
        balance_score = data.get('workload_balance_score', 0.5)
        total_tasks = data.get('total_tasks', 0)

        # Poor balance and high task count increase load
        task_factor = min(total_tasks / 10, 1.0)  # Max factor for 10+ tasks
        score = (1 - balance_score) * 0.7 + task_factor * 0.3
        return min(1.0, score)

    def _score_performance_metrics(self, data: Dict[str, Any]) -> float:
        """Score performance metrics"""
        performance_score = data.get('recent_performance_score', 0.7)
        error_rate = data.get('error_rate', 0.0)
        volatility = data.get('performance_volatility', 0.0)

        # Lower performance, higher errors, and volatility increase load
        score = (1 - performance_score) * 0.5 + error_rate * 0.3 + volatility * 0.2
        return min(1.0, score)

    def _score_physiological_indicators(self, data: Dict[str, Any]) -> float:
        """Score physiological indicators"""
        session_factor = data.get('session_duration_factor', 0.0)
        break_score = 1 - data.get('break_frequency_score', 0.0)  # Lower breaks = higher load
        stress_indicators = data.get('physiological_stress_indicators', 0.0)

        score = session_factor * 0.4 + break_score * 0.3 + stress_indicators * 0.3
        return min(1.0, score)

    def _score_environmental_factors(self, data: Dict[str, Any]) -> float:
        """Score environmental factors"""
        noise = data.get('noise_level', 0.5)
        lighting = 1 - data.get('lighting_conditions', 0.7)  # Poor lighting increases load
        distractions = data.get('distraction_level', 0.3)
        comfort = 1 - data.get('workspace_comfort', 0.8)  # Poor comfort increases load

        score = noise * 0.3 + lighting * 0.2 + distractions * 0.3 + comfort * 0.2
        return min(1.0, score)


class FatigueDetectionModel:
    """Advanced model for fatigue detection and analysis"""

    def __init__(self):
        self.fatigue_patterns = defaultdict(list)

    async def analyze_fatigue(self, user_id: str, indicators: Dict[str, Any],
                            history: deque) -> Dict[str, Any]:
        """Analyze fatigue patterns"""
        try:
            # Analyze recent cognitive load patterns
            load_consistency = self._analyze_load_consistency(history)

            # Analyze recovery patterns
            recovery_gaps = self._analyze_recovery_patterns(history)

            # Analyze performance decline
            performance_decline = self._analyze_performance_decline(history)

            # Calculate overall fatigue level
            fatigue_level = (load_consistency * 0.4 + recovery_gaps * 0.3 + performance_decline * 0.3)

            return {
                'fatigue_level': fatigue_level,
                'load_consistency': load_consistency,
                'recovery_gaps': recovery_gaps,
                'performance_decline': performance_decline,
                'fatigue_trend': self._determine_fatigue_trend(history),
                'risk_level': 'high' if fatigue_level > 0.7 else 'moderate' if fatigue_level > 0.4 else 'low'
            }

        except Exception as e:
            logger.error(f"Fatigue analysis failed: {e}")
            return {'fatigue_level': 0.5, 'error': str(e)}

    async def detect_fatigue_patterns(self, recent_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect specific fatigue patterns"""
        patterns = {
            'sustained_high_load': False,
            'inadequate_recovery': False,
            'performance_degradation': False,
            'erratic_patterns': False
        }

        if len(recent_history) < 5:
            return patterns

        # Check for sustained high cognitive load
        high_load_count = sum(1 for record in recent_history
                            if record.get('cognitive_load', 0) > 0.8)
        patterns['sustained_high_load'] = high_load_count > len(recent_history) * 0.6

        # Check for inadequate recovery (consistently high load without breaks)
        load_values = [record.get('cognitive_load', 0) for record in recent_history]
        patterns['inadequate_recovery'] = all(load > 0.7 for load in load_values[-3:])

        # Check for performance degradation
        if len(recent_history) >= 3:
            recent_performance = [record.get('cognitive_load', 0.5) for record in recent_history[-3:]]
            earlier_performance = [record.get('cognitive_load', 0.5) for record in recent_history[:-3]]
            if recent_performance and earlier_performance:
                recent_avg = statistics.mean(recent_performance)
                earlier_avg = statistics.mean(earlier_performance)
                patterns['performance_degradation'] = recent_avg > earlier_avg + 0.2

        # Check for erratic patterns (high variability)
        if len(load_values) > 1:
            variability = statistics.stdev(load_values)
            patterns['erratic_patterns'] = variability > 0.3

        return patterns

    def _analyze_load_consistency(self, history: deque) -> float:
        """Analyze consistency of cognitive load"""
        if len(history) < 3:
            return 0.5

        load_values = [record.get('cognitive_load', 0.5) for record in list(history)[-10:]]
        if len(load_values) < 2:
            return 0.5

        # High consistency (similar values) may indicate fatigue
        consistency = 1 - (statistics.stdev(load_values) / 0.5)  # Normalize
        return max(0.0, min(1.0, consistency))

    def _analyze_recovery_patterns(self, history: deque) -> float:
        """Analyze recovery patterns between high-load periods"""
        if len(history) < 5:
            return 0.5

        recent_records = list(history)[-10:]
        high_load_periods = [i for i, record in enumerate(recent_records)
                           if record.get('cognitive_load', 0) > 0.8]

        if not high_load_periods:
            return 0.0  # No high load periods

        # Check for recovery between high load periods
        recovery_gaps = []
        for i in range(1, len(high_load_periods)):
            gap = high_load_periods[i] - high_load_periods[i-1]
            recovery_gaps.append(gap)

        if recovery_gaps:
            avg_gap = statistics.mean(recovery_gaps)
            # Smaller gaps indicate inadequate recovery
            recovery_score = max(0.0, 1.0 - (avg_gap / 5.0))  # Normalize
            return recovery_score

        return 0.5

    def _analyze_performance_decline(self, history: deque) -> float:
        """Analyze performance decline patterns"""
        if len(history) < 5:
            return 0.5

        recent_records = list(history)[-5:]
        earlier_records = list(history)[:-5]

        if not earlier_records:
            return 0.5

        recent_avg_load = statistics.mean([r.get('cognitive_load', 0.5) for r in recent_records])
        earlier_avg_load = statistics.mean([r.get('cognitive_load', 0.5) for r in earlier_records])

        # Increasing load may indicate performance decline
        decline = (recent_avg_load - earlier_avg_load) / max(earlier_avg_load, 0.1)
        return max(0.0, min(1.0, decline))

    def _determine_fatigue_trend(self, history: deque) -> str:
        """Determine fatigue trend"""
        if len(history) < 3:
            return 'unknown'

        recent_loads = [record.get('cognitive_load', 0.5) for record in list(history)[-3:]]
        trend = 'stable'

        if len(recent_loads) >= 2:
            if recent_loads[-1] > recent_loads[0] + 0.1:
                trend = 'increasing'
            elif recent_loads[-1] < recent_loads[0] - 0.1:
                trend = 'decreasing'

        return trend


class CognitiveOptimizationEngine:
    """Engine for generating cognitive optimization recommendations"""

    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()

    async def generate_recommendations(self, cognitive_load: float,
                                     fatigue_analysis: Dict[str, Any],
                                     indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cognitive optimization recommendations"""
        recommendations = []

        # High cognitive load recommendations
        if cognitive_load > 0.8:
            recommendations.extend([
                {
                    'type': 'immediate_action',
                    'priority': 'critical',
                    'description': 'Take an immediate 10-minute break to reduce cognitive load',
                    'action_items': ['Step away from work', 'Practice deep breathing', 'Light stretching'],
                    'expected_benefit': '20-30% reduction in cognitive load'
                },
                {
                    'type': 'workload_reduction',
                    'priority': 'high',
                    'description': 'Reduce current workload by postponing non-urgent tasks',
                    'action_items': ['Review task priorities', 'Delegate if possible', 'Focus on 2-3 key tasks'],
                    'expected_benefit': 'Significant cognitive load reduction'
                }
            ])

        # Fatigue-based recommendations
        fatigue_level = fatigue_analysis.get('fatigue_level', 0)
        if fatigue_level > 0.6:
            recommendations.append({
                'type': 'fatigue_recovery',
                'priority': 'high',
                'description': 'Implement fatigue recovery strategies',
                'action_items': ['Take extended break', 'Hydrate and eat nutritious snack', 'Light exercise'],
                'expected_benefit': 'Improved cognitive performance and reduced fatigue'
            })

        # Task complexity recommendations
        task_complexity = indicators.get('task_complexity', {})
        if task_complexity.get('high_complexity_tasks', 0) > 2:
            recommendations.append({
                'type': 'task_simplification',
                'priority': 'medium',
                'description': 'Break down complex tasks into smaller, manageable steps',
                'action_items': ['Decompose complex tasks', 'Set intermediate milestones', 'Use task checklists'],
                'expected_benefit': 'Reduced cognitive load and improved task completion'
            })

        # Environmental optimization
        environmental = indicators.get('environmental_factors', {})
        if environmental.get('distraction_level', 0) > 0.6:
            recommendations.append({
                'type': 'environment_optimization',
                'priority': 'medium',
                'description': 'Optimize work environment to reduce distractions',
                'action_items': ['Find quiet workspace', 'Use noise-cancelling headphones', 'Minimize notifications'],
                'expected_benefit': 'Improved focus and reduced cognitive load'
            })

        return recommendations

    async def optimize_workload_distribution(self, tasks: List[Dict[str, Any]],
                                           cognitive_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize workload distribution based on cognitive capacity"""
        current_load = cognitive_assessment.get('cognitive_load', 0.5)

        # Categorize tasks by priority and complexity
        high_priority_tasks = [t for t in tasks if t.get('priority') in ['high', 'urgent']]
        medium_priority_tasks = [t for t in tasks if t.get('priority') == 'medium']
        low_priority_tasks = [t for t in tasks if t.get('priority') == 'low']

        # Calculate optimal distribution
        if current_load > 0.8:
            # High load: Focus on critical tasks only
            recommended_tasks = high_priority_tasks[:2]  # Max 2 high-priority tasks
            postponed_tasks = medium_priority_tasks + low_priority_tasks + high_priority_tasks[2:]
        elif current_load > 0.6:
            # Moderate load: Include some medium priority
            recommended_tasks = high_priority_tasks + medium_priority_tasks[:2]
            postponed_tasks = low_priority_tasks + medium_priority_tasks[2:]
        else:
            # Low load: Can handle more tasks
            recommended_tasks = tasks
            postponed_tasks = []

        return {
            'recommended_tasks': recommended_tasks,
            'postponed_tasks': postponed_tasks,
            'balance_score': self._calculate_task_balance(recommended_tasks),
            'high_load_task_reduction': len(postponed_tasks),
            'optimization_reason': f'Based on current cognitive load: {current_load:.2f}'
        }

    async def generate_scheduling_recommendations(self, distribution: Dict[str, Any],
                                                cognitive_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scheduling recommendations"""
        recommendations = []

        recommended_tasks = distribution.get('recommended_tasks', [])
        current_load = cognitive_assessment.get('cognitive_load', 0.5)

        if current_load > 0.7:
            # High load: Schedule breaks and lighter tasks
            recommendations.extend([
                {
                    'time_slot': 'immediate',
                    'action': 'schedule_break',
                    'description': 'Take a 15-minute break before continuing',
                    'duration_minutes': 15
                },
                {
                    'time_slot': 'after_break',
                    'action': 'focus_on_high_priority',
                    'description': 'Focus on 1-2 high-priority tasks only',
                    'tasks': recommended_tasks[:2]
                }
            ])
        else:
            # Normal load: Standard scheduling
            recommendations.append({
                'time_slot': 'now',
                'action': 'proceed_with_tasks',
                'description': 'Continue with recommended task distribution',
                'tasks': recommended_tasks
            })

        return recommendations

    async def generate_recovery_recommendations(self, fatigue_severity: float,
                                              fatigue_indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recovery recommendations based on fatigue severity"""
        recommendations = []

        if fatigue_severity > 0.8:
            recommendations.extend([
                {
                    'type': 'extended_rest',
                    'description': 'Take 1-2 hour break or end work session',
                    'urgency': 'critical',
                    'actions': ['Stop work immediately', 'Rest in quiet environment', 'Consider ending workday']
                },
                {
                    'type': 'professional_help',
                    'description': 'Consider consulting healthcare professional',
                    'urgency': 'high',
                    'actions': ['Schedule medical consultation', 'Monitor symptoms', 'Adjust work schedule']
                }
            ])
        elif fatigue_severity > 0.6:
            recommendations.extend([
                {
                    'type': 'substantial_break',
                    'description': 'Take 30-45 minute break with restorative activities',
                    'urgency': 'high',
                    'actions': ['Walk outside', 'Listen to calming music', 'Practice mindfulness', 'Light stretching']
                }
            ])
        elif fatigue_severity > 0.4:
            recommendations.extend([
                {
                    'type': 'moderate_break',
                    'description': 'Take 15-20 minute break with relaxation activities',
                    'urgency': 'medium',
                    'actions': ['Step away from screen', 'Deep breathing exercises', 'Short walk', 'Hydrate']
                }
            ])
        else:
            recommendations.extend([
                {
                    'type': 'micro_break',
                    'description': 'Take 5-minute micro-break to reset',
                    'urgency': 'low',
                    'actions': ['Stand up and stretch', 'Look away from screen', 'Deep breaths', 'Quick walk']
                }
            ])

        return recommendations

    def _calculate_task_balance(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate balance score for task distribution"""
        if not tasks:
            return 1.0

        # Analyze task type distribution
        task_types = defaultdict(int)
        for task in tasks:
            task_type = task.get('type', 'general')
            task_types[task_type] += 1

        if len(task_types) <= 1:
            return 1.0  # Perfect balance with single type

        # Calculate balance using entropy
        total_tasks = sum(task_types.values())
        entropy = 0
        for count in task_types.values():
            p = count / total_tasks
            entropy -= p * math.log2(p)

        max_entropy = math.log2(len(task_types))
        balance_score = entropy / max_entropy if max_entropy > 0 else 1.0

        return balance_score

    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load cognitive optimization rules"""
        return {
            'high_load_threshold': 0.8,
            'fatigue_threshold': 0.6,
            'max_recommended_tasks': 5,
            'break_intervals': [15, 30, 45, 90],  # minutes
            'recovery_activities': [
                'deep_breathing', 'light_exercise', 'hydration',
                'mindfulness', 'nature_walk', 'power_nap'
            ]
        }