"""CES AI Assistant Specialization.

Provides specialized capabilities and expertise-based task routing for different AI assistants.
Implements advanced specialization features including domain expertise, performance optimization,
and adaptive assistant selection based on task requirements and historical performance.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

from ..core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class AssistantProfile:
    """Profile for an AI assistant with specialization details."""
    name: str
    display_name: str
    primary_domain: str
    expertise_level: int  # 1-10 scale
    specializations: List[str]
    performance_metrics: Dict[str, float]
    task_history: List[Dict[str, Any]]
    strengths: List[str]
    limitations: List[str]
    cost_efficiency: float
    response_time_avg: float
    accuracy_rate: float
    last_updated: str

@dataclass
class TaskProfile:
    """Profile for a task with requirements analysis."""
    description: str
    domain: str
    complexity: int  # 1-10 scale
    required_skills: List[str]
    estimated_duration: int  # minutes
    priority: str
    context_requirements: List[str]
    quality_requirements: Dict[str, Any]

class AISpecializationManager:
    """Manages AI assistant specialization and intelligent task routing."""

    def __init__(self):
        self.assistant_profiles = {}
        self.task_profiles = []
        self.performance_history = defaultdict(list)
        self.domain_expertise = defaultdict(dict)
        self.specialization_rules = self._load_specialization_rules()
        self._initialize_assistant_profiles()

    def _initialize_assistant_profiles(self):
        """Initialize default assistant profiles with specialization data."""
        self.assistant_profiles = {
            'grok': AssistantProfile(
                name='grok',
                display_name='Grok (xAI)',
                primary_domain='general_reasoning',
                expertise_level=9,
                specializations=[
                    'logical_reasoning', 'creative_problem_solving', 'general_knowledge',
                    'ethical_analysis', 'mathematical_reasoning', 'scientific_explanation'
                ],
                performance_metrics={
                    'response_time_avg': 450,  # ms
                    'accuracy_rate': 0.92,
                    'cost_efficiency': 0.85,
                    'success_rate': 0.95
                },
                task_history=[],
                strengths=[
                    'Excellent reasoning and analysis',
                    'Creative problem solving',
                    'Broad knowledge base',
                    'Ethical considerations',
                    'Clear explanations'
                ],
                limitations=[
                    'May be verbose for simple tasks',
                    'Less specialized in narrow technical domains'
                ],
                cost_efficiency=0.85,
                response_time_avg=450,
                accuracy_rate=0.92,
                last_updated=datetime.now().isoformat()
            ),

            'qwen-cli-coder': AssistantProfile(
                name='qwen-cli-coder',
                display_name='Qwen Coder',
                primary_domain='code_generation',
                expertise_level=10,
                specializations=[
                    'code_generation', 'code_review', 'debugging', 'refactoring',
                    'algorithm_implementation', 'api_design', 'testing'
                ],
                performance_metrics={
                    'response_time_avg': 380,
                    'accuracy_rate': 0.94,
                    'cost_efficiency': 0.90,
                    'success_rate': 0.97
                },
                task_history=[],
                strengths=[
                    'Excellent code generation',
                    'Strong debugging capabilities',
                    'Multiple programming languages',
                    'Clean, efficient code output',
                    'Good understanding of best practices'
                ],
                limitations=[
                    'May need clarification for complex requirements',
                    'Limited general reasoning outside coding'
                ],
                cost_efficiency=0.90,
                response_time_avg=380,
                accuracy_rate=0.94,
                last_updated=datetime.now().isoformat()
            ),

            'gemini-cli': AssistantProfile(
                name='gemini-cli',
                display_name='Gemini CLI',
                primary_domain='code_analysis',
                expertise_level=9,
                specializations=[
                    'code_analysis', 'documentation', 'code_explanation',
                    'architecture_review', 'security_analysis', 'performance_optimization'
                ],
                performance_metrics={
                    'response_time_avg': 420,
                    'accuracy_rate': 0.91,
                    'cost_efficiency': 0.80,
                    'success_rate': 0.93
                },
                task_history=[],
                strengths=[
                    'Excellent code analysis and review',
                    'Strong documentation capabilities',
                    'Good at explaining complex code',
                    'Security and performance insights',
                    'Clear technical writing'
                ],
                limitations=[
                    'May be slower for simple tasks',
                    'Less creative in code generation'
                ],
                cost_efficiency=0.80,
                response_time_avg=420,
                accuracy_rate=0.91,
                last_updated=datetime.now().isoformat()
            )
        }

        logger.info("Initialized AI assistant specialization profiles")

    def _load_specialization_rules(self) -> Dict[str, Any]:
        """Load specialization rules for task routing."""
        return {
            'domain_mapping': {
                'coding': ['qwen-cli-coder', 'grok'],
                'debugging': ['qwen-cli-coder', 'gemini-cli'],
                'analysis': ['gemini-cli', 'grok'],
                'documentation': ['gemini-cli', 'grok'],
                'reasoning': ['grok', 'gemini-cli'],
                'design': ['grok', 'qwen-cli-coder'],
                'testing': ['qwen-cli-coder', 'gemini-cli'],
                'optimization': ['gemini-cli', 'qwen-cli-coder']
            },
            'complexity_thresholds': {
                'low': {'max_complexity': 3, 'preferred_assistants': ['grok']},
                'medium': {'max_complexity': 6, 'preferred_assistants': ['qwen-cli-coder', 'gemini-cli']},
                'high': {'max_complexity': 10, 'preferred_assistants': ['qwen-cli-coder', 'gemini-cli', 'grok']}
            },
            'performance_weights': {
                'accuracy': 0.4,
                'speed': 0.3,
                'cost_efficiency': 0.2,
                'expertise_match': 0.1
            }
        }

    async def analyze_task_requirements(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> TaskProfile:
        """Analyze task requirements and create a task profile."""
        # Extract domain from task description
        domain = self._classify_task_domain(task_description)

        # Estimate complexity
        complexity = self._estimate_complexity(task_description, context)

        # Identify required skills
        required_skills = self._extract_required_skills(task_description, domain)

        # Estimate duration
        estimated_duration = self._estimate_duration(complexity, required_skills)

        # Determine priority
        priority = self._determine_priority(task_description, context)

        # Define context requirements
        context_requirements = self._identify_context_needs(task_description, domain)

        # Define quality requirements
        quality_requirements = self._define_quality_requirements(domain, complexity)

        task_profile = TaskProfile(
            description=task_description,
            domain=domain,
            complexity=complexity,
            required_skills=required_skills,
            estimated_duration=estimated_duration,
            priority=priority,
            context_requirements=context_requirements,
            quality_requirements=quality_requirements
        )

        self.task_profiles.append(task_profile)
        logger.info(f"Analyzed task requirements: domain={domain}, complexity={complexity}")

        return task_profile

    def _classify_task_domain(self, task_description: str) -> str:
        """Classify task into a domain category."""
        description_lower = task_description.lower()

        domain_keywords = {
            'coding': ['code', 'program', 'function', 'class', 'implement', 'develop', 'build'],
            'debugging': ['debug', 'fix', 'error', 'bug', 'issue', 'problem', 'resolve'],
            'analysis': ['analyze', 'review', 'examine', 'assess', 'evaluate', 'study'],
            'documentation': ['document', 'docstring', 'comment', 'explain', 'describe'],
            'reasoning': ['reason', 'logic', 'think', 'decide', 'choose', 'plan'],
            'design': ['design', 'architecture', 'structure', 'plan', 'organize'],
            'testing': ['test', 'unit test', 'integration', 'validate', 'verify'],
            'optimization': ['optimize', 'performance', 'efficiency', 'speed', 'improve']
        }

        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return 'general'

    def _estimate_complexity(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> int:
        """Estimate task complexity on a 1-10 scale."""
        complexity = 5  # Default medium complexity

        # Length-based complexity
        if len(task_description) > 500:
            complexity += 2
        elif len(task_description) < 100:
            complexity -= 1

        # Keyword-based complexity
        complex_keywords = ['complex', 'advanced', 'sophisticated', 'multi-step', 'integrate']
        simple_keywords = ['simple', 'basic', 'straightforward', 'quick']

        description_lower = task_description.lower()
        complex_count = sum(1 for keyword in complex_keywords if keyword in description_lower)
        simple_count = sum(1 for keyword in simple_keywords if keyword in description_lower)

        complexity += complex_count - simple_count

        # Context-based adjustments
        if context:
            if context.get('has_dependencies'):
                complexity += 1
            if context.get('time_pressure'):
                complexity += 1
            if context.get('requires_research'):
                complexity += 2

        return max(1, min(10, complexity))

    def _extract_required_skills(self, task_description: str, domain: str) -> List[str]:
        """Extract required skills from task description."""
        skills_mapping = {
            'coding': ['programming', 'algorithm_design', 'code_structure'],
            'debugging': ['problem_solving', 'code_analysis', 'testing'],
            'analysis': ['critical_thinking', 'data_analysis', 'evaluation'],
            'documentation': ['technical_writing', 'explanation', 'clarity'],
            'reasoning': ['logical_thinking', 'decision_making', 'planning'],
            'design': ['system_design', 'architecture', 'organization'],
            'testing': ['test_design', 'validation', 'quality_assurance'],
            'optimization': ['performance_analysis', 'efficiency', 'optimization']
        }

        return skills_mapping.get(domain, ['general_problem_solving'])

    def _estimate_duration(self, complexity: int, skills: List[str]) -> int:
        """Estimate task duration in minutes."""
        base_duration = 30  # Base 30 minutes

        # Complexity multiplier
        complexity_multiplier = 1 + (complexity - 5) * 0.2

        # Skills multiplier
        skills_multiplier = 1 + len(skills) * 0.1

        estimated = base_duration * complexity_multiplier * skills_multiplier

        return max(15, min(480, int(estimated)))  # Between 15 minutes and 8 hours

    def _determine_priority(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Determine task priority."""
        description_lower = task_description.lower()

        # High priority keywords
        high_priority_keywords = ['urgent', 'critical', 'emergency', 'asap', 'deadline']
        if any(keyword in description_lower for keyword in high_priority_keywords):
            return 'high'

        # Medium priority keywords
        medium_priority_keywords = ['important', 'priority', 'soon', 'needed']
        if any(keyword in description_lower for keyword in medium_priority_keywords):
            return 'medium'

        return 'low'

    def _identify_context_needs(self, task_description: str, domain: str) -> List[str]:
        """Identify context requirements for the task."""
        context_needs = []

        if domain in ['coding', 'debugging']:
            context_needs.extend(['codebase_access', 'programming_language'])

        if domain == 'analysis':
            context_needs.extend(['data_access', 'analysis_tools'])

        if 'existing' in task_description.lower():
            context_needs.append('existing_solution_review')

        if 'integrate' in task_description.lower():
            context_needs.append('integration_requirements')

        return context_needs

    def _define_quality_requirements(self, domain: str, complexity: int) -> Dict[str, Any]:
        """Define quality requirements based on domain and complexity."""
        base_requirements = {
            'accuracy': 0.8,
            'completeness': 0.8,
            'clarity': 0.8
        }

        # Domain-specific adjustments
        if domain == 'coding':
            base_requirements.update({
                'syntax_correctness': 1.0,
                'best_practices': 0.9,
                'testability': 0.8
            })

        if domain == 'analysis':
            base_requirements.update({
                'thoroughness': 0.9,
                'objectivity': 0.9,
                'actionability': 0.8
            })

        # Complexity adjustments
        if complexity > 7:
            for key in base_requirements:
                base_requirements[key] = min(1.0, base_requirements[key] + 0.1)

        return base_requirements

    async def recommend_assistants(self, task_profile: TaskProfile) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Recommend AI assistants for a task with confidence scores."""
        recommendations = []

        for assistant_name, profile in self.assistant_profiles.items():
            # Calculate expertise match
            expertise_score = self._calculate_expertise_match(profile, task_profile)

            # Calculate performance score
            performance_score = self._calculate_performance_score(profile, task_profile)

            # Calculate domain suitability
            domain_score = self._calculate_domain_suitability(profile, task_profile)

            # Calculate overall confidence
            weights = self.specialization_rules['performance_weights']
            confidence = (
                expertise_score * weights['expertise_match'] +
                performance_score * weights['accuracy'] +
                domain_score * weights['speed'] +
                profile.cost_efficiency * weights['cost_efficiency']
            )

            # Additional factors
            factors = {
                'expertise_match': expertise_score,
                'performance_score': performance_score,
                'domain_suitability': domain_score,
                'cost_efficiency': profile.cost_efficiency,
                'estimated_response_time': profile.response_time_avg,
                'strengths': profile.strengths[:3],  # Top 3 strengths
                'limitations': profile.limitations[:2]  # Top 2 limitations
            }

            recommendations.append((assistant_name, confidence, factors))

        # Sort by confidence score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Recommended assistants for task: {[r[0] for r in recommendations[:3]]}")
        return recommendations

    def _calculate_expertise_match(self, profile: AssistantProfile, task_profile: TaskProfile) -> float:
        """Calculate how well the assistant's expertise matches the task."""
        skill_matches = 0
        for skill in task_profile.required_skills:
            if skill in profile.specializations:
                skill_matches += 1

        expertise_ratio = skill_matches / len(task_profile.required_skills) if task_profile.required_skills else 0

        # Factor in expertise level
        expertise_factor = profile.expertise_level / 10.0

        return (expertise_ratio + expertise_factor) / 2

    def _calculate_performance_score(self, profile: AssistantProfile, task_profile: TaskProfile) -> float:
        """Calculate performance score based on historical data."""
        # Use current performance metrics as base
        base_score = (
            profile.accuracy_rate * 0.6 +
            (1 - profile.response_time_avg / 2000) * 0.4  # Normalize response time
        )

        # Adjust for task complexity
        complexity_factor = 1.0
        if task_profile.complexity > 7 and profile.expertise_level >= 8:
            complexity_factor = 1.2
        elif task_profile.complexity < 4 and profile.expertise_level >= 6:
            complexity_factor = 1.1

        return min(1.0, base_score * complexity_factor)

    def _calculate_domain_suitability(self, profile: AssistantProfile, task_profile: TaskProfile) -> float:
        """Calculate how suitable the assistant is for the task domain."""
        if profile.primary_domain == task_profile.domain:
            return 1.0

        # Check if domain is in assistant's specializations
        domain_keywords = {
            'coding': ['code', 'programming', 'development'],
            'debugging': ['debug', 'fix', 'error'],
            'analysis': ['analyze', 'review', 'study'],
            'documentation': ['document', 'explain', 'describe']
        }

        domain_terms = domain_keywords.get(task_profile.domain, [task_profile.domain])
        specialization_match = any(
            any(term in spec.lower() for term in domain_terms)
            for spec in profile.specializations
        )

        return 0.8 if specialization_match else 0.4

    async def update_performance_metrics(self, assistant_name: str, task_result: Dict[str, Any]):
        """Update assistant performance metrics based on task results."""
        if assistant_name not in self.assistant_profiles:
            return

        profile = self.assistant_profiles[assistant_name]

        # Add to task history
        task_record = {
            'timestamp': datetime.now().isoformat(),
            'task_description': task_result.get('task_description', ''),
            'success': task_result.get('success', False),
            'response_time': task_result.get('response_time', 0),
            'quality_score': task_result.get('quality_score', 0.8)
        }

        profile.task_history.append(task_record)

        # Keep only last 100 tasks
        if len(profile.task_history) > 100:
            profile.task_history = profile.task_history[-100:]

        # Recalculate metrics
        recent_tasks = profile.task_history[-20:]  # Last 20 tasks

        if recent_tasks:
            success_rate = sum(1 for task in recent_tasks if task['success']) / len(recent_tasks)
            avg_response_time = sum(task['response_time'] for task in recent_tasks) / len(recent_tasks)
            avg_quality = sum(task['quality_score'] for task in recent_tasks) / len(recent_tasks)

            # Update profile metrics
            profile.accuracy_rate = (profile.accuracy_rate + success_rate) / 2
            profile.response_time_avg = (profile.response_time_avg + avg_response_time) / 2
            profile.performance_metrics['quality_score'] = avg_quality

        profile.last_updated = datetime.now().isoformat()

        logger.info(f"Updated performance metrics for {assistant_name}")

    def get_specialization_report(self) -> Dict[str, Any]:
        """Generate a comprehensive specialization report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'assistants': {},
            'domain_coverage': {},
            'performance_summary': {},
            'recommendations': []
        }

        # Assistant details
        for name, profile in self.assistant_profiles.items():
            report['assistants'][name] = {
                'display_name': profile.display_name,
                'primary_domain': profile.primary_domain,
                'expertise_level': profile.expertise_level,
                'specializations': profile.specializations,
                'performance_metrics': profile.performance_metrics,
                'task_count': len(profile.task_history),
                'strengths': profile.strengths,
                'limitations': profile.limitations
            }

        # Domain coverage analysis
        all_domains = set()
        for profile in self.assistant_profiles.values():
            all_domains.add(profile.primary_domain)
            all_domains.update(profile.specializations)

        for domain in all_domains:
            coverage = []
            for name, profile in self.assistant_profiles.items():
                if domain == profile.primary_domain:
                    coverage.append({'assistant': name, 'level': 'primary', 'expertise': profile.expertise_level})
                elif domain in profile.specializations:
                    coverage.append({'assistant': name, 'level': 'secondary', 'expertise': profile.expertise_level})

            report['domain_coverage'][domain] = coverage

        # Performance summary
        report['performance_summary'] = {
            'total_assistants': len(self.assistant_profiles),
            'avg_expertise_level': sum(p.expertise_level for p in self.assistant_profiles.values()) / len(self.assistant_profiles),
            'total_tasks_processed': sum(len(p.task_history) for p in self.assistant_profiles.values()),
            'best_performer': max(self.assistant_profiles.keys(),
                                key=lambda x: self.assistant_profiles[x].accuracy_rate)
        }

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving specialization."""
        recommendations = []

        # Check for domain gaps
        covered_domains = set()
        for profile in self.assistant_profiles.values():
            covered_domains.add(profile.primary_domain)
            covered_domains.update(profile.specializations)

        common_domains = {'security', 'database', 'web_development', 'mobile_development', 'data_science'}
        missing_domains = common_domains - covered_domains

        if missing_domains:
            recommendations.append(f"Consider adding assistants specialized in: {', '.join(missing_domains)}")

        # Check performance balance
        performance_scores = {name: profile.accuracy_rate for name, profile in self.assistant_profiles.items()}
        if max(performance_scores.values()) - min(performance_scores.values()) > 0.2:
            recommendations.append("Consider balancing assistant performance through targeted training or optimization")

        # Check task distribution
        task_counts = {name: len(profile.task_history) for name, profile in self.assistant_profiles.items()}
        if max(task_counts.values()) / max(min(task_counts.values()), 1) > 3:
            recommendations.append("Task distribution is uneven - consider load balancing improvements")

        return recommendations

    def get_assistant_status(self, assistant_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific assistant."""
        if assistant_name not in self.assistant_profiles:
            return None

        profile = self.assistant_profiles[assistant_name]

        # Calculate recent performance (last 10 tasks)
        recent_tasks = profile.task_history[-10:]
        recent_performance = {}
        if recent_tasks:
            recent_performance = {
                'success_rate': sum(1 for task in recent_tasks if task['success']) / len(recent_tasks),
                'avg_response_time': sum(task['response_time'] for task in recent_tasks) / len(recent_tasks),
                'avg_quality_score': sum(task.get('quality_score', 0.8) for task in recent_tasks) / len(recent_tasks)
            }

        return {
            'name': profile.name,
            'display_name': profile.display_name,
            'status': 'active',
            'primary_domain': profile.primary_domain,
            'expertise_level': profile.expertise_level,
            'current_performance': profile.performance_metrics,
            'recent_performance': recent_performance,
            'specializations': profile.specializations,
            'task_count': len(profile.task_history),
            'last_updated': profile.last_updated,
            'strengths': profile.strengths,
            'limitations': profile.limitations
        }

    def is_healthy(self) -> bool:
        """Check if specialization manager is healthy."""
        return len(self.assistant_profiles) > 0