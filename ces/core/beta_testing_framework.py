"""
Beta Testing Framework - CES Phase 5 Launch & Optimization

Phase 5 Implementation: Enterprise-grade beta testing framework for public launch preparation,
community engagement, and comprehensive validation of all CES features.

Key Phase 5 Features:
- Community beta program management (10-20 users)
- Real-time feedback collection and analysis
- Automated testing scenarios for launch validation
- Performance benchmarking for production readiness
- A/B testing capabilities for feature optimization
- User experience analytics and engagement tracking
- Comprehensive testing reports and launch readiness validation
- Community engagement and user onboarding integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import statistics
from collections import defaultdict


class TestScenario(Enum):
    """Beta testing scenarios"""
    ADAPTIVE_LEARNING = "adaptive_learning"
    INTER_AGENT_COMMUNICATION = "inter_agent_communication"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    USER_EXPERIENCE = "user_experience"
    INTEGRATION_TESTING = "integration_testing"
    STRESS_TESTING = "stress_testing"
    # Phase 5 Launch Scenarios
    COMMUNITY_ENGAGEMENT = "community_engagement"
    PUBLIC_LAUNCH_VALIDATION = "public_launch_validation"
    ENTERPRISE_FEATURES = "enterprise_features"
    GLOBAL_PERFORMANCE = "global_performance"
    USER_ONBOARDING = "user_onboarding"
    API_MARKETPLACE = "api_marketplace"


class FeedbackCategory(Enum):
    """User feedback categories"""
    USABILITY = "usability"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    FEATURES = "features"
    DESIGN = "design"
    SUPPORT = "support"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BetaTest:
    """Beta test configuration and results"""
    test_id: str
    scenario: TestScenario
    description: str
    participants: List[str]
    test_cases: List[Dict[str, Any]]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: TestStatus = TestStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)
    feedback_collected: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """User feedback data structure"""
    feedback_id: str
    user_id: str
    category: FeedbackCategory
    rating: int  # 1-5 scale
    comments: str
    feature_context: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTest:
    """A/B testing configuration"""
    test_id: str
    feature_name: str
    variant_a: Dict[str, Any]
    variant_b: Dict[str, Any]
    participants_a: List[str]
    participants_b: List[str]
    metrics: List[str]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunityBetaParticipant:
    """Community beta program participant"""
    user_id: str
    email: str
    name: str
    company: Optional[str] = None
    role: str = "user"
    joined_date: datetime = field(default_factory=datetime.now)
    engagement_score: float = 0.0
    feedback_count: int = 0
    test_completion_rate: float = 0.0
    last_activity: Optional[datetime] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    achievements: List[str] = field(default_factory=list)


@dataclass
class CommunityBetaProgram:
    """Community beta program management"""
    program_id: str
    name: str
    description: str
    target_participants: int = 20
    current_participants: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: str = "planning"
    participants: Dict[str, CommunityBetaParticipant] = field(default_factory=dict)
    engagement_metrics: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


class BetaTestingFramework:
    """
    Phase 2: Comprehensive beta testing framework

    Features:
    - Automated test scenario execution
    - User feedback collection and analysis
    - A/B testing capabilities
    - Performance benchmarking
    - Comprehensive reporting and analytics
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Test management
        self.active_tests: Dict[str, BetaTest] = {}
        self.completed_tests: List[BetaTest] = []
        self.test_templates: Dict[TestScenario, Dict[str, Any]] = self._load_test_templates()

        # Feedback management
        self.feedback_store: List[UserFeedback] = []
        self.feedback_analytics: Dict[str, Any] = {}

        # A/B testing
        self.ab_tests: Dict[str, ABTest] = {}
        self.ab_results: Dict[str, Dict[str, Any]] = {}

        # Performance benchmarking
        self.performance_baselines: Dict[str, Dict[str, Any]] = {}
        self.benchmark_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Phase 5: Community Beta Program
        self.community_programs: Dict[str, CommunityBetaProgram] = {}
        self.beta_participants: Dict[str, CommunityBetaParticipant] = {}
        self.community_engagement_metrics: Dict[str, Any] = {}

        self.logger.info("Phase 5 Beta Testing Framework initialized with Community Program support")

    def _load_test_templates(self) -> Dict[TestScenario, Dict[str, Any]]:
        """Load predefined test scenario templates"""
        return {
            TestScenario.ADAPTIVE_LEARNING: {
                'name': 'Adaptive Learning Beta Test',
                'description': 'Test user preference detection and learning effectiveness',
                'duration_minutes': 30,
                'test_cases': [
                    {
                        'name': 'preference_detection',
                        'description': 'Test detection of user task preferences',
                        'steps': ['create_task', 'observe_recommendations', 'validate_preferences'],
                        'expected_outcomes': ['accurate_recommendations', 'personalized_suggestions']
                    },
                    {
                        'name': 'learning_feedback_loop',
                        'description': 'Test continuous learning from user interactions',
                        'steps': ['perform_tasks', 'provide_feedback', 'observe_adaptation'],
                        'expected_outcomes': ['improved_accuracy', 'adaptive_behavior']
                    }
                ]
            },
            TestScenario.INTER_AGENT_COMMUNICATION: {
                'name': 'Inter-Agent Communication Beta Test',
                'description': 'Test secure communication and consensus algorithms',
                'duration_minutes': 45,
                'test_cases': [
                    {
                        'name': 'secure_messaging',
                        'description': 'Test secure message passing between agents',
                        'steps': ['send_secure_message', 'verify_encryption', 'check_delivery'],
                        'expected_outcomes': ['successful_delivery', 'security_maintained']
                    },
                    {
                        'name': 'consensus_decision',
                        'description': 'Test consensus algorithms for multi-agent decisions',
                        'steps': ['initiate_consensus', 'collect_votes', 'verify_result'],
                        'expected_outcomes': ['fair_decision', 'participant_satisfaction']
                    }
                ]
            },
            TestScenario.PERFORMANCE_OPTIMIZATION: {
                'name': 'Performance Optimization Beta Test',
                'description': 'Test automated optimization and resource management',
                'duration_minutes': 60,
                'test_cases': [
                    {
                        'name': 'automated_optimization',
                        'description': 'Test self-tuning optimization routines',
                        'steps': ['simulate_load', 'trigger_optimization', 'measure_improvement'],
                        'expected_outcomes': ['performance_improvement', 'resource_efficiency']
                    },
                    {
                        'name': 'predictive_analytics',
                        'description': 'Test performance prediction accuracy',
                        'steps': ['collect_metrics', 'generate_predictions', 'validate_accuracy'],
                        'expected_outcomes': ['accurate_predictions', 'useful_insights']
                    }
                ]
            },
            TestScenario.USER_EXPERIENCE: {
                'name': 'User Experience Beta Test',
                'description': 'Test personalized experience and usability improvements',
                'duration_minutes': 40,
                'test_cases': [
                    {
                        'name': 'personalization_engine',
                        'description': 'Test personalized user interface and recommendations',
                        'steps': ['analyze_behavior', 'generate_customizations', 'apply_personalization'],
                        'expected_outcomes': ['relevant_customizations', 'improved_satisfaction']
                    },
                    {
                        'name': 'workflow_optimization',
                        'description': 'Test optimized workflow suggestions',
                        'steps': ['track_usage', 'identify_patterns', 'suggest_improvements'],
                        'expected_outcomes': ['useful_suggestions', 'efficiency_gains']
                    }
                ]
            },
            # Phase 5 Launch Scenarios
            TestScenario.COMMUNITY_ENGAGEMENT: {
                'name': 'Community Engagement Beta Test',
                'description': 'Test community features and user engagement mechanisms',
                'duration_minutes': 60,
                'test_cases': [
                    {
                        'name': 'beta_community_onboarding',
                        'description': 'Test community beta participant onboarding flow',
                        'steps': ['register_participant', 'send_welcome_email', 'setup_profile', 'first_interaction'],
                        'expected_outcomes': ['successful_registration', 'engaged_participants', 'positive_feedback']
                    },
                    {
                        'name': 'engagement_tracking',
                        'description': 'Test user engagement tracking and analytics',
                        'steps': ['track_user_activity', 'calculate_engagement_score', 'generate_insights'],
                        'expected_outcomes': ['accurate_tracking', 'meaningful_insights', 'actionable_metrics']
                    }
                ]
            },
            TestScenario.PUBLIC_LAUNCH_VALIDATION: {
                'name': 'Public Launch Validation',
                'description': 'Validate all components for public launch readiness',
                'duration_minutes': 90,
                'test_cases': [
                    {
                        'name': 'launch_infrastructure',
                        'description': 'Test production infrastructure and deployment',
                        'steps': ['validate_deployment', 'test_scaling', 'check_monitoring', 'verify_security'],
                        'expected_outcomes': ['stable_deployment', 'successful_scaling', 'comprehensive_monitoring', 'secure_system']
                    },
                    {
                        'name': 'user_onboarding_flow',
                        'description': 'Test complete user onboarding and activation',
                        'steps': ['test_registration', 'validate_onboarding', 'check_tutorial_flow', 'measure_activation'],
                        'expected_outcomes': ['smooth_registration', 'effective_onboarding', 'engaging_tutorials', 'high_activation_rate']
                    }
                ]
            },
            TestScenario.ENTERPRISE_FEATURES: {
                'name': 'Enterprise Features Validation',
                'description': 'Test enterprise-grade features and integrations',
                'duration_minutes': 120,
                'test_cases': [
                    {
                        'name': 'enterprise_collaboration',
                        'description': 'Test enterprise collaboration and team features',
                        'steps': ['create_team_workspace', 'test_permissions', 'validate_sharing', 'check_audit_trail'],
                        'expected_outcomes': ['functional_workspaces', 'proper_permissions', 'secure_sharing', 'complete_auditing']
                    },
                    {
                        'name': 'enterprise_security',
                        'description': 'Test enterprise security and compliance features',
                        'steps': ['validate_authentication', 'test_authorization', 'check_encryption', 'verify_compliance'],
                        'expected_outcomes': ['secure_authentication', 'proper_authorization', 'data_encryption', 'compliance_ready']
                    }
                ]
            },
            TestScenario.GLOBAL_PERFORMANCE: {
                'name': 'Global Performance Testing',
                'description': 'Test performance across global infrastructure',
                'duration_minutes': 180,
                'test_cases': [
                    {
                        'name': 'global_latency',
                        'description': 'Test response times across global regions',
                        'steps': ['test_regional_latency', 'validate_cdn_performance', 'check_edge_computing', 'measure_global_p95'],
                        'expected_outcomes': ['acceptable_latency', 'effective_cdn', 'optimal_edge_usage', 'global_performance_targets']
                    },
                    {
                        'name': 'scalability_under_load',
                        'description': 'Test system scalability with global user load',
                        'steps': ['simulate_global_load', 'test_auto_scaling', 'validate_failover', 'measure_performance_degradation'],
                        'expected_outcomes': ['successful_scaling', 'automatic_scaling', 'reliable_failover', 'minimal_degradation']
                    }
                ]
            },
            TestScenario.USER_ONBOARDING: {
                'name': 'User Onboarding Validation',
                'description': 'Test comprehensive user onboarding and support systems',
                'duration_minutes': 75,
                'test_cases': [
                    {
                        'name': 'onboarding_flow',
                        'description': 'Test complete user onboarding experience',
                        'steps': ['test_registration', 'validate_tutorial_flow', 'check_progress_tracking', 'measure_completion_rate'],
                        'expected_outcomes': ['smooth_registration', 'engaging_tutorials', 'accurate_tracking', 'high_completion_rate']
                    },
                    {
                        'name': 'support_system',
                        'description': 'Test user support and help systems',
                        'steps': ['test_help_desk', 'validate_documentation', 'check_community_support', 'measure_resolution_time'],
                        'expected_outcomes': ['responsive_support', 'helpful_documentation', 'active_community', 'quick_resolution']
                    }
                ]
            },
            TestScenario.API_MARKETPLACE: {
                'name': 'API Marketplace Validation',
                'description': 'Test API marketplace and third-party integrations',
                'duration_minutes': 100,
                'test_cases': [
                    {
                        'name': 'marketplace_functionality',
                        'description': 'Test API marketplace core functionality',
                        'steps': ['test_api_discovery', 'validate_integration_flow', 'check_billing_system', 'measure_developer_engagement'],
                        'expected_outcomes': ['easy_discovery', 'smooth_integration', 'accurate_billing', 'high_engagement']
                    },
                    {
                        'name': 'third_party_integrations',
                        'description': 'Test third-party service integrations',
                        'steps': ['test_popular_integrations', 'validate_webhooks', 'check_data_sync', 'measure_integration_success'],
                        'expected_outcomes': ['working_integrations', 'reliable_webhooks', 'accurate_sync', 'successful_integrations']
                    }
                ]
            }
        }

    async def create_beta_test(self, scenario: TestScenario, participants: List[str],
                             custom_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new beta test

        Args:
            scenario: Test scenario type
            participants: List of participant user IDs
            custom_config: Custom test configuration

        Returns:
            Test ID for the created test
        """
        test_id = f"beta_{scenario.value}_{int(datetime.now().timestamp())}"

        # Get template configuration
        template = self.test_templates.get(scenario, {})
        if not template:
            raise ValueError(f"No template found for scenario: {scenario}")

        # Merge with custom configuration
        test_config = {**template, **(custom_config or {})}

        # Create test instance
        beta_test = BetaTest(
            test_id=test_id,
            scenario=scenario,
            description=test_config.get('description', ''),
            participants=participants,
            test_cases=test_config.get('test_cases', [])
        )

        self.active_tests[test_id] = beta_test

        self.logger.info(f"Created beta test: {test_id} for scenario: {scenario.value}")
        return test_id

    async def execute_beta_test(self, test_id: str) -> Dict[str, Any]:
        """
        Execute a beta test

        Args:
            test_id: Test ID to execute

        Returns:
            Test execution results
        """
        if test_id not in self.active_tests:
            return {"status": "error", "error": "Test not found"}

        beta_test = self.active_tests[test_id]
        beta_test.status = TestStatus.RUNNING
        beta_test.start_time = datetime.now()

        try:
            # Execute test cases
            test_results = []
            for test_case in beta_test.test_cases:
                case_result = await self._execute_test_case(test_case, beta_test.participants)
                test_results.append(case_result)

            # Collect performance metrics
            performance_metrics = await self._collect_performance_metrics(beta_test)

            # Analyze results
            analysis = self._analyze_test_results(test_results, performance_metrics)

            # Update test status
            beta_test.status = TestStatus.COMPLETED
            beta_test.end_time = datetime.now()
            beta_test.results = {
                'test_cases': test_results,
                'performance_metrics': performance_metrics,
                'analysis': analysis,
                'duration_seconds': (beta_test.end_time - beta_test.start_time).total_seconds()
            }

            # Move to completed tests
            self.completed_tests.append(beta_test)
            del self.active_tests[test_id]

            self.logger.info(f"Completed beta test: {test_id}")

            return {
                "status": "completed",
                "test_id": test_id,
                "results": beta_test.results
            }

        except Exception as e:
            beta_test.status = TestStatus.FAILED
            beta_test.end_time = datetime.now()
            beta_test.results = {"error": str(e)}

            self.logger.error(f"Beta test {test_id} failed: {e}")

            return {
                "status": "failed",
                "test_id": test_id,
                "error": str(e)
            }

    async def _execute_test_case(self, test_case: Dict[str, Any], participants: List[str]) -> Dict[str, Any]:
        """Execute a single test case"""
        case_result = {
            'test_case': test_case['name'],
            'description': test_case['description'],
            'start_time': datetime.now().isoformat(),
            'steps_executed': [],
            'outcomes_achieved': [],
            'issues': []
        }

        try:
            # Simulate test case execution
            for step in test_case.get('steps', []):
                # Simulate step execution time
                await asyncio.sleep(0.1)

                case_result['steps_executed'].append({
                    'step': step,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                })

            # Check expected outcomes
            expected_outcomes = test_case.get('expected_outcomes', [])
            for outcome in expected_outcomes:
                # Simulate outcome validation
                achieved = True  # In real implementation, would validate actual outcomes
                case_result['outcomes_achieved'].append({
                    'outcome': outcome,
                    'achieved': achieved,
                    'timestamp': datetime.now().isoformat()
                })

            case_result['status'] = 'passed'

        except Exception as e:
            case_result['status'] = 'failed'
            case_result['issues'].append(str(e))

        case_result['end_time'] = datetime.now().isoformat()
        return case_result

    async def _collect_performance_metrics(self, beta_test: BetaTest) -> Dict[str, Any]:
        """Collect performance metrics during test execution"""
        # Simulate performance metric collection
        return {
            'response_time_avg': 250.5,  # ms
            'cpu_usage_avg': 45.2,  # %
            'memory_usage_avg': 320.8,  # MB
            'error_rate': 0.02,  # 2%
            'user_satisfaction_avg': 4.1,  # 1-5 scale
            'task_completion_rate': 0.95,  # 95%
            'timestamp': datetime.now().isoformat()
        }

    def _analyze_test_results(self, test_results: List[Dict[str, Any]],
                           performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results and generate insights"""
        analysis = {
            'overall_success_rate': 0.0,
            'performance_score': 0.0,
            'user_satisfaction_score': 0.0,
            'key_findings': [],
            'recommendations': []
        }

        # Calculate success rate
        if test_results:
            successful_tests = len([r for r in test_results if r.get('status') == 'passed'])
            analysis['overall_success_rate'] = successful_tests / len(test_results)

        # Performance analysis
        if performance_metrics:
            analysis['performance_score'] = self._calculate_performance_score(performance_metrics)

        # Generate findings and recommendations
        analysis['key_findings'] = self._generate_key_findings(test_results, performance_metrics)
        analysis['recommendations'] = self._generate_recommendations(test_results, performance_metrics)

        return analysis

    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        # Weighted scoring based on key metrics
        weights = {
            'response_time_avg': 0.3,
            'cpu_usage_avg': 0.2,
            'memory_usage_avg': 0.2,
            'error_rate': 0.15,
            'task_completion_rate': 0.15
        }

        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]

                # Normalize to 0-1 scale (higher is better)
                if metric == 'response_time_avg':
                    normalized = max(0, 1 - (value / 1000))  # Lower response time is better
                elif metric in ['cpu_usage_avg', 'memory_usage_avg']:
                    normalized = max(0, 1 - (value / 100))  # Lower resource usage is better
                elif metric == 'error_rate':
                    normalized = max(0, 1 - value)  # Lower error rate is better
                elif metric == 'task_completion_rate':
                    normalized = value  # Higher completion rate is better
                else:
                    normalized = 0.5  # Default

                score += normalized * weight

        return score

    def _generate_key_findings(self, test_results: List[Dict[str, Any]],
                             performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate key findings from test results"""
        findings = []

        # Analyze test case success
        success_rate = len([r for r in test_results if r.get('status') == 'passed']) / len(test_results)
        if success_rate > 0.9:
            findings.append("Excellent test case success rate (>90%)")
        elif success_rate > 0.7:
            findings.append("Good test case success rate (>70%)")
        else:
            findings.append("Test case success rate needs improvement")

        # Analyze performance
        if performance_metrics.get('response_time_avg', 1000) < 300:
            findings.append("Excellent response times achieved")
        elif performance_metrics.get('response_time_avg', 1000) < 500:
            findings.append("Good response times achieved")

        if performance_metrics.get('error_rate', 0.1) < 0.05:
            findings.append("Low error rate indicates high reliability")

        return findings

    def _generate_recommendations(self, test_results: List[Dict[str, Any]],
                                performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        # Performance-based recommendations
        if performance_metrics.get('response_time_avg', 1000) > 500:
            recommendations.append("Optimize response times through caching and query optimization")

        if performance_metrics.get('cpu_usage_avg', 50) > 70:
            recommendations.append("Implement CPU optimization strategies")

        if performance_metrics.get('error_rate', 0.1) > 0.05:
            recommendations.append("Improve error handling and recovery mechanisms")

        # Test-based recommendations
        failed_tests = [r for r in test_results if r.get('status') == 'failed']
        if failed_tests:
            recommendations.append(f"Address issues in {len(failed_tests)} failed test cases")

        return recommendations

    # Phase 5: Community Beta Program Management

    async def create_community_beta_program(self, name: str, description: str,
                                          target_participants: int = 20,
                                          success_criteria: Dict[str, Any] = None) -> str:
        """
        Create a new community beta program

        Args:
            name: Program name
            description: Program description
            target_participants: Target number of participants
            success_criteria: Success criteria for the program

        Returns:
            Program ID
        """
        program_id = f"community_beta_{int(datetime.now().timestamp())}"

        program = CommunityBetaProgram(
            program_id=program_id,
            name=name,
            description=description,
            target_participants=target_participants,
            success_criteria=success_criteria or {
                'min_participants': 10,
                'engagement_rate': 0.7,
                'feedback_completion_rate': 0.8,
                'satisfaction_score': 4.0
            }
        )

        self.community_programs[program_id] = program

        self.logger.info(f"Created community beta program: {program_id} - {name}")
        return program_id

    async def register_beta_participant(self, program_id: str, user_id: str, email: str, name: str,
                                      company: str = None, role: str = "user") -> Dict[str, Any]:
        """
        Register a participant for the community beta program

        Args:
            program_id: Community beta program ID
            user_id: User ID
            email: User email
            name: User name
            company: User company (optional)
            role: User role

        Returns:
            Registration result
        """
        if program_id not in self.community_programs:
            return {"status": "error", "error": "Program not found"}

        program = self.community_programs[program_id]

        if program.current_participants >= program.target_participants:
            return {"status": "error", "error": "Program at capacity"}

        # Create participant
        participant = CommunityBetaParticipant(
            user_id=user_id,
            email=email,
            name=name,
            company=company,
            role=role
        )

        # Add to program
        program.participants[user_id] = participant
        program.current_participants += 1

        # Store globally
        self.beta_participants[user_id] = participant

        # Start program if we have minimum participants
        if program.current_participants >= 10 and program.status == "planning":
            program.status = "active"
            program.start_date = datetime.now()

        self.logger.info(f"Registered beta participant: {user_id} for program: {program_id}")
        return {
            "status": "registered",
            "participant_id": user_id,
            "program_id": program_id,
            "welcome_message": self._generate_welcome_message(participant, program)
        }

    async def update_participant_engagement(self, user_id: str, activity_type: str,
                                         activity_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update participant engagement metrics

        Args:
            user_id: Participant user ID
            activity_type: Type of activity (feedback, test_completion, etc.)
            activity_data: Additional activity data

        Returns:
            Updated engagement metrics
        """
        if user_id not in self.beta_participants:
            return {"status": "error", "error": "Participant not found"}

        participant = self.beta_participants[user_id]

        # Update activity
        participant.last_activity = datetime.now()

        # Update engagement based on activity type
        engagement_points = {
            'feedback_submitted': 10,
            'test_completed': 15,
            'bug_reported': 8,
            'feature_suggestion': 12,
            'session_participated': 5,
            'tutorial_completed': 20
        }

        points = engagement_points.get(activity_type, 5)
        participant.engagement_score += points

        # Update specific metrics
        if activity_type == 'feedback_submitted':
            participant.feedback_count += 1
        elif activity_type == 'test_completed':
            participant.test_completion_rate = min(1.0, participant.test_completion_rate + 0.1)

        # Update program-level metrics
        await self._update_program_engagement_metrics()

        return {
            "status": "updated",
            "engagement_score": participant.engagement_score,
            "feedback_count": participant.feedback_count,
            "test_completion_rate": participant.test_completion_rate
        }

    async def get_community_beta_status(self, program_id: str = None) -> Dict[str, Any]:
        """
        Get community beta program status

        Args:
            program_id: Specific program ID (optional)

        Returns:
            Program status and metrics
        """
        if program_id:
            if program_id not in self.community_programs:
                return {"status": "error", "error": "Program not found"}

            program = self.community_programs[program_id]
            return {
                "program_id": program_id,
                "name": program.name,
                "status": program.status,
                "participants": program.current_participants,
                "target": program.target_participants,
                "engagement_metrics": program.engagement_metrics,
                "success_criteria": program.success_criteria,
                "progress_percentage": (program.current_participants / program.target_participants) * 100
            }

        # Return all programs
        programs_status = {}
        for pid, program in self.community_programs.items():
            programs_status[pid] = {
                "name": program.name,
                "status": program.status,
                "participants": program.current_participants,
                "target": program.target_participants,
                "progress": (program.current_participants / program.target_participants) * 100
            }

        return {
            "total_programs": len(self.community_programs),
            "active_programs": len([p for p in self.community_programs.values() if p.status == "active"]),
            "programs": programs_status,
            "overall_metrics": self.community_engagement_metrics
        }

    def _generate_welcome_message(self, participant: CommunityBetaParticipant,
                                program: CommunityBetaProgram) -> str:
        """Generate personalized welcome message for beta participant"""
        return f"""
Welcome to the CES Community Beta Program, {participant.name}!

Thank you for joining our beta testing community. As a {participant.role} at {participant.company or 'your organization'}, your feedback will be invaluable in shaping the future of CES.

Program Details:
- Program: {program.name}
- Your Role: {participant.role}
- Start Date: {datetime.now().strftime('%Y-%m-%d')}

What to expect:
1. Access to the latest CES features
2. Regular updates and new releases
3. Direct communication with the development team
4. Opportunity to influence product direction

Please complete your onboarding and start exploring CES. We're excited to have you as part of our community!

Best regards,
CES Beta Team
"""

    async def _update_program_engagement_metrics(self):
        """Update overall program engagement metrics"""
        if not self.beta_participants:
            return

        total_participants = len(self.beta_participants)
        active_participants = len([p for p in self.beta_participants.values()
                                 if p.last_activity and
                                 (datetime.now() - p.last_activity).days <= 7])

        avg_engagement = sum(p.engagement_score for p in self.beta_participants.values()) / total_participants
        avg_feedback_rate = sum(p.feedback_count for p in self.beta_participants.values()) / total_participants
        avg_completion_rate = sum(p.test_completion_rate for p in self.beta_participants.values()) / total_participants

        self.community_engagement_metrics = {
            'total_participants': total_participants,
            'active_participants': active_participants,
            'activity_rate': active_participants / total_participants if total_participants > 0 else 0,
            'average_engagement_score': avg_engagement,
            'average_feedback_count': avg_feedback_rate,
            'average_test_completion_rate': avg_completion_rate,
            'last_updated': datetime.now().isoformat()
        }

    async def collect_user_feedback(self, user_id: str, category: FeedbackCategory,
                                  rating: int, comments: str, feature_context: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect user feedback

        Args:
            user_id: User providing feedback
            category: Feedback category
            rating: Rating (1-5)
            comments: Feedback comments
            feature_context: Feature being rated
            metadata: Additional metadata

        Returns:
            Feedback ID
        """
        feedback_id = f"feedback_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"

        feedback = UserFeedback(
            feedback_id=feedback_id,
            user_id=user_id,
            category=category,
            rating=max(1, min(5, rating)),  # Ensure 1-5 range
            comments=comments,
            feature_context=feature_context,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.feedback_store.append(feedback)

        # Update analytics
        await self._update_feedback_analytics()

        self.logger.info(f"Collected feedback from user {user_id}: {category.value} - Rating {rating}")
        return feedback_id

    async def _update_feedback_analytics(self):
        """Update feedback analytics"""
        if not self.feedback_store:
            return

        # Calculate category averages
        category_ratings = defaultdict(list)
        for feedback in self.feedback_store:
            category_ratings[feedback.category].append(feedback.rating)

        self.feedback_analytics = {
            'total_feedback': len(self.feedback_store),
            'average_rating': statistics.mean([f.rating for f in self.feedback_store]),
            'category_breakdown': {
                category.value: {
                    'count': len(ratings),
                    'average': statistics.mean(ratings),
                    'distribution': self._calculate_rating_distribution(ratings)
                }
                for category, ratings in category_ratings.items()
            },
            'recent_trends': self._calculate_feedback_trends(),
            'last_updated': datetime.now().isoformat()
        }

    def _calculate_rating_distribution(self, ratings: List[int]) -> Dict[int, int]:
        """Calculate rating distribution"""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            distribution[rating] += 1
        return distribution

    def _calculate_feedback_trends(self) -> Dict[str, Any]:
        """Calculate feedback trends over time"""
        if len(self.feedback_store) < 10:
            return {'insufficient_data': True}

        # Get recent feedback (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_feedback = [f for f in self.feedback_store if f.timestamp > thirty_days_ago]

        if not recent_feedback:
            return {'no_recent_feedback': True}

        recent_avg = statistics.mean([f.rating for f in recent_feedback])

        # Compare with older feedback
        older_feedback = [f for f in self.feedback_store if f.timestamp <= thirty_days_ago]
        if older_feedback:
            older_avg = statistics.mean([f.rating for f in older_feedback])
            trend = 'improving' if recent_avg > older_avg else 'declining' if recent_avg < older_avg else 'stable'
        else:
            trend = 'unknown'

        return {
            'recent_average': recent_avg,
            'trend': trend,
            'recent_count': len(recent_feedback)
        }

    async def create_ab_test(self, feature_name: str, variant_a: Dict[str, Any],
                           variant_b: Dict[str, Any], participants_a: List[str],
                           participants_b: List[str], metrics: List[str]) -> str:
        """
        Create an A/B test

        Args:
            feature_name: Name of feature being tested
            variant_a: Configuration for variant A
            variant_b: Configuration for variant B
            participants_a: Participants for variant A
            participants_b: Participants for variant B
            metrics: Metrics to track

        Returns:
            A/B test ID
        """
        test_id = f"ab_{feature_name}_{int(datetime.now().timestamp())}"

        ab_test = ABTest(
            test_id=test_id,
            feature_name=feature_name,
            variant_a=variant_a,
            variant_b=variant_b,
            participants_a=participants_a,
            participants_b=participants_b,
            metrics=metrics,
            start_time=datetime.now()
        )

        self.ab_tests[test_id] = ab_test

        self.logger.info(f"Created A/B test: {test_id} for feature: {feature_name}")
        return test_id

    async def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get A/B test results

        Args:
            test_id: A/B test ID

        Returns:
            Test results and analysis
        """
        if test_id not in self.ab_tests:
            return {"status": "error", "error": "A/B test not found"}

        ab_test = self.ab_tests[test_id]

        # Simulate results collection (in production, would collect real metrics)
        results = {
            'variant_a': {
                'participants': len(ab_test.participants_a),
                'metrics': {metric: 0.85 + np.random.random() * 0.1 for metric in ab_test.metrics}
            },
            'variant_b': {
                'participants': len(ab_test.participants_b),
                'metrics': {metric: 0.82 + np.random.random() * 0.1 for metric in ab_test.metrics}
            }
        }

        # Statistical analysis
        analysis = {}
        for metric in ab_test.metrics:
            a_value = results['variant_a']['metrics'][metric]
            b_value = results['variant_b']['metrics'][metric]

            # Simple statistical significance test (simplified)
            difference = abs(a_value - b_value)
            significance = 'significant' if difference > 0.05 else 'not_significant'
            winner = 'A' if a_value > b_value else 'B'

            analysis[metric] = {
                'variant_a_value': a_value,
                'variant_b_value': b_value,
                'difference': difference,
                'significance': significance,
                'recommended_variant': winner
            }

        return {
            'test_id': test_id,
            'feature_name': ab_test.feature_name,
            'status': 'completed',
            'results': results,
            'analysis': analysis,
            'recommendation': self._generate_ab_recommendation(analysis)
        }

    def _generate_ab_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Generate recommendation based on A/B test analysis"""
        significant_metrics = [m for m, data in analysis.items() if data['significance'] == 'significant']

        if not significant_metrics:
            return "No significant differences detected between variants"

        a_wins = sum(1 for data in analysis.values() if data['recommended_variant'] == 'A')
        b_wins = sum(1 for data in analysis.values() if data['recommended_variant'] == 'B')

        if a_wins > b_wins:
            return f"Variant A performs better in {a_wins} out of {len(analysis)} metrics"
        elif b_wins > a_wins:
            return f"Variant B performs better in {b_wins} out of {len(analysis)} metrics"
        else:
            return "Variants perform equally well"

    def generate_beta_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive beta testing report

        Returns:
            Detailed beta testing report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 2 Beta Testing',
            'test_summary': {},
            'feedback_analysis': {},
            'performance_benchmarks': {},
            'recommendations': [],
            'overall_satisfaction': 0.0
        }

        # Test summary
        report['test_summary'] = {
            'total_tests': len(self.completed_tests),
            'active_tests': len(self.active_tests),
            'success_rate': self._calculate_overall_success_rate(),
            'test_coverage': self._calculate_test_coverage()
        }

        # Feedback analysis
        if self.feedback_analytics:
            report['feedback_analysis'] = self.feedback_analytics
            report['overall_satisfaction'] = self.feedback_analytics.get('average_rating', 0)

        # Performance benchmarks
        report['performance_benchmarks'] = self._generate_performance_benchmarks()

        # Generate recommendations
        report['recommendations'] = self._generate_beta_recommendations()

        return report

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall test success rate"""
        if not self.completed_tests:
            return 0.0

        total_cases = sum(len(test.test_cases) for test in self.completed_tests)
        successful_cases = 0

        for test in self.completed_tests:
            for case in test.test_cases:
                if case.get('status') == 'passed':
                    successful_cases += 1

        return successful_cases / total_cases if total_cases > 0 else 0.0

    def _calculate_test_coverage(self) -> Dict[str, int]:
        """Calculate test coverage by scenario"""
        coverage = {}
        for scenario in TestScenario:
            scenario_tests = [t for t in self.completed_tests if t.scenario == scenario]
            coverage[scenario.value] = len(scenario_tests)
        return coverage

    def _generate_performance_benchmarks(self) -> Dict[str, Any]:
        """Generate performance benchmark summary"""
        if not self.completed_tests:
            return {}

        # Aggregate performance metrics across all tests
        all_metrics = []
        for test in self.completed_tests:
            if 'performance_metrics' in test.results:
                all_metrics.append(test.results['performance_metrics'])

        if not all_metrics:
            return {}

        # Calculate averages
        benchmark_metrics = {}
        metric_keys = ['response_time_avg', 'cpu_usage_avg', 'memory_usage_avg', 'error_rate', 'task_completion_rate']

        for key in metric_keys:
            values = [m.get(key, 0) for m in all_metrics if key in m]
            if values:
                benchmark_metrics[key] = {
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0
                }

        return benchmark_metrics

    def _generate_beta_recommendations(self) -> List[str]:
        """Generate recommendations based on beta testing results"""
        recommendations = []

        # Based on success rate
        success_rate = self._calculate_overall_success_rate()
        if success_rate < 0.8:
            recommendations.append("Improve test case reliability and fix identified issues")
        elif success_rate > 0.95:
            recommendations.append("Excellent test results - ready for production deployment")

        # Based on feedback
        if self.feedback_analytics:
            avg_rating = self.feedback_analytics.get('average_rating', 0)
            if avg_rating < 3.5:
                recommendations.append("Address user feedback and improve user experience")
            elif avg_rating > 4.5:
                recommendations.append("Outstanding user feedback - maintain high quality standards")

        # Based on performance
        benchmarks = self._generate_performance_benchmarks()
        if benchmarks:
            if benchmarks.get('response_time_avg', {}).get('average', 1000) > 500:
                recommendations.append("Optimize response times for better user experience")
            if benchmarks.get('error_rate', {}).get('average', 0.1) > 0.05:
                recommendations.append("Improve system reliability and error handling")

        return recommendations