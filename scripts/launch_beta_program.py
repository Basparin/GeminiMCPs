#!/usr/bin/env python3
"""
CES Community Beta Program Launcher - Phase 5

Launches the community beta testing program with 10-20 users,
integrates feedback collection, and manages the beta testing lifecycle.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add CES to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ces.core.beta_testing_framework import BetaTestingFramework, TestScenario
from ces.feedback.feedback_collector import FeedbackCollector
from ces.feedback.feedback_analyzer import FeedbackAnalyzer
from ces.feedback.feedback_integrator import FeedbackIntegrator
from ces.core.logging_config import get_logger

logger = get_logger(__name__)


class BetaProgramLauncher:
    """Launcher for CES Community Beta Program"""

    def __init__(self):
        self.beta_framework = BetaTestingFramework()
        self.feedback_collector = FeedbackCollector()
        self.feedback_analyzer = FeedbackAnalyzer()
        self.feedback_integrator = FeedbackIntegrator()

        # Beta program configuration
        self.program_config = {
            'name': 'CES Community Beta Program - Phase 5 Launch',
            'description': 'Enterprise-grade beta testing for CES public launch with community engagement',
            'target_participants': 20,
            'min_participants': 10,
            'duration_weeks': 8,
            'success_criteria': {
                'engagement_rate': 0.75,
                'feedback_completion_rate': 0.85,
                'satisfaction_score': 4.2,
                'retention_rate': 0.8
            }
        }

        logger.info("CES Beta Program Launcher initialized")

    async def launch_beta_program(self) -> Dict[str, Any]:
        """
        Launch the community beta program

        Returns:
            Launch results and status
        """
        logger.info("Launching CES Community Beta Program...")

        try:
            # 1. Create community beta program
            program_id = await self.beta_framework.create_community_beta_program(
                name=self.program_config['name'],
                description=self.program_config['description'],
                target_participants=self.program_config['target_participants'],
                success_criteria=self.program_config['success_criteria']
            )

            # 2. Register initial beta participants
            participants = await self._register_initial_participants(program_id)

            # 3. Create beta testing campaigns
            campaigns = await self._create_beta_campaigns()

            # 4. Setup feedback collection campaigns
            feedback_campaigns = await self._setup_feedback_campaigns()

            # 5. Launch initial test scenarios
            test_results = await self._launch_initial_tests()

            # 6. Setup monitoring and analytics
            monitoring_setup = await self._setup_monitoring()

            launch_results = {
                'program_id': program_id,
                'status': 'launched',
                'participants_registered': len(participants),
                'campaigns_created': len(campaigns),
                'feedback_campaigns': len(feedback_campaigns),
                'initial_tests': len(test_results),
                'monitoring_active': monitoring_setup['status'] == 'active',
                'launch_timestamp': datetime.now().isoformat(),
                'next_milestones': self._get_next_milestones()
            }

            logger.info(f"CES Beta Program launched successfully with {len(participants)} participants")
            return launch_results

        except Exception as e:
            logger.error(f"Failed to launch beta program: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _register_initial_participants(self, program_id: str) -> List[Dict[str, Any]]:
        """Register initial beta participants"""
        # Simulated initial participants (in production, this would come from a database or API)
        initial_participants = [
            {'email': 'alice.dev@company.com', 'name': 'Alice Johnson', 'company': 'TechCorp', 'role': 'Senior Developer'},
            {'email': 'bob.architect@startup.io', 'name': 'Bob Smith', 'company': 'StartupXYZ', 'role': 'Solutions Architect'},
            {'email': 'carol.manager@enterprise.org', 'name': 'Carol Williams', 'company': 'Enterprise Inc', 'role': 'Engineering Manager'},
            {'email': 'david.analyst@consulting.com', 'name': 'David Brown', 'company': 'Consulting Pro', 'role': 'Data Analyst'},
            {'email': 'eve.researcher@university.edu', 'name': 'Eve Davis', 'company': 'State University', 'role': 'Researcher'},
            {'email': 'frank.freelancer@independent.dev', 'name': 'Frank Miller', 'company': 'Independent', 'role': 'Freelance Developer'},
            {'email': 'grace.tester@qa-team.com', 'name': 'Grace Wilson', 'company': 'QA Solutions', 'role': 'QA Lead'},
            {'email': 'henry.product@product-team.io', 'name': 'Henry Taylor', 'company': 'ProductCo', 'role': 'Product Manager'},
            {'email': 'iris.designer@design-studio.com', 'name': 'Iris Anderson', 'company': 'Design Studio', 'role': 'UX Designer'},
            {'email': 'jack.ops@devops-team.net', 'name': 'Jack Thomas', 'company': 'DevOps Pro', 'role': 'DevOps Engineer'},
            {'email': 'kate.lead@leadership.org', 'name': 'Kate Jackson', 'company': 'Leadership Inc', 'role': 'Tech Lead'},
            {'email': 'liam.student@university.edu', 'name': 'Liam White', 'company': 'Tech University', 'role': 'Computer Science Student'},
            {'email': 'mia.startup@newco.io', 'name': 'Mia Garcia', 'company': 'NewCo', 'role': 'Founder'},
            {'email': 'noah.consultant@advice.com', 'name': 'Noah Martinez', 'company': 'Tech Advisors', 'role': 'Technical Consultant'},
            {'email': 'olivia.writer@tech-blog.com', 'name': 'Olivia Robinson', 'company': 'Tech Blog', 'role': 'Technical Writer'}
        ]

        registered_participants = []

        for participant in initial_participants:
            try:
                result = await self.beta_framework.register_beta_participant(
                    program_id=program_id,
                    user_id=f"user_{len(registered_participants) + 1}",
                    email=participant['email'],
                    name=participant['name'],
                    company=participant['company'],
                    role=participant['role']
                )

                if result['status'] == 'registered':
                    registered_participants.append({
                        'user_id': result['participant_id'],
                        'name': participant['name'],
                        'email': participant['email'],
                        'company': participant['company'],
                        'welcome_message': result.get('welcome_message', '')
                    })

                    logger.info(f"Registered beta participant: {participant['name']} ({participant['email']})")

            except Exception as e:
                logger.error(f"Failed to register participant {participant['email']}: {e}")

        return registered_participants

    async def _create_beta_campaigns(self) -> List[str]:
        """Create beta testing campaigns"""
        campaigns = []

        # Community engagement campaign
        campaign_id = await self.beta_framework.create_beta_test(
            scenario=TestScenario.COMMUNITY_ENGAGEMENT,
            participants=[f"user_{i+1}" for i in range(15)]  # First 15 participants
        )
        campaigns.append(campaign_id)

        # Enterprise features campaign
        campaign_id = await self.beta_framework.create_beta_test(
            scenario=TestScenario.ENTERPRISE_FEATURES,
            participants=[f"user_{i+1}" for i in range(8, 16)]  # Participants 8-15
        )
        campaigns.append(campaign_id)

        # Global performance campaign
        campaign_id = await self.beta_framework.create_beta_test(
            scenario=TestScenario.GLOBAL_PERFORMANCE,
            participants=[f"user_{i+1}" for i in range(12, 20)]  # Last 8 participants
        )
        campaigns.append(campaign_id)

        logger.info(f"Created {len(campaigns)} beta testing campaigns")
        return campaigns

    async def _setup_feedback_campaigns(self) -> List[str]:
        """Setup feedback collection campaigns"""
        campaigns = []

        # Initial onboarding feedback
        campaign_id = await self.feedback_collector.create_feedback_campaign(
            name="CES Beta Onboarding Feedback",
            description="Collect feedback on initial CES onboarding experience",
            channel=self.feedback_collector.feedback_handlers.keys().__iter__().__next__(),  # Get first channel
            target_users=[f"user_{i+1}" for i in range(15)],
            questions=[
                {
                    "id": "onboarding_experience",
                    "type": "rating",
                    "question": "How would you rate your onboarding experience?",
                    "scale": "1-5"
                },
                {
                    "id": "setup_difficulty",
                    "type": "text",
                    "question": "How easy was it to set up and start using CES?"
                },
                {
                    "id": "first_impressions",
                    "type": "text",
                    "question": "What were your first impressions of CES?"
                }
            ]
        )
        campaigns.append(campaign_id)

        # Feature usage feedback
        campaign_id = await self.feedback_collector.create_feedback_campaign(
            name="CES Feature Usage Feedback",
            description="Collect feedback on CES feature usage and effectiveness",
            channel=self.feedback_collector.feedback_handlers.keys().__iter__().__next__(),
            target_users=[f"user_{i+1}" for i in range(10, 20)],
            questions=[
                {
                    "id": "most_used_feature",
                    "type": "multiple_choice",
                    "question": "Which CES feature have you used the most?",
                    "options": ["AI Assistant", "Task Management", "Collaboration", "Analytics", "Code Generation"]
                },
                {
                    "id": "feature_effectiveness",
                    "type": "rating",
                    "question": "How effective are CES features for your workflow?",
                    "scale": "1-5"
                },
                {
                    "id": "improvement_suggestions",
                    "type": "text",
                    "question": "What improvements would you suggest for CES?"
                }
            ]
        )
        campaigns.append(campaign_id)

        logger.info(f"Created {len(campaigns)} feedback collection campaigns")
        return campaigns

    async def _launch_initial_tests(self) -> List[Dict[str, Any]]:
        """Launch initial test scenarios"""
        test_results = []

        # Launch community engagement test
        result = await self.beta_framework.execute_beta_test("beta_community_engagement_1")
        test_results.append(result)

        # Launch enterprise features test
        result = await self.beta_framework.execute_beta_test("beta_enterprise_features_1")
        test_results.append(result)

        logger.info(f"Launched {len(test_results)} initial test scenarios")
        return test_results

    async def _setup_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring and analytics for beta program"""
        try:
            # Get initial program status
            status = await self.beta_framework.get_community_beta_status()

            # Setup feedback analytics
            analytics = await self.feedback_collector.get_feedback_analytics(time_range_days=7)

            # Setup real-time insight generation
            insight_queue = await self.feedback_analyzer.generate_real_time_insights(
                asyncio.Queue()  # In production, this would be connected to actual feedback stream
            )

            return {
                'status': 'active',
                'program_status': status,
                'feedback_analytics': analytics,
                'real_time_insights': 'configured',
                'monitoring_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _get_next_milestones(self) -> List[Dict[str, Any]]:
        """Get next milestones for beta program"""
        now = datetime.now()

        return [
            {
                'milestone': 'Week 1 Check-in',
                'date': (now + timedelta(days=7)).strftime('%Y-%m-%d'),
                'description': 'Initial feedback collection and engagement assessment'
            },
            {
                'milestone': 'Week 2 Feature Testing',
                'date': (now + timedelta(days=14)).strftime('%Y-%m-%d'),
                'description': 'Complete initial feature testing scenarios'
            },
            {
                'milestone': 'Week 4 Mid-Program Review',
                'date': (now + timedelta(days=28)).strftime('%Y-%m-%d'),
                'description': 'Comprehensive program review and iteration planning'
            },
            {
                'milestone': 'Week 6 Advanced Features',
                'date': (now + timedelta(days=42)).strftime('%Y-%m-%d'),
                'description': 'Test advanced features and enterprise capabilities'
            },
            {
                'milestone': 'Week 8 Final Evaluation',
                'date': (now + timedelta(days=56)).strftime('%Y-%m-%d'),
                'description': 'Final program evaluation and launch readiness assessment'
            }
        ]

    async def get_program_status(self) -> Dict[str, Any]:
        """Get current beta program status"""
        try:
            # Get program status
            program_status = await self.beta_framework.get_community_beta_status()

            # Get feedback analytics
            feedback_analytics = await self.feedback_collector.get_feedback_analytics(time_range_days=7)

            # Get sentiment analysis
            sentiment_summary = await self.feedback_analyzer.get_sentiment_summary(time_range_days=7)

            # Get integration status
            integration_status = await self.feedback_integrator.get_integration_status()

            return {
                'program_status': program_status,
                'feedback_analytics': feedback_analytics,
                'sentiment_summary': sentiment_summary,
                'integration_status': integration_status,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get program status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


async def main():
    """Main launcher function"""
    launcher = BetaProgramLauncher()

    print("🚀 Launching CES Community Beta Program - Phase 5")
    print("=" * 60)

    # Launch the program
    launch_results = await launcher.launch_beta_program()

    if launch_results['status'] == 'launched':
        print("✅ CES Beta Program launched successfully!")
        print(f"📊 Program ID: {launch_results['program_id']}")
        print(f"👥 Participants registered: {launch_results['participants_registered']}")
        print(f"📋 Campaigns created: {launch_results['campaigns_created']}")
        print(f"💬 Feedback campaigns: {launch_results['feedback_campaigns']}")
        print(f"🧪 Initial tests launched: {launch_results['initial_tests']}")
        print(f"📊 Monitoring: {'Active' if launch_results['monitoring_active'] else 'Inactive'}")

        print("\n📅 Next Milestones:")
        for milestone in launch_results['next_milestones']:
            print(f"  • {milestone['date']}: {milestone['description']}")

        print("\n🎯 Success Criteria:")
        for criterion, value in launcher.program_config['success_criteria'].items():
            print(f"  • {criterion.replace('_', ' ').title()}: {value}")

        # Save launch results
        with open('beta_program_launch_results.json', 'w') as f:
            json.dump(launch_results, f, indent=2, default=str)

        print("\n💾 Launch results saved to: beta_program_launch_results.json")

    else:
        print("❌ Failed to launch beta program:")
        print(f"Error: {launch_results.get('error', 'Unknown error')}")
        return 1

    # Get initial status
    print("\n📊 Getting initial program status...")
    status = await launcher.get_program_status()

    if status.get('status') != 'error':
        print("✅ Program status retrieved successfully")
        print(f"📈 Current participants: {status.get('program_status', {}).get('total_programs', 0)} programs")

        # Save status
        with open('beta_program_initial_status.json', 'w') as f:
            json.dump(status, f, indent=2, default=str)

        print("💾 Initial status saved to: beta_program_initial_status.json")

    print("\n🎉 CES Community Beta Program is now live!")
    print("Monitor progress at: beta_program_status.json")
    print("Launch results at: beta_program_launch_results.json")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)