#!/usr/bin/env python3
"""
Simple CES Community Beta Program Launcher - Phase 5

A simplified launcher for the CES community beta program that avoids
circular import issues and focuses on core beta program functionality.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only the beta testing framework directly
from ces.core.beta_testing_framework import BetaTestingFramework, TestScenario


class SimpleBetaLauncher:
    """Simplified launcher for CES Community Beta Program"""

    def __init__(self):
        self.beta_framework = BetaTestingFramework()
        self.program_config = {
            'name': 'CES Community Beta Program - Phase 5 Launch',
            'description': 'Enterprise-grade beta testing for CES public launch',
            'target_participants': 20,
            'min_participants': 10,
            'success_criteria': {
                'engagement_rate': 0.75,
                'feedback_completion_rate': 0.85,
                'satisfaction_score': 4.2
            }
        }

    async def launch_program(self) -> Dict[str, Any]:
        """Launch the beta program"""
        logger.info("🚀 Launching CES Community Beta Program...")

        try:
            # 1. Create community beta program
            program_id = await self.beta_framework.create_community_beta_program(
                name=self.program_config['name'],
                description=self.program_config['description'],
                target_participants=self.program_config['target_participants'],
                success_criteria=self.program_config['success_criteria']
            )

            # 2. Register initial participants
            participants = await self._register_participants(program_id)

            # 3. Create test scenarios
            test_scenarios = await self._create_test_scenarios()

            # 4. Setup monitoring
            monitoring = await self._setup_monitoring(program_id)

            launch_results = {
                'program_id': program_id,
                'status': 'launched',
                'participants': len(participants),
                'test_scenarios': len(test_scenarios),
                'monitoring': monitoring,
                'launch_timestamp': datetime.now().isoformat(),
                'next_steps': [
                    'Send welcome emails to participants',
                    'Setup feedback collection channels',
                    'Monitor engagement metrics',
                    'Schedule weekly check-ins'
                ]
            }

            logger.info(f"✅ Beta program launched with {len(participants)} participants")
            return launch_results

        except Exception as e:
            logger.error(f"❌ Launch failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def _register_participants(self, program_id: str) -> List[Dict[str, Any]]:
        """Register beta participants"""
        participants_data = [
            {'email': 'alice@techcorp.com', 'name': 'Alice Johnson', 'company': 'TechCorp', 'role': 'Developer'},
            {'email': 'bob@startup.io', 'name': 'Bob Smith', 'company': 'StartupXYZ', 'role': 'Architect'},
            {'email': 'carol@enterprise.org', 'name': 'Carol Williams', 'company': 'Enterprise Inc', 'role': 'Manager'},
            {'email': 'david@consulting.com', 'name': 'David Brown', 'company': 'Consulting Pro', 'role': 'Analyst'},
            {'email': 'eve@university.edu', 'name': 'Eve Davis', 'company': 'State University', 'role': 'Researcher'},
            {'email': 'frank@independent.dev', 'name': 'Frank Miller', 'company': 'Independent', 'role': 'Freelancer'},
            {'email': 'grace@qa-team.com', 'name': 'Grace Wilson', 'company': 'QA Solutions', 'role': 'QA Lead'},
            {'email': 'henry@product-team.io', 'name': 'Henry Taylor', 'company': 'ProductCo', 'role': 'Product Manager'},
            {'email': 'iris@design-studio.com', 'name': 'Iris Anderson', 'company': 'Design Studio', 'role': 'UX Designer'},
            {'email': 'jack@devops-team.net', 'name': 'Jack Thomas', 'company': 'DevOps Pro', 'role': 'DevOps Engineer'},
            {'email': 'kate@leadership.org', 'name': 'Kate Jackson', 'company': 'Leadership Inc', 'role': 'Tech Lead'},
            {'email': 'liam@university.edu', 'name': 'Liam White', 'company': 'Tech University', 'role': 'Student'},
            {'email': 'mia@newco.io', 'name': 'Mia Garcia', 'company': 'NewCo', 'role': 'Founder'},
            {'email': 'noah@advice.com', 'name': 'Noah Martinez', 'company': 'Tech Advisors', 'role': 'Consultant'},
            {'email': 'olivia@tech-blog.com', 'name': 'Olivia Robinson', 'company': 'Tech Blog', 'role': 'Writer'}
        ]

        registered = []

        for i, participant in enumerate(participants_data):
            try:
                result = await self.beta_framework.register_beta_participant(
                    program_id=program_id,
                    user_id=f"beta_user_{i+1:03d}",
                    email=participant['email'],
                    name=participant['name'],
                    company=participant['company'],
                    role=participant['role']
                )

                if result['status'] == 'registered':
                    registered.append({
                        'user_id': result['participant_id'],
                        'name': participant['name'],
                        'email': participant['email'],
                        'welcome_message': result.get('welcome_message', '')
                    })
                    logger.info(f"✓ Registered: {participant['name']}")

            except Exception as e:
                logger.error(f"✗ Failed to register {participant['email']}: {e}")

        return registered

    async def _create_test_scenarios(self) -> List[str]:
        """Create beta test scenarios"""
        scenarios = []

        # Community engagement test
        test_id = await self.beta_framework.create_beta_test(
            scenario=TestScenario.COMMUNITY_ENGAGEMENT,
            participants=[f"beta_user_{i:03d}" for i in range(1, 16)]
        )
        scenarios.append(test_id)

        # Enterprise features test
        test_id = await self.beta_framework.create_beta_test(
            scenario=TestScenario.ENTERPRISE_FEATURES,
            participants=[f"beta_user_{i:03d}" for i in range(8, 21)]
        )
        scenarios.append(test_id)

        # User onboarding test
        test_id = await self.beta_framework.create_beta_test(
            scenario=TestScenario.USER_ONBOARDING,
            participants=[f"beta_user_{i:03d}" for i in range(1, 11)]
        )
        scenarios.append(test_id)

        logger.info(f"✓ Created {len(scenarios)} test scenarios")
        return scenarios

    async def _setup_monitoring(self, program_id: str) -> Dict[str, Any]:
        """Setup monitoring for the beta program"""
        try:
            status = await self.beta_framework.get_community_beta_status(program_id)

            return {
                'status': 'active',
                'program_status': status,
                'monitoring_metrics': [
                    'participant_engagement',
                    'feedback_volume',
                    'test_completion_rate',
                    'satisfaction_scores'
                ],
                'alerts_configured': True
            }

        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """Get current program status"""
        try:
            status = await self.beta_framework.get_community_beta_status()
            return {
                'status': 'active',
                'program_info': status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


async def main():
    """Main launcher function"""
    launcher = SimpleBetaLauncher()

    print("🚀 CES Community Beta Program Launcher - Phase 5")
    print("=" * 60)

    # Launch program
    results = await launcher.launch_program()

    if results['status'] == 'launched':
        print("✅ CES Beta Program launched successfully!")
        print(f"📊 Program ID: {results['program_id']}")
        print(f"👥 Participants registered: {results['participants']}")
        print(f"🧪 Test scenarios created: {results['test_scenarios']}")
        print(f"📊 Monitoring: {results['monitoring']['status']}")

        print("\n📅 Next Steps:")
        for step in results['next_steps']:
            print(f"  • {step}")

        print("\n🎯 Success Criteria:")
        for criterion, value in launcher.program_config['success_criteria'].items():
            print(f"  • {criterion.replace('_', ' ').title()}: {value}")

        # Save results
        with open('beta_launch_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 Results saved to: beta_launch_results.json")

        # Get status
        status = await launcher.get_status()
        with open('beta_program_status.json', 'w') as f:
            json.dump(status, f, indent=2, default=str)

        print("📊 Status saved to: beta_program_status.json")

    else:
        print("❌ Launch failed:")
        print(f"Error: {results.get('error', 'Unknown error')}")
        return 1

    print("\n🎉 CES Community Beta Program is now live!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)