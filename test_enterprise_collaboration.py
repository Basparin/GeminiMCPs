#!/usr/bin/env python3
"""
Test script for CES Enterprise Collaboration features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ces.collaborative.enterprise_collaboration import get_enterprise_collaboration, TeamRole
from ces.collaborative.session_manager import SessionManager

def test_enterprise_collaboration():
    """Test enterprise collaboration features."""
    print("🧪 Testing CES Enterprise Collaboration Features")
    print("=" * 50)

    # Initialize components
    session_manager = SessionManager()
    enterprise_collab = get_enterprise_collaboration(session_manager)

    # Test project creation
    print("\n📁 Testing Project Creation...")
    project_data = {
        "name": "CES Enterprise Project",
        "description": "Testing enterprise collaboration features",
        "status": "active",
        "security_level": "confidential",
        "metadata": {"department": "engineering", "priority": "high"}
    }

    project_id = enterprise_collab.create_project(project_data, "user1")
    print(f"✅ Project created: {project_id}")

    # Test team member addition
    print("\n👥 Testing Team Member Management...")
    success = enterprise_collab.add_team_member(project_id, "user2", TeamRole.DEVELOPER, "user1")
    print(f"✅ Team member added: {success}")

    # Test security clearance
    print("\n🔒 Testing Security Features...")
    has_clearance = enterprise_collab.check_security_clearance(project_id, "user2")
    print(f"✅ Security clearance check: {has_clearance}")

    # Test project analytics
    print("\n📊 Testing Project Analytics...")
    analytics = enterprise_collab.get_project_analytics(project_id, "user1")
    print(f"✅ Project analytics retrieved: {analytics['project_name']}")

    # Test team productivity report
    print("\n📈 Testing Team Productivity...")
    productivity = enterprise_collab.get_team_productivity_report(project_id, "user1", days=7)
    print(f"✅ Team productivity report: {productivity['total_activity_events']} events")

    print("\n🎉 Enterprise Collaboration Testing Complete!")
    return True

if __name__ == "__main__":
    test_enterprise_collaboration()