"""
Ethical Controller - CES Ethical Guidelines and Safety Measures

Ensures all CES operations comply with ethical standards, privacy requirements,
and safety guidelines. Monitors for potential misuse and provides oversight.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime


class EthicalController:
    """
    Monitors and enforces ethical guidelines for CES operations.

    Key responsibilities:
    - Task ethics assessment
    - Privacy protection
    - Bias detection and mitigation
    - User consent management
    - Safety monitoring
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Ethical guidelines and prohibited activities
        self.prohibited_keywords = {
            'harmful': ['harm', 'damage', 'destroy', 'attack', 'exploit'],
            'illegal': ['illegal', 'unlawful', 'fraud', 'deception', 'scam'],
            'privacy': ['surveillance', 'tracking', 'monitoring', 'spy'],
            'discriminatory': ['discriminate', 'bias', 'prejudice', 'exclude']
        }

        self.logger.info("Ethical Controller initialized")

    def check_task_ethics(self, task_description: str) -> List[str]:
        """
        Check if a task raises ethical concerns

        Args:
            task_description: Description of the task to evaluate

        Returns:
            List of ethical concerns identified
        """
        concerns = []
        task_lower = task_description.lower()

        # Check for prohibited keywords
        for category, keywords in self.prohibited_keywords.items():
            for keyword in keywords:
                if keyword in task_lower:
                    concerns.append(f"Contains {category} term: '{keyword}'")

        # Check for sensitive data handling
        if any(term in task_lower for term in ['password', 'credential', 'secret', 'private']):
            concerns.append("Involves sensitive data handling")

        # Check for user manipulation
        if any(term in task_lower for term in ['manipulate', 'influence', 'persuade', 'trick']):
            concerns.append("Potential user manipulation")

        # Check for autonomous decision making in critical areas
        if any(term in task_lower for term in ['medical', 'legal', 'financial', 'safety']):
            concerns.append("Involves critical domain requiring human oversight")

        return concerns

    def approve_task(self, concerns: List[str]) -> bool:
        """
        Determine if a task can proceed despite identified concerns

        Args:
            concerns: List of ethical concerns

        Returns:
            True if task can proceed, False otherwise
        """
        if not concerns:
            return True

        # Critical concerns that always block
        critical_concerns = ['illegal', 'harmful', 'privacy violation']
        for concern in concerns:
            if any(critical in concern.lower() for critical in critical_concerns):
                self.logger.warning(f"Task blocked due to critical concern: {concern}")
                return False

        # For non-critical concerns, log warning but allow with oversight
        self.logger.warning(f"Task approved with concerns: {concerns}")
        return True

    def validate_output(self, task_description: str, output: str) -> Dict[str, Any]:
        """
        Validate the output of a task for ethical compliance

        Args:
            task_description: Original task description
            output: Generated output to validate

        Returns:
            Validation result with any issues found
        """
        issues = []
        output_lower = output.lower()

        # Check for harmful content in output
        if any(term in output_lower for term in ['harm', 'damage', 'illegal', 'unethical']):
            issues.append("Output contains potentially harmful content")

        # Check for misinformation
        if task_description.lower().count('fact') > 0 and 'uncertain' not in output_lower:
            # This is a simplified check - would need more sophisticated analysis
            pass

        # Check for privacy violations
        if any(term in output_lower for term in ['personal', 'private', 'confidential']):
            issues.append("Output may contain sensitive information")

        return {
            "approved": len(issues) == 0,
            "issues": issues,
            "validation_timestamp": datetime.now().isoformat()
        }

    def log_ethical_decision(self, task_description: str, decision: str, concerns: List[str]):
        """
        Log ethical decisions for audit purposes

        Args:
            task_description: Task that was evaluated
            decision: Decision made (approved/rejected)
            concerns: Ethical concerns identified
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task_description[:100],  # Truncate for logging
            "decision": decision,
            "concerns": concerns,
            "controller_version": "1.0"
        }

        self.logger.info(f"Ethical decision logged: {decision} - {len(concerns)} concerns")

    def get_ethical_guidelines(self) -> Dict[str, Any]:
        """
        Get current ethical guidelines and policies

        Returns:
            Dictionary containing ethical guidelines
        """
        return {
            "version": "1.0",
            "principles": [
                "Do no harm",
                "Respect user privacy",
                "Ensure transparency",
                "Prevent discrimination",
                "Maintain accountability"
            ],
            "prohibited_activities": list(self.prohibited_keywords.keys()),
            "last_updated": datetime.now().isoformat()
        }

    def assess_bias_risk(self, content: str) -> Dict[str, Any]:
        """
        Assess content for potential bias

        Args:
            content: Content to assess for bias

        Returns:
            Bias assessment results
        """
        # Placeholder for bias detection - would use ML models in production
        bias_indicators = {
            'gender_bias': ['he', 'she', 'man', 'woman', 'male', 'female'],
            'racial_bias': ['race', 'ethnic', 'color', 'origin'],
            'age_bias': ['young', 'old', 'age', 'generation']
        }

        found_indicators = {}
        content_lower = content.lower()

        for bias_type, indicators in bias_indicators.items():
            found = [ind for ind in indicators if ind in content_lower]
            if found:
                found_indicators[bias_type] = found

        return {
            "bias_detected": len(found_indicators) > 0,
            "bias_types": list(found_indicators.keys()),
            "indicators_found": found_indicators,
            "risk_level": "high" if len(found_indicators) > 2 else "medium" if found_indicators else "low"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get ethical controller status"""
        return {
            "status": "operational",
            "guidelines_version": "1.0",
            "prohibited_categories": len(self.prohibited_keywords),
            "last_check": datetime.now().isoformat()
        }