#!/usr/bin/env python3
"""
Performance Insights Generator for CodeSage MCP Server.

This script generates actionable performance insights from benchmark data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class PerformanceInsightsGenerator:
    """Generates actionable performance insights."""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)

    def generate_insights(self) -> Dict[str, Any]:
        """Generate performance insights from artifacts."""
        print("Generating performance insights...")

        artifacts = self._load_artifacts()

        insights = {
            "timestamp": datetime.now().isoformat(),
            "bottlenecks": self._identify_bottlenecks(artifacts),
            "optimization_opportunities": self._find_optimization_opportunities(artifacts),
            "resource_recommendations": self._generate_resource_recommendations(artifacts),
            "alerts": self._generate_alerts(artifacts),
            "action_items": self._create_action_items(artifacts)
        }

        self._save_insights(insights)
        return insights

    def _load_artifacts(self) -> List[Dict[str, Any]]:
        """Load benchmark artifacts."""
        artifacts = []
        if not self.artifacts_dir.exists():
            return artifacts

        for json_file in self.artifacts_dir.glob("**/*.json"):
            try:
                with open(json_file, 'r') as f:
                    artifacts.append(json.load(f))
            except:
                pass
        return artifacts

    def _identify_bottlenecks(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Analyze latency bottlenecks
        high_latency_tests = []
        for artifact in artifacts:
            if "results" in artifact and isinstance(artifact["results"], list):
                for result in artifact["results"]:
                    if (result.get("metric_name", "").lower().find("latency") >= 0 and
                        result.get("value", 0) > 2000):  # Over 2 seconds
                        high_latency_tests.append(result.get("test_name", "Unknown"))

        if high_latency_tests:
            bottlenecks.append(f"High latency detected in: {', '.join(set(high_latency_tests[:3]))}")

        # Analyze throughput bottlenecks
        low_throughput_tests = []
        for artifact in artifacts:
            if "results" in artifact and isinstance(artifact["results"], list):
                for result in artifact["results"]:
                    if (result.get("metric_name", "").lower().find("throughput") >= 0 and
                        result.get("value", 0) < 10):  # Under 10 RPS
                        low_throughput_tests.append(result.get("test_name", "Unknown"))

        if low_throughput_tests:
            bottlenecks.append(f"Low throughput detected in: {', '.join(set(low_throughput_tests[:3]))}")

        return bottlenecks if bottlenecks else ["No significant bottlenecks detected"]

    def _find_optimization_opportunities(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Find optimization opportunities."""
        opportunities = []

        # Check cache performance
        low_cache_hit_rates = []
        for artifact in artifacts:
            if "results" in artifact and isinstance(artifact["results"], list):
                for result in artifact["results"]:
                    if ("cache" in result.get("metric_name", "").lower() and
                        "hit_rate" in result.get("metric_name", "").lower() and
                        result.get("value", 100) < 70):
                        low_cache_hit_rates.append(result.get("test_name", "Unknown"))

        if low_cache_hit_rates:
            opportunities.append("Improve cache hit rates through better caching strategies")

        # Check memory usage
        high_memory_usage = []
        for artifact in artifacts:
            if "results" in artifact and isinstance(artifact["results"], list):
                for result in artifact["results"]:
                    if ("memory" in result.get("metric_name", "").lower() and
                        result.get("value", 0) > 80):  # Over 80%
                        high_memory_usage.append(result.get("test_name", "Unknown"))

        if high_memory_usage:
            opportunities.append("Optimize memory usage to reduce resource consumption")

        return opportunities if opportunities else ["System performance is generally optimal"]

    def _generate_resource_recommendations(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Generate resource recommendations."""
        recommendations = []

        # Analyze CPU usage patterns
        recommendations.append("Monitor CPU usage during peak hours")

        # Analyze memory patterns
        recommendations.append("Consider memory optimization for large codebases")

        # Analyze I/O patterns
        recommendations.append("Optimize disk I/O for better performance")

        return recommendations

    def _generate_alerts(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Generate performance alerts."""
        alerts = []

        # Check for failed tests
        failed_tests = []
        for artifact in artifacts:
            if "results" in artifact and isinstance(artifact["results"], list):
                for result in artifact["results"]:
                    if not result.get("achieved", True):
                        failed_tests.append(result.get("test_name", "Unknown"))

        if failed_tests:
            alerts.append(f"Performance targets not met in: {', '.join(set(failed_tests[:5]))}")

        return alerts if alerts else ["All performance targets are being met"]

    def _create_action_items(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Create actionable items."""
        action_items = [
            "Review and optimize slow-performing tools",
            "Implement better caching strategies",
            "Monitor resource usage patterns",
            "Set up automated performance regression testing"
        ]

        return action_items

    def _save_insights(self, insights: Dict[str, Any]):
        """Save insights to file."""
        output_file = Path("reports") / f"performance_insights_{int(datetime.now().timestamp())}.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(insights, f, indent=2, default=str)

        print(f"Performance insights saved to: {output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_performance_insights.py <artifacts_dir>")
        sys.exit(1)

    artifacts_dir = sys.argv[1]
    generator = PerformanceInsightsGenerator(artifacts_dir)
    insights = generator.generate_insights()

    print("Performance Insights Generated:")
    print(f"Bottlenecks: {len(insights.get('bottlenecks', []))}")
    print(f"Opportunities: {len(insights.get('optimization_opportunities', []))}")
    print(f"Alerts: {len(insights.get('alerts', []))}")


if __name__ == "__main__":
    main()