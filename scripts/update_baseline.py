#!/usr/bin/env python3
"""
Baseline Update Script for CodeSage MCP Server.

This script updates the performance baseline when significant improvements are detected.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class BaselineUpdater:
    """Updates performance baseline with improved results."""

    def __init__(self, artifacts_dir: str = "artifacts", baseline_file: str = "baseline_results.json"):
        self.artifacts_dir = Path(artifacts_dir)
        self.baseline_file = Path(baseline_file)

    def update_baseline_if_improved(self, artifacts: List[Dict[str, Any]]) -> bool:
        """Update baseline if significant improvements are detected."""
        print("Checking for baseline updates...")

        if not artifacts:
            print("No artifacts to analyze")
            return False

        # Load current baseline
        current_baseline = self._load_current_baseline()

        # Analyze improvements
        improvements = self._analyze_improvements(artifacts, current_baseline)

        if not improvements:
            print("No significant improvements detected")
            return False

        # Check if improvements warrant baseline update
        if self._should_update_baseline(improvements):
            self._update_baseline(artifacts, improvements)
            print("Baseline updated with improvements")
            return True
        else:
            print("Improvements detected but not significant enough for baseline update")
            return False

    def _load_current_baseline(self) -> Dict[str, Any]:
        """Load current baseline."""
        if not self.baseline_file.exists():
            return {"results": []}

        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except:
            return {"results": []}

    def _analyze_improvements(self, artifacts: List[Dict[str, Any]],
                            baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze improvements in the artifacts compared to baseline."""
        improvements = {
            "metrics_improved": 0,
            "significant_improvements": [],
            "overall_improvement_score": 0
        }

        # Extract current metrics
        current_metrics = {}
        for artifact in artifacts:
            if "results" in artifact and isinstance(artifact["results"], list):
                for result in artifact["results"]:
                    metric_name = result.get("metric_name", "")
                    test_name = result.get("test_name", "")
                    key = f"{test_name}_{metric_name}"
                    current_metrics[key] = result

        # Extract baseline metrics
        baseline_metrics = {}
        if "results" in baseline and isinstance(baseline["results"], list):
            for result in baseline["results"]:
                metric_name = result.get("metric_name", "")
                test_name = result.get("test_name", "")
                key = f"{test_name}_{metric_name}"
                baseline_metrics[key] = result

        # Compare metrics
        for key, current in current_metrics.items():
            baseline_result = baseline_metrics.get(key)
            if baseline_result:
                improvement = self._calculate_improvement(current, baseline_result)
                if improvement["is_improvement"]:
                    improvements["metrics_improved"] += 1
                    improvements["significant_improvements"].append({
                        "metric": key,
                        "improvement_percent": improvement["percent"],
                        "current_value": current.get("value"),
                        "baseline_value": baseline_result.get("value")
                    })

        return improvements

    def _calculate_improvement(self, current: Dict[str, Any],
                             baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement for a single metric."""
        current_value = current.get("value")
        baseline_value = baseline.get("value")

        if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
            return {"is_improvement": False, "percent": 0}

        if baseline_value == 0:
            return {"is_improvement": False, "percent": 0}

        # For most metrics, lower values are better (latency, etc.)
        # For some metrics, higher values are better (throughput, hit rates)
        metric_name = current.get("metric_name", "").lower()

        if any(term in metric_name for term in ["throughput", "hit_rate", "success", "score"]):
            # Higher is better
            percent_change = ((current_value - baseline_value) / baseline_value) * 100
            is_improvement = percent_change > 5  # 5% improvement threshold
        else:
            # Lower is better
            percent_change = ((baseline_value - current_value) / baseline_value) * 100
            is_improvement = percent_change > 5  # 5% improvement threshold

        return {
            "is_improvement": is_improvement,
            "percent": percent_change
        }

    def _should_update_baseline(self, improvements: Dict[str, Any]) -> bool:
        """Determine if baseline should be updated."""
        # Update if we have significant improvements
        return improvements["metrics_improved"] >= 3  # At least 3 improved metrics

    def _update_baseline(self, artifacts: List[Dict[str, Any]],
                        improvements: Dict[str, Any]):
        """Update the baseline with new results."""
        # Create backup of old baseline
        if self.baseline_file.exists():
            backup_file = self.baseline_file.with_suffix(f".backup_{int(datetime.now().timestamp())}")
            shutil.copy2(self.baseline_file, backup_file)

        # Create new baseline
        new_baseline = {
            "timestamp": datetime.now().isoformat(),
            "description": f"Updated baseline with {improvements['metrics_improved']} improvements",
            "improvements_summary": improvements,
            "results": []
        }

        # Collect all results from artifacts
        for artifact in artifacts:
            if "results" in artifact and isinstance(artifact["results"], list):
                new_baseline["results"].extend(artifact["results"])

        # Save new baseline
        with open(self.baseline_file, 'w') as f:
            json.dump(new_baseline, f, indent=2, default=str)

        print(f"Baseline updated. Backup saved as: {backup_file}")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python update_baseline.py <artifacts_dir>")
        sys.exit(1)

    artifacts_dir = sys.argv[1]

    # Load artifacts
    artifacts = []
    artifacts_path = Path(artifacts_dir)
    if artifacts_path.exists():
        for json_file in artifacts_path.glob("**/*.json"):
            try:
                with open(json_file, 'r') as f:
                    artifacts.append(json.load(f))
            except:
                pass

    updater = BaselineUpdater(artifacts_dir)
    updated = updater.update_baseline_if_improved(artifacts)

    if updated:
        print("✅ Baseline updated successfully")
    else:
        print("ℹ️  No baseline update needed")


if __name__ == "__main__":
    main()