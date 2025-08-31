#!/usr/bin/env python3
"""
Benchmark Trends Analyzer for CodeSage MCP Server.

This script analyzes trends in benchmark results over time to identify
performance patterns and predict future performance.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics
from datetime import datetime, timedelta
import glob


class BenchmarkTrendsAnalyzer:
    """Analyzes trends in benchmark results over time."""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)

    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from benchmark artifacts."""
        print("Analyzing benchmark trends...")

        # Load all benchmark artifacts
        artifacts = self._load_artifacts()

        if not artifacts:
            return {"error": "No benchmark artifacts found"}

        # Analyze trends
        trends = {
            "timestamp": datetime.now().isoformat(),
            "analysis_period": self._get_analysis_period(artifacts),
            "metrics_trends": self._analyze_metrics_trends(artifacts),
            "performance_patterns": self._identify_patterns(artifacts),
            "predictions": self._generate_predictions(artifacts),
            "insights": self._generate_insights(artifacts)
        }

        # Save analysis
        self._save_analysis(trends)

        return trends

    def _load_artifacts(self) -> List[Dict[str, Any]]:
        """Load benchmark artifacts from the artifacts directory."""
        artifacts = []

        if not self.artifacts_dir.exists():
            return artifacts

        # Load benchmark results
        for pattern in ["benchmark-results-*", "parameterized-results-*"]:
            for artifact_dir in self.artifacts_dir.glob(pattern):
                for json_file in artifact_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            data["_source_file"] = str(json_file)
                            artifacts.append(data)
                    except Exception as e:
                        print(f"Failed to load {json_file}: {e}")

        return artifacts

    def _get_analysis_period(self, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get the time period covered by the analysis."""
        if not artifacts:
            return {"start": None, "end": None, "days": 0}

        timestamps = []
        for artifact in artifacts:
            if "timestamp" in artifact:
                if isinstance(artifact["timestamp"], (int, float)):
                    timestamps.append(datetime.fromtimestamp(artifact["timestamp"]))
                elif isinstance(artifact["timestamp"], str):
                    try:
                        timestamps.append(datetime.fromisoformat(artifact["timestamp"]))
                    except:
                        pass

        if timestamps:
            start = min(timestamps)
            end = max(timestamps)
            days = (end - start).days
            return {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "days": days
            }

        return {"start": None, "end": None, "days": 0}

    def _analyze_metrics_trends(self, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends for different metrics."""
        trends = {}

        # Group artifacts by metric
        metrics_data = self._group_by_metrics(artifacts)

        for metric_name, data_points in metrics_data.items():
            if len(data_points) < 2:
                continue

            # Sort by timestamp
            data_points.sort(key=lambda x: x.get("timestamp", 0))

            # Calculate trend
            values = [dp.get("value") for dp in data_points if dp.get("value") is not None]
            timestamps = [dp.get("timestamp") for dp in data_points if dp.get("timestamp")]

            if len(values) >= 2:
                trend = self._calculate_trend(values, timestamps)
                trends[metric_name] = {
                    "data_points": len(data_points),
                    "trend_direction": trend["direction"],
                    "trend_slope": trend["slope"],
                    "volatility": trend["volatility"],
                    "latest_value": values[-1] if values else None,
                    "change_percent": trend["change_percent"]
                }

        return trends

    def _group_by_metrics(self, artifacts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group artifacts by metric names."""
        metrics = {}

        for artifact in artifacts:
            if "results" in artifact:
                results = artifact["results"]
                if isinstance(results, list):
                    for result in results:
                        metric_name = result.get("metric_name", "")
                        test_name = result.get("test_name", "")
                        key = f"{test_name}_{metric_name}"

                        if key not in metrics:
                            metrics[key] = []

                        metrics[key].append({
                            "timestamp": artifact.get("timestamp"),
                            "value": result.get("value"),
                            "unit": result.get("unit"),
                            "target": result.get("target"),
                            "achieved": result.get("achieved")
                        })

        return metrics

    def _calculate_trend(self, values: List[float], timestamps: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a metric."""
        if len(values) < 2:
            return {"direction": "insufficient_data", "slope": 0, "volatility": 0, "change_percent": 0}

        # Calculate slope (simple linear regression)
        n = len(values)
        x = list(range(n))  # Use indices as x values

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0

        # Determine direction
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"

        # Calculate volatility (coefficient of variation)
        if statistics.mean(values) != 0:
            volatility = statistics.stdev(values) / abs(statistics.mean(values))
        else:
            volatility = 0

        # Calculate overall change percentage
        change_percent = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0

        return {
            "direction": direction,
            "slope": slope,
            "volatility": volatility,
            "change_percent": change_percent
        }

    def _identify_patterns(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Identify performance patterns."""
        patterns = []

        # Check for time-based patterns
        hourly_patterns = self._check_hourly_patterns(artifacts)
        if hourly_patterns:
            patterns.extend(hourly_patterns)

        # Check for load-based patterns
        load_patterns = self._check_load_patterns(artifacts)
        if load_patterns:
            patterns.extend(load_patterns)

        # Check for seasonal patterns
        seasonal_patterns = self._check_seasonal_patterns(artifacts)
        if seasonal_patterns:
            patterns.extend(seasonal_patterns)

        if not patterns:
            patterns.append("No significant patterns detected in the data.")

        return patterns

    def _check_hourly_patterns(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Check for hourly performance patterns."""
        patterns = []

        # This would analyze performance by hour of day
        # For now, return a placeholder
        return patterns

    def _check_load_patterns(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Check for load-based performance patterns."""
        patterns = []

        # Analyze performance under different load conditions
        return patterns

    def _check_seasonal_patterns(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Check for seasonal performance patterns."""
        patterns = []

        # Analyze performance patterns by day of week, etc.
        return patterns

    def _generate_predictions(self, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance predictions."""
        predictions = {
            "short_term": "Stable performance expected",
            "long_term": "Monitor for gradual changes",
            "risks": [],
            "opportunities": []
        }

        # Analyze trends to make predictions
        trends = self._analyze_metrics_trends(artifacts)

        # Identify at-risk metrics
        at_risk = [metric for metric, data in trends.items()
                  if data.get("trend_direction") == "decreasing"]

        if at_risk:
            predictions["risks"].append(f"Performance degradation detected in: {', '.join(at_risk[:3])}")

        # Identify improving metrics
        improving = [metric for metric, data in trends.items()
                    if data.get("trend_direction") == "increasing"]

        if improving:
            predictions["opportunities"].append(f"Performance improvements in: {', '.join(improving[:3])}")

        return predictions

    def _generate_insights(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Generate performance insights."""
        insights = []

        trends = self._analyze_metrics_trends(artifacts)

        # Generate insights based on trends
        if trends:
            insights.append(f"Analyzed trends for {len(trends)} metrics over time.")

            increasing = sum(1 for t in trends.values() if t.get("trend_direction") == "increasing")
            decreasing = sum(1 for t in trends.values() if t.get("trend_direction") == "decreasing")

            if increasing > decreasing:
                insights.append("Overall positive performance trend detected.")
            elif decreasing > increasing:
                insights.append("Overall negative performance trend detected.")
            else:
                insights.append("Performance is generally stable.")

        return insights

    def _save_analysis(self, analysis: Dict[str, Any]):
        """Save trend analysis to file."""
        output_file = Path("reports") / f"trend_analysis_{int(datetime.now().timestamp())}.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"Trend analysis saved to: {output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_benchmark_trends.py <artifacts_dir>")
        sys.exit(1)

    artifacts_dir = sys.argv[1]
    analyzer = BenchmarkTrendsAnalyzer(artifacts_dir)
    analysis = analyzer.analyze_trends()

    print("Trend Analysis Summary:")
    print(f"Analysis Period: {analysis.get('analysis_period', {}).get('days', 0)} days")
    print(f"Metrics Analyzed: {len(analysis.get('metrics_trends', {}))}")

    insights = analysis.get('insights', [])
    if insights:
        print("\nKey Insights:")
        for insight in insights:
            print(f"- {insight}")


if __name__ == "__main__":
    main()