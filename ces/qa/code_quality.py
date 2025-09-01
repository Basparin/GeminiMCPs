"""CES Code Quality Analyzer.

Provides comprehensive code quality analysis including PEP 8 compliance,
type hints validation, documentation coverage, and complexity analysis.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import radon.complexity as radon_complexity
    import radon.metrics as radon_metrics
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

try:
    import ruff
    RUFF_AVAILABLE = True
except ImportError:
    RUFF_AVAILABLE = False


@dataclass
class CodeQualityMetrics:
    """Code quality metrics for a file or module."""
    file_path: str
    pep8_compliance: float  # Percentage of PEP 8 compliance
    type_hint_coverage: float  # Percentage of functions with type hints
    documentation_coverage: float  # Percentage of functions/classes with docstrings
    complexity_score: float  # Average complexity score
    maintainability_index: float  # Code maintainability index
    issues: List[Dict[str, Any]]  # List of identified issues


@dataclass
class CodeQualityReport:
    """Comprehensive code quality report."""
    overall_score: float
    files_analyzed: int
    total_lines: int
    metrics: Dict[str, CodeQualityMetrics]
    summary: Dict[str, Any]
    recommendations: List[str]


class CodeQualityAnalyzer:
    """Analyzes code quality across the CES codebase."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)
        self.targets = {
            'pep8_compliance': 95.0,  # 95% PEP 8 compliance
            'type_hint_coverage': 100.0,  # 100% type hint coverage
            'documentation_coverage': 100.0,  # 100% documentation coverage
            'max_complexity': 10.0,  # Maximum complexity score
            'min_maintainability': 70.0  # Minimum maintainability index
        }

    def analyze_codebase(self, include_patterns: Optional[List[str]] = None,
                        exclude_patterns: Optional[List[str]] = None) -> CodeQualityReport:
        """Analyze the entire codebase for quality metrics."""
        if include_patterns is None:
            include_patterns = ['*.py']
        if exclude_patterns is None:
            exclude_patterns = ['test_*', '*_test.py', '__pycache__', '.git', 'venv', 'node_modules']

        python_files = self._find_python_files(include_patterns, exclude_patterns)

        metrics = {}
        total_lines = 0

        for file_path in python_files:
            try:
                file_metrics = self.analyze_file(file_path)
                metrics[str(file_path)] = file_metrics
                total_lines += file_metrics.issues[0].get('lines', 0) if file_metrics.issues else 0
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue

        # Calculate overall scores
        overall_score = self._calculate_overall_score(metrics)
        summary = self._generate_summary(metrics)
        recommendations = self._generate_recommendations(metrics)

        return CodeQualityReport(
            overall_score=overall_score,
            files_analyzed=len(metrics),
            total_lines=total_lines,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations
        )

    def analyze_file(self, file_path: Path) -> CodeQualityMetrics:
        """Analyze a single Python file for quality metrics."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse AST for analysis
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return CodeQualityMetrics(
                file_path=str(file_path),
                pep8_compliance=0.0,
                type_hint_coverage=0.0,
                documentation_coverage=0.0,
                complexity_score=100.0,
                maintainability_index=0.0,
                issues=[{'type': 'syntax_error', 'message': 'File contains syntax errors', 'lines': len(content.splitlines())}]
            )

        # Analyze different aspects
        pep8_score = self._check_pep8_compliance(content)
        type_hint_score = self._check_type_hints(tree)
        doc_score = self._check_documentation(tree)
        complexity_score = self._check_complexity(content)
        maintainability = self._check_maintainability(content)

        # Collect issues
        issues = []
        issues.extend(self._get_pep8_issues(content))
        issues.extend(self._get_type_hint_issues(tree))
        issues.extend(self._get_doc_issues(tree))
        issues.extend(self._get_complexity_issues(content))

        return CodeQualityMetrics(
            file_path=str(file_path),
            pep8_compliance=pep8_score,
            type_hint_coverage=type_hint_score,
            documentation_coverage=doc_score,
            complexity_score=complexity_score,
            maintainability_index=maintainability,
            issues=issues
        )

    def _find_python_files(self, include_patterns: List[str], exclude_patterns: List[str]) -> List[Path]:
        """Find all Python files matching the patterns."""
        python_files = []

        for pattern in include_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file() and not self._should_exclude(file_path, exclude_patterns):
                    python_files.append(file_path)

        return python_files

    def _should_exclude(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded based on patterns."""
        file_str = str(file_path)

        for pattern in exclude_patterns:
            if pattern in file_str:
                return True

        return False

    def _check_pep8_compliance(self, content: str) -> float:
        """Check PEP 8 compliance using available tools."""
        if RUFF_AVAILABLE:
            return self._check_ruff_compliance(content)
        else:
            # Basic PEP 8 checks
            return self._basic_pep8_check(content)

    def _check_ruff_compliance(self, content: str) -> float:
        """Check PEP 8 compliance using Ruff."""
        try:
            # Run ruff check on content
            result = subprocess.run(
                ['ruff', 'check', '--stdin-filename', 'temp.py'],
                input=content,
                text=True,
                capture_output=True,
                timeout=30
            )

            # Count violations
            violations = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            total_lines = len(content.splitlines())

            # Calculate compliance percentage
            if total_lines == 0:
                return 100.0

            # Assume 1 violation per 10 lines is acceptable
            expected_violations = max(1, total_lines // 10)
            if violations <= expected_violations:
                return 100.0
            else:
                return max(0.0, 100.0 - (violations - expected_violations) * 5.0)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self._basic_pep8_check(content)

    def _basic_pep8_check(self, content: str) -> float:
        """Basic PEP 8 compliance check."""
        lines = content.splitlines()
        violations = 0

        for i, line in enumerate(lines, 1):
            # Check line length (PEP 8: max 79 characters)
            if len(line) > 79:
                violations += 1

            # Check trailing whitespace
            if line.rstrip() != line:
                violations += 1

            # Check multiple spaces (should use 4 spaces for indentation)
            if '\t' in line:
                violations += 1

        total_lines = len(lines)
        if total_lines == 0:
            return 100.0

        # Calculate compliance
        violation_rate = violations / total_lines
        return max(0.0, 100.0 - violation_rate * 200.0)  # 2 violations per line = 0%

    def _check_type_hints(self, tree: ast.AST) -> float:
        """Check type hint coverage in the AST."""
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)

        total_functions = len(functions)
        if total_functions == 0:
            return 100.0

        functions_with_hints = 0
        for func in functions:
            has_hints = (
                func.returns is not None or
                any(arg.annotation is not None for arg in func.args.args)
            )
            if has_hints:
                functions_with_hints += 1

        return (functions_with_hints / total_functions) * 100.0

    def _check_documentation(self, tree: ast.AST) -> float:
        """Check documentation coverage in the AST."""
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)

        total_items = len(functions) + len(classes)
        if total_items == 0:
            return 100.0

        documented_items = 0
        for item in functions + classes:
            if ast.get_docstring(item):
                documented_items += 1

        return (documented_items / total_items) * 100.0

    def _check_complexity(self, content: str) -> float:
        """Check code complexity using radon if available."""
        if not RADON_AVAILABLE:
            return 5.0  # Default moderate complexity

        try:
            # Calculate average complexity
            complexity_results = radon_complexity.cc_visit(content)
            if not complexity_results:
                return 1.0

            complexities = [result.complexity for result in complexity_results]
            return sum(complexities) / len(complexities)

        except Exception:
            return 5.0

    def _check_maintainability(self, content: str) -> float:
        """Check code maintainability index using radon if available."""
        if not RADON_AVAILABLE:
            return 75.0  # Default good maintainability

        try:
            mi = radon_metrics.mi_visit(content, multi=True)
            return mi

        except Exception:
            return 75.0

    def _get_pep8_issues(self, content: str) -> List[Dict[str, Any]]:
        """Get detailed PEP 8 issues."""
        issues = []
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            if len(line) > 79:
                issues.append({
                    'type': 'pep8',
                    'category': 'line_length',
                    'line': i,
                    'message': f'Line too long ({len(line)} > 79 characters)',
                    'severity': 'warning'
                })

            if line.rstrip() != line:
                issues.append({
                    'type': 'pep8',
                    'category': 'trailing_whitespace',
                    'line': i,
                    'message': 'Trailing whitespace',
                    'severity': 'warning'
                })

            if '\t' in line:
                issues.append({
                    'type': 'pep8',
                    'category': 'indentation',
                    'line': i,
                    'message': 'Use spaces instead of tabs',
                    'severity': 'error'
                })

        return issues

    def _get_type_hint_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Get type hint issues."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_hints = (
                    node.returns is not None or
                    any(arg.annotation is not None for arg in node.args.args)
                )

                if not has_hints:
                    issues.append({
                        'type': 'type_hints',
                        'category': 'missing_hints',
                        'line': node.lineno,
                        'message': f'Function {node.name} missing type hints',
                        'severity': 'warning'
                    })

        return issues

    def _get_doc_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Get documentation issues."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    node_type = 'function' if isinstance(node, ast.FunctionDef) else 'class'
                    issues.append({
                        'type': 'documentation',
                        'category': 'missing_docstring',
                        'line': node.lineno,
                        'message': f'{node_type.capitalize()} {node.name} missing docstring',
                        'severity': 'warning'
                    })

        return issues

    def _get_complexity_issues(self, content: str) -> List[Dict[str, Any]]:
        """Get complexity issues."""
        issues = []

        if RADON_AVAILABLE:
            try:
                complexity_results = radon_complexity.cc_visit(content)
                for result in complexity_results:
                    if result.complexity > self.targets['max_complexity']:
                        issues.append({
                            'type': 'complexity',
                            'category': 'high_complexity',
                            'line': result.lineno,
                            'message': f'{result.name} has complexity {result.complexity} > {self.targets["max_complexity"]}',
                            'severity': 'warning'
                        })
            except Exception:
                pass

        return issues

    def _calculate_overall_score(self, metrics: Dict[str, CodeQualityMetrics]) -> float:
        """Calculate overall code quality score."""
        if not metrics:
            return 0.0

        scores = []
        for file_metrics in metrics.values():
            # Weight different aspects
            pep8_weight = 0.2
            type_hint_weight = 0.25
            doc_weight = 0.25
            complexity_weight = 0.15
            maintainability_weight = 0.15

            # Normalize complexity score (lower is better)
            normalized_complexity = max(0, 100 - file_metrics.complexity_score * 10)

            # Normalize maintainability (already 0-100)
            normalized_maintainability = file_metrics.maintainability_index

            file_score = (
                file_metrics.pep8_compliance * pep8_weight +
                file_metrics.type_hint_coverage * type_hint_weight +
                file_metrics.documentation_coverage * doc_weight +
                normalized_complexity * complexity_weight +
                normalized_maintainability * maintainability_weight
            )

            scores.append(file_score)

        return sum(scores) / len(scores) if scores else 0.0

    def _generate_summary(self, metrics: Dict[str, CodeQualityMetrics]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not metrics:
            return {}

        pep8_scores = [m.pep8_compliance for m in metrics.values()]
        type_hint_scores = [m.type_hint_coverage for m in metrics.values()]
        doc_scores = [m.documentation_coverage for m in metrics.values()]
        complexity_scores = [m.complexity_score for m in metrics.values()]
        maintainability_scores = [m.maintainability_index for m in metrics.values()]

        total_issues = sum(len(m.issues) for m in metrics.values())

        return {
            'average_pep8_compliance': sum(pep8_scores) / len(pep8_scores),
            'average_type_hint_coverage': sum(type_hint_scores) / len(type_hint_scores),
            'average_documentation_coverage': sum(doc_scores) / len(doc_scores),
            'average_complexity': sum(complexity_scores) / len(complexity_scores),
            'average_maintainability': sum(maintainability_scores) / len(maintainability_scores),
            'total_issues': total_issues,
            'files_with_issues': sum(1 for m in metrics.values() if m.issues),
            'target_compliance': {
                'pep8_target': self.targets['pep8_compliance'],
                'type_hint_target': self.targets['type_hint_coverage'],
                'documentation_target': self.targets['documentation_coverage'],
                'max_complexity_target': self.targets['max_complexity'],
                'min_maintainability_target': self.targets['min_maintainability']
            }
        }

    def _generate_recommendations(self, metrics: Dict[str, CodeQualityMetrics]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        summary = self._generate_summary(metrics)

        if summary.get('average_pep8_compliance', 0) < self.targets['pep8_compliance']:
            recommendations.append("Improve PEP 8 compliance by running code formatters like black or ruff")

        if summary.get('average_type_hint_coverage', 0) < self.targets['type_hint_coverage']:
            recommendations.append("Add type hints to functions and methods to improve code clarity and catch errors")

        if summary.get('average_documentation_coverage', 0) < self.targets['documentation_coverage']:
            recommendations.append("Add comprehensive docstrings to all public functions and classes")

        if summary.get('average_complexity', 0) > self.targets['max_complexity']:
            recommendations.append("Refactor complex functions by breaking them into smaller, more focused functions")

        if summary.get('average_maintainability', 0) < self.targets['min_maintainability']:
            recommendations.append("Improve code maintainability by reducing complexity and improving structure")

        if summary.get('total_issues', 0) > 0:
            recommendations.append(f"Address {summary['total_issues']} identified issues across {summary['files_with_issues']} files")

        return recommendations