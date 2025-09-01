"""CES QA Report Generator.

Generates comprehensive QA compliance reports combining all quality assurance
components into a production-ready Phase 1 validation document.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from .code_quality import CodeQualityReport
from .test_framework import TestSuiteResult
from .performance_validator import PerformanceValidationReport
from .security_analyzer import SecurityComplianceReport
from .accessibility_checker import AccessibilityComplianceReport
from .risk_mitigator import RiskMitigationReport
from .context_validator import ContextValidationReport
from .compliance_engine import ComplianceReport
from .monitoring_system import MonitoringReport


@dataclass
class QAComplianceSummary:
    """Overall QA compliance summary."""
    phase: str
    overall_compliance: bool
    compliance_score: float
    critical_issues: int
    high_priority_issues: int
    medium_priority_issues: int
    low_priority_issues: int
    total_issues: int
    recommendations_count: int
    assessment_date: datetime


@dataclass
class QASectionReport:
    """Individual QA section report."""
    section_name: str
    compliance_status: str  # 'compliant', 'non_compliant', 'partial', 'not_assessed'
    score: float
    issues_count: int
    critical_issues: int
    recommendations: List[str]
    key_findings: List[str]
    data: Dict[str, Any]


class QAReportGenerator:
    """Generates comprehensive QA compliance reports for CES Phase 1."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)

    def generate_phase1_qa_report(self,
                                 code_quality_report: Optional[CodeQualityReport] = None,
                                 test_results: Optional[Dict[str, TestSuiteResult]] = None,
                                 performance_report: Optional[PerformanceValidationReport] = None,
                                 security_report: Optional[SecurityComplianceReport] = None,
                                 accessibility_report: Optional[AccessibilityComplianceReport] = None,
                                 risk_report: Optional[RiskMitigationReport] = None,
                                 context_report: Optional[ContextValidationReport] = None,
                                 compliance_report: Optional[ComplianceReport] = None,
                                 monitoring_report: Optional[MonitoringReport] = None) -> str:
        """Generate comprehensive Phase 1 QA compliance report."""

        # Generate individual section reports
        sections = []

        if code_quality_report:
            sections.append(self._generate_code_quality_section(code_quality_report))

        if test_results:
            sections.append(self._generate_testing_section(test_results))

        if performance_report:
            sections.append(self._generate_performance_section(performance_report))

        if security_report:
            sections.append(self._generate_security_section(security_report))

        if accessibility_report:
            sections.append(self._generate_accessibility_section(accessibility_report))

        if risk_report:
            sections.append(self._generate_risk_mitigation_section(risk_report))

        if context_report:
            sections.append(self._generate_context_management_section(context_report))

        if compliance_report:
            sections.append(self._generate_compliance_engine_section(compliance_report))

        if monitoring_report:
            sections.append(self._generate_monitoring_section(monitoring_report))

        # Generate overall summary
        summary = self._generate_overall_summary(sections)

        # Generate full report
        report_content = self._format_full_report(summary, sections)

        # Save report
        output_path = self._save_report(report_content)

        return output_path

    def _generate_code_quality_section(self, report: CodeQualityReport) -> QASectionReport:
        """Generate code quality section report."""
        compliance_status = 'compliant' if report.overall_score >= 80.0 else 'non_compliant'

        recommendations = report.recommendations
        key_findings = [
            f"Overall code quality score: {report.overall_score:.1f}%",
            f"Files analyzed: {report.files_analyzed}",
            f"Total lines of code: {report.total_lines}",
            f"PEP 8 compliance: {report.summary.get('average_pep8_compliance', 0):.1f}%",
            f"Type hint coverage: {report.summary.get('average_type_hint_coverage', 0):.1f}%",
            f"Documentation coverage: {report.summary.get('average_documentation_coverage', 0):.1f}%"
        ]

        return QASectionReport(
            section_name='Code Quality Standards',
            compliance_status=compliance_status,
            score=report.overall_score,
            issues_count=sum(len(m.issues) for m in report.metrics.values()),
            critical_issues=0,  # Code quality doesn't have severity levels
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'overall_score': report.overall_score,
                'files_analyzed': report.files_analyzed,
                'summary': report.summary,
                'target_compliance': report.summary.get('target_compliance', {})
            }
        )

    def _generate_testing_section(self, test_results: Dict[str, TestSuiteResult]) -> QASectionReport:
        """Generate testing section report."""
        total_tests = sum(r.total_tests for r in test_results.values())
        total_passed = sum(r.passed for r in test_results.values())
        total_failed = sum(r.failed for r in test_results.values())

        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        compliance_status = 'compliant' if success_rate >= 80.0 else 'non_compliant'

        coverage_scores = [r.coverage_percentage for r in test_results.values() if r.coverage_percentage > 0]
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0

        recommendations = []
        if success_rate < 80.0:
            recommendations.append("Improve test success rate by fixing failing test cases")
        if avg_coverage < 90.0:
            recommendations.append("Increase test coverage to meet 90% target")

        key_findings = [
            f"Overall test success rate: {success_rate:.1f}%",
            f"Total tests executed: {total_tests}",
            f"Tests passed: {total_passed}",
            f"Tests failed: {total_failed}",
            f"Average code coverage: {avg_coverage:.1f}%",
            f"Test suites executed: {len(test_results)}"
        ]

        return QASectionReport(
            section_name='Testing Framework',
            compliance_status=compliance_status,
            score=success_rate,
            issues_count=total_failed,
            critical_issues=total_failed,  # Failed tests are critical
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': success_rate,
                'average_coverage': avg_coverage,
                'suite_results': {name: {
                    'total': r.total_tests,
                    'passed': r.passed,
                    'failed': r.failed,
                    'coverage': r.coverage_percentage
                } for name, r in test_results.items()}
            }
        )

    def _generate_performance_section(self, report: PerformanceValidationReport) -> QASectionReport:
        """Generate performance section report."""
        compliance_status = 'compliant' if report.overall_compliance else 'non_compliant'

        recommendations = report.recommendations
        key_findings = [
            f"Overall performance compliance: {'PASS' if report.overall_compliance else 'FAIL'}",
            f"Performance tests executed: {len(report.test_results)}",
            f"Tests passed: {report.passed_tests}",
            f"Tests failed: {report.failed_tests}",
            f"Compliance score: {report.compliance_score:.1f}%"
        ]

        # Add specific performance metrics
        for result in report.test_results:
            if result.success and result.metrics:
                metric = result.metrics[0]
                key_findings.append(f"{metric.name}: {metric.value:.1f}{metric.unit}")

        return QASectionReport(
            section_name='Performance Standards',
            compliance_status=compliance_status,
            score=report.compliance_score,
            issues_count=report.failed_tests,
            critical_issues=report.failed_tests,  # Failed performance tests are critical
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'overall_compliance': report.overall_compliance,
                'tests_passed': report.passed_tests,
                'tests_failed': report.failed_tests,
                'compliance_score': report.compliance_score,
                'test_results': [{
                    'test_name': r.test_name,
                    'success': r.success,
                    'duration': r.duration,
                    'metrics': [{
                        'name': m.name,
                        'value': m.value,
                        'unit': m.unit,
                        'compliant': m.compliant
                    } for m in r.metrics]
                } for r in report.test_results]
            }
        )

    def _generate_security_section(self, report: SecurityComplianceReport) -> QASectionReport:
        """Generate security section report."""
        compliance_status = 'compliant' if report.overall_compliance else 'non_compliant'

        recommendations = report.recommendations
        key_findings = [
            f"Overall security compliance: {'PASS' if report.overall_compliance else 'FAIL'}",
            f"Security scans executed: {len(report.scan_results)}",
            f"Total security issues: {report.total_issues}",
            f"Critical issues: {report.critical_issues}",
            f"High severity issues: {report.high_issues}",
            f"Security compliance score: {report.compliance_score:.1f}%"
        ]

        return QASectionReport(
            section_name='Security Standards',
            compliance_status=compliance_status,
            score=report.compliance_score,
            issues_count=report.total_issues,
            critical_issues=report.critical_issues,
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'overall_compliance': report.overall_compliance,
                'total_issues': report.total_issues,
                'critical_issues': report.critical_issues,
                'high_issues': report.high_issues,
                'compliance_score': report.compliance_score,
                'scan_results': [{
                    'scan_type': r.scan_type,
                    'issues_found': len(r.issues_found),
                    'compliance_score': r.compliance_score
                } for r in report.scan_results]
            }
        )

    def _generate_accessibility_section(self, report: AccessibilityComplianceReport) -> QASectionReport:
        """Generate accessibility section report."""
        compliance_status = 'compliant' if report.overall_compliant else 'non_compliant'

        recommendations = report.recommendations
        key_findings = [
            f"WCAG 2.1 AA compliance: {'PASS' if report.overall_compliant else 'FAIL'}",
            f"Accessibility scans executed: {len(report.scan_results)}",
            f"Total accessibility issues: {report.total_issues}",
            f"Critical issues: {report.critical_issues}",
            f"Serious issues: {report.serious_issues}",
            f"Compliance score: {report.compliance_score:.1f}%"
        ]

        return QASectionReport(
            section_name='Accessibility Standards',
            compliance_status=compliance_status,
            score=report.compliance_score,
            issues_count=report.total_issues,
            critical_issues=report.critical_issues,
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'overall_compliant': report.overall_compliant,
                'total_issues': report.total_issues,
                'critical_issues': report.critical_issues,
                'serious_issues': report.serious_issues,
                'compliance_score': report.compliance_score,
                'scan_results': [{
                    'url_or_file': r.url_or_file,
                    'issues_found': len(r.issues_found),
                    'wcag_aa_compliant': r.wcag_aa_compliant
                } for r in report.scan_results]
            }
        )

    def _generate_risk_mitigation_section(self, report: RiskMitigationReport) -> QASectionReport:
        """Generate risk mitigation section report."""
        mitigation_effectiveness = report.mitigation_effectiveness
        compliance_status = 'compliant' if mitigation_effectiveness >= 80.0 else 'non_compliant'

        recommendations = report.recommendations
        key_findings = [
            f"Risk mitigation effectiveness: {mitigation_effectiveness:.1f}%",
            f"Total risk events: {report.total_events}",
            f"Events mitigated: {report.mitigated_events}",
            f"Failed mitigations: {report.failed_mitigations}",
            f"Active strategies: {report.active_strategies}",
            f"Recent events: {len(report.recent_events)}"
        ]

        return QASectionReport(
            section_name='Risk Mitigation',
            compliance_status=compliance_status,
            score=mitigation_effectiveness,
            issues_count=report.failed_mitigations,
            critical_issues=report.failed_mitigations,  # Failed mitigations are critical
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'mitigation_effectiveness': mitigation_effectiveness,
                'total_events': report.total_events,
                'mitigated_events': report.mitigated_events,
                'failed_mitigations': report.failed_mitigations,
                'active_strategies': report.active_strategies
            }
        )

    def _generate_context_management_section(self, report: ContextValidationReport) -> QASectionReport:
        """Generate context management section report."""
        compliance_status = 'compliant' if report.overall_compliance else 'non_compliant'

        recommendations = report.recommendations
        key_findings = [
            f"Context management compliance: {'PASS' if report.overall_compliance else 'FAIL'}",
            f"Validations executed: {report.total_validations}",
            f"Validations passed: {report.passed_validations}",
            f"Validations failed: {report.failed_validations}",
            f"Compliance score: {report.compliance_score:.1f}%"
        ]

        return QASectionReport(
            section_name='Context Management',
            compliance_status=compliance_status,
            score=report.compliance_score,
            issues_count=report.failed_validations,
            critical_issues=report.failed_validations,  # Failed validations are critical
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'overall_compliance': report.overall_compliance,
                'total_validations': report.total_validations,
                'passed_validations': report.passed_validations,
                'failed_validations': report.failed_validations,
                'compliance_score': report.compliance_score
            }
        )

    def _generate_compliance_engine_section(self, report: ComplianceReport) -> QASectionReport:
        """Generate compliance engine section report."""
        compliance_status = 'compliant' if report.overall_compliance else 'non_compliant'

        recommendations = report.recommendations
        key_findings = [
            f"Automated compliance: {'PASS' if report.overall_compliance else 'FAIL'}",
            f"Overall compliance score: {report.overall_score:.1f}%",
            f"Total checks executed: {report.total_checks}",
            f"Checks passed: {report.passed_checks}",
            f"Checks failed: {report.failed_checks}",
            f"Execution time: {report.execution_time:.2f}s"
        ]

        return QASectionReport(
            section_name='Automated Compliance Validation',
            compliance_status=compliance_status,
            score=report.overall_score,
            issues_count=report.failed_checks,
            critical_issues=report.failed_checks,  # Failed checks are critical
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'overall_compliance': report.overall_compliance,
                'overall_score': report.overall_score,
                'total_checks': report.total_checks,
                'passed_checks': report.passed_checks,
                'failed_checks': report.failed_checks,
                'execution_time': report.execution_time
            }
        )

    def _generate_monitoring_section(self, report: MonitoringReport) -> QASectionReport:
        """Generate monitoring section report."""
        health_score = report.system_health_score
        compliance_status = 'compliant' if health_score >= 80.0 else 'non_compliant'

        recommendations = report.recommendations
        key_findings = [
            f"System health score: {health_score:.1f}%",
            f"Total metrics monitored: {report.total_metrics}",
            f"Active alerts: {report.active_alerts}",
            f"Critical alerts: {report.critical_alerts}",
            f"Warning alerts: {report.warning_alerts}",
            f"Resolved alerts: {report.resolved_alerts}"
        ]

        return QASectionReport(
            section_name='Quality Monitoring',
            compliance_status=compliance_status,
            score=health_score,
            issues_count=report.active_alerts,
            critical_issues=report.critical_alerts,
            recommendations=recommendations,
            key_findings=key_findings,
            data={
                'system_health_score': health_score,
                'total_metrics': report.total_metrics,
                'active_alerts': report.active_alerts,
                'critical_alerts': report.critical_alerts,
                'warning_alerts': report.warning_alerts,
                'resolved_alerts': report.resolved_alerts
            }
        )

    def _generate_overall_summary(self, sections: List[QASectionReport]) -> QAComplianceSummary:
        """Generate overall QA compliance summary."""
        if not sections:
            return QAComplianceSummary(
                phase='Phase 1',
                overall_compliance=False,
                compliance_score=0.0,
                critical_issues=0,
                high_priority_issues=0,
                medium_priority_issues=0,
                low_priority_issues=0,
                total_issues=0,
                recommendations_count=0,
                assessment_date=datetime.now()
            )

        # Calculate overall metrics
        total_score = sum(s.score for s in sections)
        overall_score = total_score / len(sections)

        total_issues = sum(s.issues_count for s in sections)
        critical_issues = sum(s.critical_issues for s in sections)

        # Count compliant sections
        compliant_sections = sum(1 for s in sections if s.compliance_status == 'compliant')
        overall_compliance = compliant_sections == len(sections)

        # Count recommendations
        recommendations_count = sum(len(s.recommendations) for s in sections)

        # Categorize issues (simplified)
        high_priority_issues = critical_issues
        medium_priority_issues = total_issues // 3
        low_priority_issues = total_issues - high_priority_issues - medium_priority_issues

        return QAComplianceSummary(
            phase='Phase 1',
            overall_compliance=overall_compliance,
            compliance_score=overall_score,
            critical_issues=critical_issues,
            high_priority_issues=high_priority_issues,
            medium_priority_issues=medium_priority_issues,
            low_priority_issues=low_priority_issues,
            total_issues=total_issues,
            recommendations_count=recommendations_count,
            assessment_date=datetime.now()
        )

    def _format_full_report(self, summary: QAComplianceSummary, sections: List[QASectionReport]) -> str:
        """Format the complete QA report."""
        lines = []

        # Header
        lines.append("=" * 100)
        lines.append("CES PHASE 1 QUALITY ASSURANCE COMPLIANCE REPORT")
        lines.append("=" * 100)
        lines.append(f"Assessment Date: {summary.assessment_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Phase: {summary.phase}")
        lines.append("")

        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 50)
        lines.append(f"Overall Compliance: {'PASS' if summary.overall_compliance else 'FAIL'}")
        lines.append(f"Compliance Score: {summary.compliance_score:.1f}%")
        lines.append(f"Total Issues: {summary.total_issues}")
        lines.append(f"Critical Issues: {summary.critical_issues}")
        lines.append(f"High Priority Issues: {summary.high_priority_issues}")
        lines.append(f"Medium Priority Issues: {summary.medium_priority_issues}")
        lines.append(f"Low Priority Issues: {summary.low_priority_issues}")
        lines.append(f"Total Recommendations: {summary.recommendations_count}")
        lines.append("")

        # Section Reports
        lines.append("DETAILED SECTION REPORTS")
        lines.append("-" * 50)
        lines.append("")

        for section in sections:
            lines.append(f"## {section.section_name}")
            lines.append(f"Compliance Status: {section.compliance_status.upper()}")
            lines.append(f"Score: {section.score:.1f}%")
            lines.append(f"Issues Count: {section.issues_count}")
            lines.append(f"Critical Issues: {section.critical_issues}")
            lines.append("")

            if section.key_findings:
                lines.append("Key Findings:")
                for finding in section.key_findings:
                    lines.append(f"• {finding}")
                lines.append("")

            if section.recommendations:
                lines.append("Recommendations:")
                for rec in section.recommendations:
                    lines.append(f"• {rec}")
                lines.append("")

        # Footer
        lines.append("=" * 100)
        lines.append("END OF CES PHASE 1 QA COMPLIANCE REPORT")
        lines.append("=" * 100)

        return "\n".join(lines)

    def _save_report(self, report_content: str) -> str:
        """Save the report to file."""
        output_dir = self.project_root / "benchmark_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(datetime.now().timestamp())
        output_path = output_dir / f"ces_phase1_qa_report_{timestamp}.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # Also save as JSON for programmatic access
        json_path = output_dir / f"ces_phase1_qa_report_{timestamp}.json"
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'CES Phase 1 QA Compliance Report',
            'content': report_content
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)

        return str(output_path)

    def generate_quick_status_report(self, sections_data: Dict[str, Any]) -> str:
        """Generate a quick status report for dashboard display."""
        lines = []

        lines.append("CES QA Status Report")
        lines.append("=" * 30)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        for section_name, data in sections_data.items():
            status = data.get('status', 'unknown')
            score = data.get('score', 0)
            issues = data.get('issues', 0)

            status_icon = "✓" if status == 'compliant' else "✗" if status == 'non_compliant' else "?"

            lines.append(f"{status_icon} {section_name}")
            lines.append(f"   Status: {status}")
            lines.append(f"   Score: {score:.1f}%")
            lines.append(f"   Issues: {issues}")
            lines.append("")

        return "\n".join(lines)