"""CES QA Report Generator.

Generates comprehensive QA compliance reports for CES Phase 1,
including executive summaries, detailed analysis, and actionable recommendations.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging


@dataclass
class QAReportSection:
    """Represents a section in the QA report."""
    title: str
    content: str
    status: str  # 'pass', 'fail', 'warning'
    score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class QAComplianceReport:
    """Complete QA compliance report."""
    title: str
    executive_summary: str
    sections: List[QAReportSection]
    overall_score: float
    compliance_status: str
    recommendations: List[str]
    generated_at: datetime
    report_version: str = "1.0"


class QAReportGenerator:
    """Generates comprehensive QA compliance reports."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_phase1_qa_report(
        self,
        code_quality_report=None,
        test_results=None,
        performance_report=None,
        security_report=None,
        accessibility_report=None,
        risk_report=None,
        context_report=None,
        compliance_report=None,
        monitoring_report=None
    ) -> str:
        """Generate comprehensive Phase 1 QA report."""

        # Create report sections
        sections = []

        # Executive Summary
        executive_summary = self._generate_executive_summary(
            code_quality_report, test_results, performance_report,
            security_report, accessibility_report, risk_report,
            context_report, compliance_report, monitoring_report
        )

        # Code Quality Section
        if code_quality_report:
            sections.append(self._generate_code_quality_section(code_quality_report))

        # Testing Section
        if test_results:
            sections.append(self._generate_testing_section(test_results))

        # Performance Section
        if performance_report:
            sections.append(self._generate_performance_section(performance_report))

        # Security Section
        if security_report:
            sections.append(self._generate_security_section(security_report))

        # Accessibility Section
        if accessibility_report:
            sections.append(self._generate_accessibility_section(accessibility_report))

        # Risk Mitigation Section
        if risk_report:
            sections.append(self._generate_risk_mitigation_section(risk_report))

        # Context Management Section
        if context_report:
            sections.append(self._generate_context_management_section(context_report))

        # Compliance Validation Section
        if compliance_report:
            sections.append(self._generate_compliance_validation_section(compliance_report))

        # Quality Monitoring Section
        if monitoring_report:
            sections.append(self._generate_quality_monitoring_section(monitoring_report))

        # Calculate overall metrics
        overall_score = self._calculate_overall_score([
            code_quality_report, test_results, performance_report,
            security_report, accessibility_report, risk_report,
            context_report, compliance_report, monitoring_report
        ])

        compliance_status = self._determine_compliance_status(overall_score)

        # Generate recommendations
        recommendations = self._generate_overall_recommendations(
            code_quality_report, test_results, performance_report,
            security_report, accessibility_report, risk_report,
            context_report, compliance_report, monitoring_report
        )

        # Create complete report
        report = QAComplianceReport(
            title="CES Phase 1 Quality Assurance Compliance Report",
            executive_summary=executive_summary,
            sections=sections,
            overall_score=overall_score,
            compliance_status=compliance_status,
            recommendations=recommendations,
            generated_at=datetime.now()
        )

        # Generate and save reports
        return self._save_report(report)

    def _generate_executive_summary(self, *reports) -> str:
        """Generate executive summary from all reports."""
        lines = []

        lines.append("# Executive Summary")
        lines.append("")
        lines.append("This report provides a comprehensive assessment of CES Phase 1 quality assurance compliance.")
        lines.append("")

        # Overall status
        valid_reports = [r for r in reports if r is not None]
        if valid_reports:
            overall_score = self._calculate_overall_score(valid_reports)
            compliance_status = self._determine_compliance_status(overall_score)

            lines.append("## Overall Assessment")
            lines.append("")
            lines.append(f"- **Compliance Status**: {compliance_status.upper()}")
            lines.append(f"- **Overall Score**: {overall_score:.1f}%")
            lines.append(f"- **Reports Analyzed**: {len(valid_reports)}")
            lines.append("")

        # Key findings
        lines.append("## Key Findings")
        lines.append("")

        findings = []

        # Check each report type
        for i, report in enumerate(reports):
            if report is None:
                continue

            report_names = [
                "Code Quality", "Testing", "Performance", "Security",
                "Accessibility", "Risk Mitigation", "Context Management",
                "Compliance", "Monitoring"
            ]

            if i < len(report_names):
                report_name = report_names[i]

                # Extract score from report
                score = self._extract_report_score(report)
                if score is not None:
                    status = "✅ PASS" if score >= 80 else ("⚠️  WARNING" if score >= 60 else "❌ FAIL")
                    findings.append(f"- **{report_name}**: {status} ({score:.1f}%)")

        lines.extend(findings)
        lines.append("")

        # Next steps
        lines.append("## Next Steps")
        lines.append("")
        lines.append("1. Review detailed sections for specific issues and recommendations")
        lines.append("2. Address critical and high-priority findings")
        lines.append("3. Implement suggested improvements")
        lines.append("4. Re-run QA assessment to validate fixes")
        lines.append("5. Set up continuous monitoring for ongoing quality assurance")
        lines.append("")

        return "\n".join(lines)

    def _generate_code_quality_section(self, report) -> QAReportSection:
        """Generate code quality section."""
        content = []

        content.append("## Code Quality Analysis")
        content.append("")

        if hasattr(report, 'overall_score'):
            score = report.overall_score
            status = "pass" if score >= 80 else "fail"
            content.append(".1f"            content.append("")

        if hasattr(report, 'summary'):
            summary = report.summary
            content.append("### Key Metrics")
            content.append("")
            content.append(f"- **PEP 8 Compliance**: {summary.get('average_pep8_compliance', 0):.1f}%")
            content.append(f"- **Type Hint Coverage**: {summary.get('average_type_hint_coverage', 0):.1f}%")
            content.append(f"- **Documentation Coverage**: {summary.get('average_documentation_coverage', 0):.1f}%")
            content.append(f"- **Average Complexity**: {summary.get('average_complexity', 0):.1f}")
            content.append(f"- **Files Analyzed**: {summary.get('files_analyzed', 0)}")
            content.append("")

        if hasattr(report, 'recommendations') and report.recommendations:
            content.append("### Recommendations")
            content.append("")
            for rec in report.recommendations:
                content.append(f"- {rec}")
            content.append("")

        return QAReportSection(
            title="Code Quality",
            content="\n".join(content),
            status="pass" if getattr(report, 'overall_score', 0) >= 80 else "fail",
            score=getattr(report, 'overall_score', None)
        )

    def _generate_testing_section(self, test_results) -> QAReportSection:
        """Generate testing section."""
        content = []

        content.append("## Testing Framework Analysis")
        content.append("")

        if hasattr(test_results, 'summary'):
            summary = test_results.summary
            content.append("### Test Results Summary")
            content.append("")
            content.append(f"- **Total Tests**: {summary.get('total_tests', 0)}")
            content.append(f"- **Passed**: {summary.get('passed', 0)}")
            content.append(f"- **Failed**: {summary.get('failed', 0)}")
            content.append(f"- **Success Rate**: {summary.get('success_rate', 0):.1f}%")
            content.append(f"- **Coverage**: {summary.get('average_coverage', 0):.1f}%")
            content.append("")

        # Suite results
        if hasattr(test_results, 'suite_results'):
            content.append("### Test Suite Results")
            content.append("")
            for suite_name, result in test_results.suite_results.items():
                success_rate = result.get('success_rate', 0)
                status = "✅" if success_rate >= 80 else ("⚠️" if success_rate >= 60 else "❌")
                content.append(f"- **{suite_name.title()}**: {status} {success_rate:.1f}% ({result.get('passed', 0)}/{result.get('total_tests', 0)})")
            content.append("")

        return QAReportSection(
            title="Testing",
            content="\n".join(content),
            status="pass" if test_results.summary.get('success_rate', 0) >= 80 else "fail",
            score=test_results.summary.get('success_rate', 0)
        )

    def _generate_performance_section(self, report) -> QAReportSection:
        """Generate performance section."""
        content = []

        content.append("## Performance Validation")
        content.append("")

        if hasattr(report, 'compliance_score'):
            score = report.compliance_score
            status = "pass" if score >= 80 else "fail"
            content.append(".1f"            content.append("")

        if hasattr(report, 'summary'):
            summary = report.summary
            content.append("### Performance Metrics")
            content.append("")
            content.append(f"- **Tests Passed**: {summary.get('passed_tests', 0)}")
            content.append(f"- **Tests Failed**: {summary.get('failed_tests', 0)}")
            content.append(f"- **Total Metrics**: {summary.get('total_metrics', 0)}")
            content.append("")

        if hasattr(report, 'recommendations') and report.recommendations:
            content.append("### Recommendations")
            content.append("")
            for rec in report.recommendations:
                content.append(f"- {rec}")
            content.append("")

        return QAReportSection(
            title="Performance",
            content="\n".join(content),
            status="pass" if getattr(report, 'compliance_score', 0) >= 80 else "fail",
            score=getattr(report, 'compliance_score', None)
        )

    def _generate_security_section(self, report) -> QAReportSection:
        """Generate security section."""
        content = []

        content.append("## Security Audit")
        content.append("")

        if hasattr(report, 'compliance_score'):
            score = report.compliance_score
            status = "pass" if score >= 80 else "fail"
            content.append(".1f"            content.append("")

        content.append("### Security Metrics")
        content.append("")
        content.append(f"- **Total Issues**: {getattr(report, 'total_issues', 0)}")
        content.append(f"- **Critical Issues**: {getattr(report, 'critical_issues', 0)}")
        content.append(f"- **High Issues**: {getattr(report, 'high_issues', 0)}")
        content.append(f"- **Medium Issues**: {getattr(report, 'medium_issues', 0)}")
        content.append(f"- **Low Issues**: {getattr(report, 'low_issues', 0)}")
        content.append("")

        if hasattr(report, 'recommendations') and report.recommendations:
            content.append("### Recommendations")
            content.append("")
            for rec in report.recommendations:
                content.append(f"- {rec}")
            content.append("")

        return QAReportSection(
            title="Security",
            content="\n".join(content),
            status="pass" if getattr(report, 'compliance_score', 0) >= 80 else "fail",
            score=getattr(report, 'compliance_score', None)
        )

    def _generate_accessibility_section(self, report) -> QAReportSection:
        """Generate accessibility section."""
        content = []

        content.append("## Accessibility Audit")
        content.append("")

        if hasattr(report, 'compliance_score'):
            score = report.compliance_score
            status = "pass" if score >= 80 else "fail"
            content.append(f"- **Score**: {score:.1f}%")
            content.append("")

        content.append("### Accessibility Metrics")
        content.append("")
        content.append(f"- **Total Issues**: {getattr(report, 'total_issues', 0)}")
        content.append(f"- **Critical Issues**: {getattr(report, 'critical_issues', 0)}")
        content.append(f"- **WCAG AA Compliant**: {'Yes' if getattr(report, 'overall_compliant', False) else 'No'}")
        content.append("")

        if hasattr(report, 'recommendations') and report.recommendations:
            content.append("### Recommendations")
            content.append("")
            for rec in report.recommendations:
                content.append(f"- {rec}")
            content.append("")

        return QAReportSection(
            title="Accessibility",
            content="\n".join(content),
            status="pass" if getattr(report, 'compliance_score', 0) >= 80 else "fail",
            score=getattr(report, 'compliance_score', None)
        )

    def _generate_risk_mitigation_section(self, report) -> QAReportSection:
        """Generate risk mitigation section."""
        content = []

        content.append("## Risk Mitigation Validation")
        content.append("")

        if hasattr(report, 'mitigation_effectiveness'):
            score = report.mitigation_effectiveness
            status = "pass" if score >= 80 else "fail"
            content.append(".1f"            content.append("")

        content.append("### Risk Metrics")
        content.append("")
        content.append(f"- **Total Events**: {getattr(report, 'total_events', 0)}")
        content.append(f"- **Mitigated Events**: {getattr(report, 'mitigated_events', 0)}")
        content.append(f"- **Failed Mitigations**: {getattr(report, 'failed_mitigations', 0)}")
        content.append(f"- **Active Strategies**: {getattr(report, 'active_strategies', 0)}")
        content.append("")

        if hasattr(report, 'recommendations') and report.recommendations:
            content.append("### Recommendations")
            content.append("")
            for rec in report.recommendations:
                content.append(f"- {rec}")
            content.append("")

        return QAReportSection(
            title="Risk Mitigation",
            content="\n".join(content),
            status="pass" if getattr(report, 'mitigation_effectiveness', 0) >= 80 else "fail",
            score=getattr(report, 'mitigation_effectiveness', None)
        )

    def _generate_context_management_section(self, report) -> QAReportSection:
        """Generate context management section."""
        content = []

        content.append("## Context Management Validation")
        content.append("")

        if hasattr(report, 'compliance_score'):
            score = report.compliance_score
            status = "pass" if score >= 80 else "fail"
            content.append(".1f"            content.append("")

        content.append("### Context Metrics")
        content.append("")
        content.append(f"- **Total Validations**: {getattr(report, 'total_validations', 0)}")
        content.append(f"- **Passed Validations**: {getattr(report, 'passed_validations', 0)}")
        content.append(f"- **Failed Validations**: {getattr(report, 'failed_validations', 0)}")
        content.append("")

        if hasattr(report, 'recommendations') and report.recommendations:
            content.append("### Recommendations")
            content.append("")
            for rec in report.recommendations:
                content.append(f"- {rec}")
            content.append("")

        return QAReportSection(
            title="Context Management",
            content="\n".join(content),
            status="pass" if getattr(report, 'compliance_score', 0) >= 80 else "fail",
            score=getattr(report, 'compliance_score', None)
        )

    def _generate_compliance_validation_section(self, report) -> QAReportSection:
        """Generate compliance validation section."""
        content = []

        content.append("## Automated Compliance Validation")
        content.append("")

        if hasattr(report, 'overall_score'):
            score = report.overall_score
            status = "pass" if score >= 80 else "fail"
            content.append(".1f"            content.append("")

        content.append("### Compliance Metrics")
        content.append("")
        content.append(f"- **Total Checks**: {getattr(report, 'total_checks', 0)}")
        content.append(f"- **Passed Checks**: {getattr(report, 'passed_checks', 0)}")
        content.append(f"- **Failed Checks**: {getattr(report, 'failed_checks', 0)}")
        content.append(f"- **Error Checks**: {getattr(report, 'error_checks', 0)}")
        content.append(f"- **Execution Time**: {getattr(report, 'execution_time', 0):.2f}s")
        content.append("")

        if hasattr(report, 'recommendations') and report.recommendations:
            content.append("### Recommendations")
            content.append("")
            for rec in report.recommendations:
                content.append(f"- {rec}")
            content.append("")

        return QAReportSection(
            title="Compliance Validation",
            content="\n".join(content),
            status="pass" if getattr(report, 'overall_score', 0) >= 80 else "fail",
            score=getattr(report, 'overall_score', None)
        )

    def _generate_quality_monitoring_section(self, report) -> QAReportSection:
        """Generate quality monitoring section."""
        content = []

        content.append("## Quality Monitoring")
        content.append("")

        if hasattr(report, 'system_health_score'):
            score = report.system_health_score
            status = "pass" if score >= 80 else "fail"
            content.append(".1f"            content.append("")

        content.append("### Monitoring Metrics")
        content.append("")
        content.append(f"- **Total Alerts**: {getattr(report, 'total_alerts', 0)}")
        content.append(f"- **Active Alerts**: {getattr(report, 'active_alerts', 0)}")
        content.append(f"- **Critical Alerts**: {getattr(report, 'critical_alerts', 0)}")
        content.append("")

        if hasattr(report, 'recent_alerts') and report.recent_alerts:
            content.append("### Recent Alerts")
            content.append("")
            for alert in report.recent_alerts[-5:]:  # Show last 5 alerts
                severity_icon = {
                    'critical': '🚨',
                    'high': '⚠️',
                    'medium': 'ℹ️',
                    'low': 'ℹ️'
                }.get(alert.severity, 'ℹ️')
                content.append(f"- {severity_icon} **{alert.title}**: {alert.description}")
            content.append("")

        return QAReportSection(
            title="Quality Monitoring",
            content="\n".join(content),
            status="pass" if getattr(report, 'system_health_score', 0) >= 80 else "fail",
            score=getattr(report, 'system_health_score', None)
        )

    def _calculate_overall_score(self, reports) -> float:
        """Calculate overall QA score from all reports."""
        valid_reports = [r for r in reports if r is not None]
        if not valid_reports:
            return 0.0

        total_score = 0
        count = 0

        for report in valid_reports:
            score = self._extract_report_score(report)
            if score is not None:
                total_score += score
                count += 1

        return total_score / count if count > 0 else 0.0

    def _extract_report_score(self, report) -> Optional[float]:
        """Extract score from a report object."""
        if hasattr(report, 'overall_score'):
            return report.overall_score
        elif hasattr(report, 'compliance_score'):
            return report.compliance_score
        elif hasattr(report, 'mitigation_effectiveness'):
            return report.mitigation_effectiveness
        elif hasattr(report, 'system_health_score'):
            return report.system_health_score
        elif hasattr(report, 'summary') and 'success_rate' in report.summary:
            return report.summary['success_rate']
        else:
            return None

    def _determine_compliance_status(self, score: float) -> str:
        """Determine compliance status based on score."""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "acceptable"
        elif score >= 60:
            return "needs_improvement"
        else:
            return "critical"

    def _generate_overall_recommendations(self, *reports) -> List[str]:
        """Generate overall recommendations from all reports."""
        recommendations = []

        # Check each report for recommendations
        for i, report in enumerate(reports):
            if report is None:
                continue

            if hasattr(report, 'recommendations') and report.recommendations:
                recommendations.extend(report.recommendations)

        # Remove duplicates and limit to top 10
        unique_recommendations = list(set(recommendations))[:10]

        # Add general recommendations
        if not unique_recommendations:
            unique_recommendations.append("All QA checks passed! System is ready for Phase 1 production.")
        else:
            unique_recommendations.append("Schedule regular QA assessments to maintain quality standards.")
            unique_recommendations.append("Implement automated monitoring and alerting for continuous quality assurance.")

        return unique_recommendations

    def _save_report(self, report: QAComplianceReport) -> str:
        """Save the report to files and return the main report path."""
        # Create reports directory
        reports_dir = Path("benchmark_results")
        reports_dir.mkdir(exist_ok=True)

        timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")

        # Save detailed JSON report
        json_report = {
            'title': report.title,
            'executive_summary': report.executive_summary,
            'overall_score': report.overall_score,
            'compliance_status': report.compliance_status,
            'sections': [
                {
                    'title': section.title,
                    'status': section.status,
                    'score': section.score,
                    'content': section.content,
                    'details': section.details
                }
                for section in report.sections
            ],
            'recommendations': report.recommendations,
            'generated_at': report.generated_at.isoformat(),
            'report_version': report.report_version
        }

        json_path = reports_dir / f"ces_phase1_qa_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)

        # Save markdown report
        markdown_content = self._generate_markdown_report(report)
        md_path = reports_dir / f"ces_phase1_qa_report_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(markdown_content)

        # Save summary text report
        summary_content = self._generate_summary_report(report)
        summary_path = reports_dir / f"ces_phase1_qa_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_content)

        self.logger.info(f"QA reports saved to {reports_dir}")

        return str(md_path)

    def _generate_markdown_report(self, report: QAComplianceReport) -> str:
        """Generate markdown version of the report."""
        lines = []

        # Header
        lines.append("# CES Phase 1 Quality Assurance Compliance Report")
        lines.append("")
        lines.append(f"**Generated**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Overall Score**: {report.overall_score:.1f}%")
        lines.append(f"**Compliance Status**: {report.compliance_status.upper()}")
        lines.append("")

        # Executive Summary
        lines.append(report.executive_summary)
        lines.append("")

        # Detailed Sections
        lines.append("# Detailed Analysis")
        lines.append("")

        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            if section.score is not None:
                status_icon = "✅" if section.status == "pass" else ("⚠️" if section.status == "warning" else "❌")
                lines.append(f"**Status**: {status_icon} {section.status.upper()}")
                lines.append(".1f"                lines.append("")
            lines.append(section.content)
            lines.append("---")
            lines.append("")

        # Recommendations
        lines.append("# Recommendations")
        lines.append("")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*This report was generated by the CES QA Framework*")
        lines.append(f"*Report Version: {report.report_version}*")

        return "\n".join(lines)

    def _generate_summary_report(self, report: QAComplianceReport) -> str:
        """Generate text summary of the report."""
        lines = []

        lines.append("CES PHASE 1 QA COMPLIANCE SUMMARY")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append(f"OVERALL SCORE: {report.overall_score:.1f}%")
        lines.append(f"COMPLIANCE STATUS: {report.compliance_status.upper()}")
        lines.append("")

        # Section summaries
        lines.append("SECTION RESULTS:")
        lines.append("-" * 30)
        for section in report.sections:
            status_icon = "PASS" if section.status == "pass" else ("WARN" if section.status == "warning" else "FAIL")
            score_str = f" ({section.score:.1f}%)" if section.score is not None else ""
            lines.append(f"{section.title:20} : {status_icon}{score_str}")
        lines.append("")

        # Top recommendations
        lines.append("TOP RECOMMENDATIONS:")
        lines.append("-" * 30)
        for i, rec in enumerate(report.recommendations[:5], 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        lines.append("=" * 50)

        return "\n".join(lines)