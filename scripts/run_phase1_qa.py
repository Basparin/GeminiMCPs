#!/usr/bin/env python3
"""
CES Phase 1 Quality Assurance Runner

This script executes the complete CES Phase 1 QA framework, including:
- Code quality analysis
- Testing framework validation
- Performance validation
- Security auditing
- Accessibility compliance
- Risk mitigation validation
- Context management validation
- Automated compliance checking
- Quality monitoring
- Comprehensive report generation

Usage:
    python3 scripts/run_phase1_qa.py [--verbose] [--save-reports]

Options:
    --verbose: Enable verbose output
    --save-reports: Save detailed reports to files
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Add CES to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ces.qa.code_quality import CodeQualityAnalyzer
from ces.qa.test_framework import EnhancedTestFramework
from ces.qa.performance_validator import PerformanceValidator
from ces.qa.security_analyzer import SecurityAnalyzer
from ces.qa.accessibility_checker import AccessibilityChecker
from ces.qa.risk_mitigator import RiskMitigator
from ces.qa.context_validator import ContextValidator
from ces.qa.compliance_engine import ComplianceEngine
from ces.qa.quality_monitor import QualityMonitor
from ces.qa.qa_report_generator import QAReportGenerator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ces_qa.log', mode='a')
        ]
    )


def run_code_quality_analysis(verbose: bool) -> dict:
    """Run code quality analysis."""
    print("🔍 Running Code Quality Analysis..." if verbose else "Running Code Quality Analysis...")
    start_time = time.time()

    try:
        analyzer = CodeQualityAnalyzer()
        report = analyzer.analyze_codebase()

        duration = time.time() - start_time
        if verbose:
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   📊 Files analyzed: {report.files_analyzed}")
            print(f"   📈 PEP 8 compliance: {report.summary.get('average_pep8_compliance', 0):.1f}%")
            print(f"   📝 Type hints: {report.summary.get('average_type_hint_coverage', 0):.1f}%")
            print(f"   📚 Documentation: {report.summary.get('average_documentation_coverage', 0):.1f}%")

        return {
            'success': True,
            'report': report,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Code quality analysis failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def run_testing_framework(verbose: bool) -> dict:
    """Run testing framework validation."""
    print("🧪 Running Testing Framework Validation..." if verbose else "Running Testing Framework...")
    start_time = time.time()

    try:
        framework = EnhancedTestFramework()
        results = framework.run_comprehensive_test_suite()

        duration = time.time() - start_time
        if verbose:
            print(f"   ⏱️  Duration: {duration:.2f}s")
            summary = results.summary
            print(f"   📊 Total tests: {summary.get('total_tests', 0)}")
            print(f"   ✅ Passed: {summary.get('passed', 0)}")
            print(f"   ❌ Failed: {summary.get('failed', 0)}")
            print(f"   📈 Success rate: {summary.get('success_rate', 0):.1f}%")
            print(f"   📊 Coverage: {summary.get('average_coverage', 0):.1f}%")
        return {
            'success': True,
            'results': results,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Testing framework validation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def run_performance_validation(verbose: bool) -> dict:
    """Run performance validation."""
    print("⚡ Running Performance Validation..." if verbose else "Running Performance Validation...")
    start_time = time.time()

    try:
        validator = PerformanceValidator()
        report = validator.validate_all_standards()

        duration = time.time() - start_time
        if verbose:
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   📊 Tests passed: {report.summary.get('passed_tests', 0)}")
            print(f"   📊 Tests failed: {report.summary.get('failed_tests', 0)}")
            print(f"   📈 Total metrics: {report.summary.get('total_metrics', 0)}")

        return {
            'success': True,
            'report': report,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Performance validation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def run_security_audit(verbose: bool) -> dict:
    """Run security audit."""
    print("🔒 Running Security Audit..." if verbose else "Running Security Audit...")
    start_time = time.time()

    try:
        analyzer = SecurityAnalyzer()
        report = analyzer.perform_comprehensive_security_audit()

        duration = time.time() - start_time
        if verbose:
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   🚨 Critical issues: {report.critical_issues}")
            print(f"   ⚠️  High issues: {report.high_issues}")
            print(f"   📊 Total issues: {report.total_issues}")

        return {
            'success': True,
            'report': report,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Security audit failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def run_accessibility_audit(verbose: bool) -> dict:
    """Run accessibility audit."""
    print("♿ Running Accessibility Audit..." if verbose else "Running Accessibility Audit...")
    start_time = time.time()

    try:
        checker = AccessibilityChecker()
        report = checker.perform_accessibility_audit()

        duration = time.time() - start_time
        if verbose:
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   🚨 Critical issues: {report.critical_issues}")
            print(f"   ⚠️  Serious issues: {report.serious_issues}")
            print(f"   📊 Total issues: {report.total_issues}")
            print(f"   ✅ WCAG AA compliant: {'Yes' if report.overall_compliant else 'No'}")

        return {
            'success': True,
            'report': report,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Accessibility audit failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def run_risk_mitigation_validation(verbose: bool) -> dict:
    """Run risk mitigation validation."""
    print("🛡️  Running Risk Mitigation Validation..." if verbose else "Running Risk Mitigation...")
    start_time = time.time()

    try:
        mitigator = RiskMitigator()
        report = mitigator.get_mitigation_report()

        duration = time.time() - start_time
        if verbose:
            print(".1f"            print(f"   📊 Total events: {report.total_events}")
            print(f"   ✅ Mitigated events: {report.mitigated_events}")
            print(f"   ❌ Failed mitigations: {report.failed_mitigations}")
            print(f"   🔧 Active strategies: {report.active_strategies}")

        return {
            'success': True,
            'report': report,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Risk mitigation validation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def run_context_validation(verbose: bool) -> dict:
    """Run context management validation."""
    print("🧠 Running Context Management Validation..." if verbose else "Running Context Validation...")
    start_time = time.time()

    try:
        validator = ContextValidator()
        report = validator.validate_all_context_management()

        duration = time.time() - start_time
        if verbose:
            print(".1f"            print(f"   📊 Total validations: {report.total_validations}")
            print(f"   ✅ Passed validations: {report.passed_validations}")
            print(f"   ❌ Failed validations: {report.failed_validations}")

        return {
            'success': True,
            'report': report,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Context validation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def run_compliance_validation(verbose: bool) -> dict:
    """Run automated compliance validation."""
    print("✅ Running Automated Compliance Validation..." if verbose else "Running Compliance Validation...")
    start_time = time.time()

    try:
        engine = ComplianceEngine()
        report = await engine.run_full_compliance_check()

        duration = time.time() - start_time
        if verbose:
            print(".1f"            print(f"   📊 Total checks: {report.total_checks}")
            print(f"   ✅ Passed checks: {report.passed_checks}")
            print(f"   ❌ Failed checks: {report.failed_checks}")
            print(f"   ⚠️  Error checks: {report.error_checks}")
            print(".1f"        return {
            'success': True,
            'report': report,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Compliance validation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def run_quality_monitoring(verbose: bool) -> dict:
    """Run quality monitoring."""
    print("📊 Running Quality Monitoring..." if verbose else "Running Quality Monitoring...")
    start_time = time.time()

    try:
        monitor = QualityMonitor()
        report = monitor.get_monitoring_report()

        duration = time.time() - start_time
        if verbose:
            print(".1f"            print(f"   🚨 Active alerts: {report.active_alerts}")
            print(f"   🚨 Critical alerts: {report.critical_alerts}")
            print(f"   📈 Total alerts: {report.total_alerts}")

        return {
            'success': True,
            'report': report,
            'duration': duration
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Quality monitoring failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


async def run_full_qa_suite(verbose: bool = False, save_reports: bool = True) -> dict:
    """Run the complete QA suite."""
    print("🚀 Starting CES Phase 1 Quality Assurance Suite")
    print("=" * 60)

    start_time = time.time()
    results = {}

    # Run all QA components
    print("\n📋 Running QA Components...\n")

    # 1. Code Quality Analysis
    results['code_quality'] = run_code_quality_analysis(verbose)

    # 2. Testing Framework
    results['testing'] = run_testing_framework(verbose)

    # 3. Performance Validation
    results['performance'] = run_performance_validation(verbose)

    # 4. Security Audit
    results['security'] = run_security_audit(verbose)

    # 5. Accessibility Audit
    results['accessibility'] = run_accessibility_audit(verbose)

    # 6. Risk Mitigation
    results['risk_mitigation'] = run_risk_mitigation_validation(verbose)

    # 7. Context Validation
    results['context'] = run_context_validation(verbose)

    # 8. Compliance Validation
    results['compliance'] = await run_compliance_validation(verbose)

    # 9. Quality Monitoring
    results['monitoring'] = run_quality_monitoring(verbose)

    # Calculate overall results
    total_duration = time.time() - start_time
    successful_components = sum(1 for r in results.values() if r.get('success', False))
    total_components = len(results)

    print("\n" + "=" * 60)
    print("🎯 QA SUITE COMPLETED")
    print("=" * 60)
    print(".1f"    print(f"📊 Components run: {total_components}")
    print(f"✅ Successful: {successful_components}")
    print(f"❌ Failed: {total_components - successful_components}")

    # Generate comprehensive report
    if save_reports:
        print("\n📄 Generating Comprehensive QA Report...")
        try:
            generator = QAReportGenerator()

            # Extract reports from results
            code_quality_report = results['code_quality'].get('report') if results['code_quality'].get('success') else None
            test_results = results['testing'].get('results') if results['testing'].get('success') else None
            performance_report = results['performance'].get('report') if results['performance'].get('success') else None
            security_report = results['security'].get('report') if results['security'].get('success') else None
            accessibility_report = results['accessibility'].get('report') if results['accessibility'].get('success') else None
            risk_report = results['risk_mitigation'].get('report') if results['risk_mitigation'].get('success') else None
            context_report = results['context'].get('report') if results['context'].get('success') else None
            compliance_report = results['compliance'].get('report') if results['compliance'].get('success') else None
            monitoring_report = results['monitoring'].get('report') if results['monitoring'].get('success') else None

            report_path = generator.generate_phase1_qa_report(
                code_quality_report=code_quality_report,
                test_results=test_results,
                performance_report=performance_report,
                security_report=security_report,
                accessibility_report=accessibility_report,
                risk_report=risk_report,
                context_report=context_report,
                compliance_report=compliance_report,
                monitoring_report=monitoring_report
            )

            print(f"📋 Report saved to: {report_path}")

        except Exception as e:
            print(f"❌ Report generation failed: {str(e)}")

    # Print summary
    print("\n📈 QA COMPONENT SUMMARY:")
    print("-" * 40)

    component_names = {
        'code_quality': 'Code Quality',
        'testing': 'Testing Framework',
        'performance': 'Performance',
        'security': 'Security',
        'accessibility': 'Accessibility',
        'risk_mitigation': 'Risk Mitigation',
        'context': 'Context Management',
        'compliance': 'Compliance',
        'monitoring': 'Quality Monitoring'
    }

    for key, result in results.items():
        component_name = component_names.get(key, key)
        status = "✅ PASS" if result.get('success') else "❌ FAIL"
        duration = ".2f"        print("<25")

    print("\n🎉 CES Phase 1 QA Suite execution completed!")
    print("=" * 60)

    return {
        'overall_success': successful_components == total_components,
        'successful_components': successful_components,
        'total_components': total_components,
        'total_duration': total_duration,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CES Phase 1 Quality Assurance Runner')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--save-reports', '-s', action='store_true', default=True, help='Save detailed reports to files')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    try:
        # Run QA suite
        import asyncio
        result = asyncio.run(run_full_qa_suite(args.verbose, args.save_reports))

        # Exit with appropriate code
        if result['overall_success']:
            print("\n🎯 All QA components completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {result['total_components'] - result['successful_components']} QA components failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  QA suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 QA suite failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()