"""CES Security Analyzer.

Performs comprehensive security audits including authentication, encryption,
access controls, HTTPS configuration, and firewall validation.
"""

import re
import hashlib
import secrets
import ssl
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse


@dataclass
class SecurityIssue:
    """Represents a security issue found during analysis."""
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    category: str
    title: str
    description: str
    impact: str
    recommendation: str
    cwe_id: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class SecurityScanResult:
    """Result of a security scan."""
    scan_type: str
    issues_found: List[SecurityIssue]
    compliance_score: float
    scan_duration: float
    timestamp: datetime


@dataclass
class SecurityComplianceReport:
    """Comprehensive security compliance report."""
    overall_compliance: bool
    compliance_score: float
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    scan_results: List[SecurityScanResult]
    recommendations: List[str]
    timestamp: datetime


class SecurityAnalyzer:
    """Analyzes CES security posture and compliance."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)

    def perform_comprehensive_security_audit(self) -> SecurityComplianceReport:
        """Perform comprehensive security audit."""
        scan_results = []

        # Authentication and Authorization Audit
        scan_results.append(self._audit_authentication())

        # Data Encryption Audit
        scan_results.append(self._audit_encryption())

        # Access Control Audit
        scan_results.append(self._audit_access_controls())

        # HTTPS Configuration Audit
        scan_results.append(self._audit_https_configuration())

        # Firewall Configuration Audit
        scan_results.append(self._audit_firewall_configuration())

        # Code Security Analysis
        scan_results.append(self._analyze_code_security())

        # Dependency Security Audit
        scan_results.append(self._audit_dependencies())

        # Calculate overall metrics
        all_issues = []
        for result in scan_results:
            all_issues.extend(result.issues_found)

        total_issues = len(all_issues)
        critical_issues = len([i for i in all_issues if i.severity == 'critical'])
        high_issues = len([i for i in all_issues if i.severity == 'high'])
        medium_issues = len([i for i in all_issues if i.severity == 'medium'])
        low_issues = len([i for i in all_issues if i.severity == 'low'])

        # Calculate compliance score
        compliance_score = self._calculate_security_score(all_issues)

        # Determine overall compliance
        overall_compliance = (
            critical_issues == 0 and
            high_issues <= 2 and
            compliance_score >= 85.0
        )

        # Generate recommendations
        recommendations = self._generate_security_recommendations(all_issues)

        return SecurityComplianceReport(
            overall_compliance=overall_compliance,
            compliance_score=compliance_score,
            total_issues=total_issues,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            scan_results=scan_results,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _audit_authentication(self) -> SecurityScanResult:
        """Audit authentication mechanisms."""
        import time
        start_time = time.time()

        issues = []

        # Check for MFA implementation
        mfa_found = self._check_mfa_implementation()
        if not mfa_found:
            issues.append(SecurityIssue(
                severity='high',
                category='authentication',
                title='Multi-Factor Authentication Not Implemented',
                description='No MFA implementation found in the codebase',
                impact='Single point of failure for user authentication',
                recommendation='Implement MFA for all user authentication flows',
                cwe_id='CWE-308'
            ))

        # Check password policies
        password_issues = self._check_password_policies()
        issues.extend(password_issues)

        # Check session management
        session_issues = self._check_session_management()
        issues.extend(session_issues)

        # Check for insecure authentication methods
        auth_issues = self._check_insecure_auth_methods()
        issues.extend(auth_issues)

        scan_duration = time.time() - start_time

        return SecurityScanResult(
            scan_type='authentication_audit',
            issues_found=issues,
            compliance_score=self._calculate_scan_score(issues),
            scan_duration=scan_duration,
            timestamp=datetime.now()
        )

    def _audit_encryption(self) -> SecurityScanResult:
        """Audit data encryption implementation."""
        import time
        start_time = time.time()

        issues = []

        # Check for encryption at rest
        encryption_issues = self._check_data_encryption()
        issues.extend(encryption_issues)

        # Check for encryption in transit
        transit_issues = self._check_encryption_in_transit()
        issues.extend(transit_issues)

        # Check cryptographic implementations
        crypto_issues = self._check_cryptographic_implementations()
        issues.extend(crypto_issues)

        scan_duration = time.time() - start_time

        return SecurityScanResult(
            scan_type='encryption_audit',
            issues_found=issues,
            compliance_score=self._calculate_scan_score(issues),
            scan_duration=scan_duration,
            timestamp=datetime.now()
        )

    def _audit_access_controls(self) -> SecurityScanResult:
        """Audit access control mechanisms."""
        import time
        start_time = time.time()

        issues = []

        # Check for proper authorization
        authz_issues = self._check_authorization()
        issues.extend(authz_issues)

        # Check for privilege escalation vulnerabilities
        privilege_issues = self._check_privilege_escalation()
        issues.extend(privilege_issues)

        # Check for insecure direct object references
        idor_issues = self._check_idor_vulnerabilities()
        issues.extend(idor_issues)

        scan_duration = time.time() - start_time

        return SecurityScanResult(
            scan_type='access_control_audit',
            issues_found=issues,
            compliance_score=self._calculate_scan_score(issues),
            scan_duration=scan_duration,
            timestamp=datetime.now()
        )

    def _audit_https_configuration(self) -> SecurityScanResult:
        """Audit HTTPS configuration."""
        import time
        start_time = time.time()

        issues = []

        # Check SSL/TLS configuration
        ssl_issues = self._check_ssl_configuration()
        issues.extend(ssl_issues)

        # Check for mixed content
        mixed_content_issues = self._check_mixed_content()
        issues.extend(mixed_content_issues)

        # Check HSTS configuration
        hsts_issues = self._check_hsts_configuration()
        issues.extend(hsts_issues)

        scan_duration = time.time() - start_time

        return SecurityScanResult(
            scan_type='https_audit',
            issues_found=issues,
            compliance_score=self._calculate_scan_score(issues),
            scan_duration=scan_duration,
            timestamp=datetime.now()
        )

    def _audit_firewall_configuration(self) -> SecurityScanResult:
        """Audit firewall configuration."""
        import time
        start_time = time.time()

        issues = []

        # Check firewall rules
        firewall_issues = self._check_firewall_rules()
        issues.extend(firewall_issues)

        # Check for open ports
        port_issues = self._check_open_ports()
        issues.extend(port_issues)

        # Check network segmentation
        network_issues = self._check_network_segmentation()
        issues.extend(network_issues)

        scan_duration = time.time() - start_time

        return SecurityScanResult(
            scan_type='firewall_audit',
            issues_found=issues,
            compliance_score=self._calculate_scan_score(issues),
            scan_duration=scan_duration,
            timestamp=datetime.now()
        )

    def _analyze_code_security(self) -> SecurityScanResult:
        """Analyze code for security vulnerabilities."""
        import time
        start_time = time.time()

        issues = []

        # Check for common security vulnerabilities
        vuln_issues = self._check_common_vulnerabilities()
        issues.extend(vuln_issues)

        # Check for insecure coding practices
        coding_issues = self._check_insecure_coding_practices()
        issues.extend(coding_issues)

        # Check for hardcoded secrets
        secret_issues = self._check_hardcoded_secrets()
        issues.extend(secret_issues)

        scan_duration = time.time() - start_time

        return SecurityScanResult(
            scan_type='code_security_analysis',
            issues_found=issues,
            compliance_score=self._calculate_scan_score(issues),
            scan_duration=scan_duration,
            timestamp=datetime.now()
        )

    def _audit_dependencies(self) -> SecurityScanResult:
        """Audit third-party dependencies for security vulnerabilities."""
        import time
        start_time = time.time()

        issues = []

        # Check Python dependencies
        python_dep_issues = self._check_python_dependencies()
        issues.extend(python_dep_issues)

        # Check for outdated dependencies
        outdated_issues = self._check_outdated_dependencies()
        issues.extend(outdated_issues)

        scan_duration = time.time() - start_time

        return SecurityScanResult(
            scan_type='dependency_audit',
            issues_found=issues,
            compliance_score=self._calculate_scan_score(issues),
            scan_duration=scan_duration,
            timestamp=datetime.now()
        )

    # Implementation methods for various checks
    def _check_mfa_implementation(self) -> bool:
        """Check if MFA is implemented."""
        # Look for MFA-related code patterns
        mfa_patterns = [
            r'multi.?factor',
            r'2fa',
            r'two.?factor',
            r'totp',
            r'sms.*auth',
            r'authenticator'
        ]

        for pattern in mfa_patterns:
            if self._search_codebase(pattern):
                return True

        return False

    def _check_password_policies(self) -> List[SecurityIssue]:
        """Check password policies."""
        issues = []

        # Check for weak password requirements
        if not self._search_codebase(r'password.*strength|password.*complexity'):
            issues.append(SecurityIssue(
                severity='medium',
                category='authentication',
                title='Weak Password Policy',
                description='No password strength requirements found',
                impact='Users can set weak passwords',
                recommendation='Implement password complexity requirements',
                cwe_id='CWE-521'
            ))

        return issues

    def _check_session_management(self) -> List[SecurityIssue]:
        """Check session management security."""
        issues = []

        # Check for secure session handling
        if self._search_codebase(r'session.*timeout|session.*expire'):
            # Good, session timeout is configured
            pass
        else:
            issues.append(SecurityIssue(
                severity='medium',
                category='authentication',
                title='Session Timeout Not Configured',
                description='No session timeout mechanism found',
                impact='Sessions may remain active indefinitely',
                recommendation='Implement session timeout and automatic logout',
                cwe_id='CWE-613'
            ))

        return issues

    def _check_insecure_auth_methods(self) -> List[SecurityIssue]:
        """Check for insecure authentication methods."""
        issues = []

        # Check for basic auth
        if self._search_codebase(r'basic.*auth|http.*basic'):
            issues.append(SecurityIssue(
                severity='high',
                category='authentication',
                title='Basic Authentication Detected',
                description='Basic HTTP authentication found in codebase',
                impact='Credentials transmitted in base64 encoding',
                recommendation='Use OAuth2, JWT, or other secure authentication methods',
                cwe_id='CWE-287'
            ))

        return issues

    def _check_data_encryption(self) -> List[SecurityIssue]:
        """Check data encryption implementation."""
        issues = []

        # Check for encryption libraries
        encryption_libs = ['cryptography', 'pycryptodome', 'fernet', 'aes']
        found_encryption = False

        for lib in encryption_libs:
            if self._search_codebase(lib):
                found_encryption = True
                break

        if not found_encryption:
            issues.append(SecurityIssue(
                severity='high',
                category='encryption',
                title='No Encryption Libraries Found',
                description='No encryption libraries detected in dependencies',
                impact='Data may not be properly encrypted',
                recommendation='Implement encryption for sensitive data using established libraries',
                cwe_id='CWE-311'
            ))

        return issues

    def _check_encryption_in_transit(self) -> List[SecurityIssue]:
        """Check encryption in transit."""
        issues = []

        # Check for HTTPS usage
        if not self._search_codebase(r'https|ssl|tls'):
            issues.append(SecurityIssue(
                severity='critical',
                category='encryption',
                title='No HTTPS Implementation',
                description='No HTTPS/SSL/TLS configuration found',
                impact='Data transmitted in plain text',
                recommendation='Implement HTTPS for all communications',
                cwe_id='CWE-319'
            ))

        return issues

    def _check_cryptographic_implementations(self) -> List[SecurityIssue]:
        """Check cryptographic implementations."""
        issues = []

        # Check for weak cryptographic algorithms
        weak_algorithms = [r'md5', r'sha1', r'des', r'rc4']
        for algorithm in weak_algorithms:
            if self._search_codebase(algorithm):
                issues.append(SecurityIssue(
                    severity='high',
                    category='encryption',
                    title=f'Weak Cryptographic Algorithm: {algorithm.upper()}',
                    description=f'Use of weak cryptographic algorithm {algorithm.upper()}',
                    impact='Data protected with weak encryption',
                    recommendation='Use strong cryptographic algorithms like AES-256, SHA-256',
                    cwe_id='CWE-327'
                ))

        return issues

    def _check_authorization(self) -> List[SecurityIssue]:
        """Check authorization mechanisms."""
        issues = []

        # Check for role-based access control
        if not self._search_codebase(r'role|permission|rbac|acl'):
            issues.append(SecurityIssue(
                severity='high',
                category='access_control',
                title='No Authorization Mechanism',
                description='No role-based access control or authorization found',
                impact='Users may access unauthorized resources',
                recommendation='Implement proper authorization and access control',
                cwe_id='CWE-284'
            ))

        return issues

    def _check_privilege_escalation(self) -> List[SecurityIssue]:
        """Check for privilege escalation vulnerabilities."""
        issues = []

        # Check for admin/sudo usage
        if self._search_codebase(r'admin|sudo|root'):
            issues.append(SecurityIssue(
                severity='medium',
                category='access_control',
                title='Potential Privilege Escalation',
                description='Administrative privileges detected in code',
                impact='Potential for privilege escalation attacks',
                recommendation='Implement principle of least privilege',
                cwe_id='CWE-250'
            ))

        return issues

    def _check_idor_vulnerabilities(self) -> List[SecurityIssue]:
        """Check for IDOR vulnerabilities."""
        issues = []

        # Look for direct object references without authorization
        if self._search_codebase(r'/user/\d+|/resource/\d+'):
            issues.append(SecurityIssue(
                severity='high',
                category='access_control',
                title='Potential IDOR Vulnerability',
                description='Direct object references found without authorization checks',
                impact='Users may access other users\' data',
                recommendation='Implement proper authorization checks for object access',
                cwe_id='CWE-639'
            ))

        return issues

    def _check_ssl_configuration(self) -> List[SecurityIssue]:
        """Check SSL/TLS configuration."""
        issues = []

        # Check for SSL context configuration
        if not self._search_codebase(r'ssl\.create_default_context|ssl\.SSLContext'):
            issues.append(SecurityIssue(
                severity='medium',
                category='https',
                title='SSL Context Not Configured',
                description='No SSL context configuration found',
                impact='Weak SSL/TLS configuration may be used',
                recommendation='Configure SSL context with secure settings',
                cwe_id='CWE-326'
            ))

        return issues

    def _check_mixed_content(self) -> List[SecurityIssue]:
        """Check for mixed content issues."""
        issues = []

        # This would require HTML parsing in a real implementation
        # For now, return empty list
        return issues

    def _check_hsts_configuration(self) -> List[SecurityIssue]:
        """Check HSTS configuration."""
        issues = []

        if not self._search_codebase(r'Strict-Transport-Security|HSTS'):
            issues.append(SecurityIssue(
                severity='medium',
                category='https',
                title='HSTS Not Configured',
                description='HTTP Strict Transport Security not implemented',
                impact='Vulnerable to protocol downgrade attacks',
                recommendation='Implement HSTS headers',
                cwe_id='CWE-319'
            ))

        return issues

    def _check_firewall_rules(self) -> List[SecurityIssue]:
        """Check firewall rules."""
        issues = []

        # This would require system-level checks in a real implementation
        # For now, return informational issue
        issues.append(SecurityIssue(
            severity='info',
            category='firewall',
            title='Firewall Configuration Check',
            description='Manual firewall configuration review required',
            impact='Unknown firewall configuration',
            recommendation='Review firewall rules and ensure proper network segmentation'
        ))

        return issues

    def _check_open_ports(self) -> List[SecurityIssue]:
        """Check for open ports."""
        issues = []

        # This would require system-level checks in a real implementation
        return issues

    def _check_network_segmentation(self) -> List[SecurityIssue]:
        """Check network segmentation."""
        issues = []

        # This would require network-level checks in a real implementation
        return issues

    def _check_common_vulnerabilities(self) -> List[SecurityIssue]:
        """Check for common security vulnerabilities."""
        issues = []

        # Check for SQL injection patterns
        if self._search_codebase(r'execute.*\+|format.*sql'):
            issues.append(SecurityIssue(
                severity='critical',
                category='code_security',
                title='Potential SQL Injection',
                description='String concatenation in SQL queries detected',
                impact='Database compromise through SQL injection',
                recommendation='Use parameterized queries or ORM',
                cwe_id='CWE-89'
            ))

        # Check for XSS patterns
        if self._search_codebase(r'innerHTML|outerHTML|document\.write'):
            issues.append(SecurityIssue(
                severity='high',
                category='code_security',
                title='Potential XSS Vulnerability',
                description='Direct HTML manipulation detected',
                impact='Cross-site scripting attacks possible',
                recommendation='Use safe HTML escaping and CSP headers',
                cwe_id='CWE-79'
            ))

        return issues

    def _check_insecure_coding_practices(self) -> List[SecurityIssue]:
        """Check for insecure coding practices."""
        issues = []

        # Check for eval usage
        if self._search_codebase(r'\beval\b'):
            issues.append(SecurityIssue(
                severity='high',
                category='code_security',
                title='Use of eval() Function',
                description='Dangerous eval() function usage detected',
                impact='Code injection and execution vulnerabilities',
                recommendation='Avoid eval() and use safe alternatives',
                cwe_id='CWE-95'
            ))

        # Check for pickle usage
        if self._search_codebase(r'\bpickle\b'):
            issues.append(SecurityIssue(
                severity='medium',
                category='code_security',
                title='Use of pickle Module',
                description='Pickle module usage detected',
                impact='Potential remote code execution',
                recommendation='Use safer serialization methods like JSON',
                cwe_id='CWE-502'
            ))

        return issues

    def _check_hardcoded_secrets(self) -> List[SecurityIssue]:
        """Check for hardcoded secrets."""
        issues = []

        # Check for API keys, passwords, etc.
        secret_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]

        for pattern in secret_patterns:
            if self._search_codebase(pattern):
                issues.append(SecurityIssue(
                    severity='high',
                    category='code_security',
                    title='Hardcoded Secrets Detected',
                    description='Potential hardcoded secrets found in code',
                    impact='Credential exposure and unauthorized access',
                    recommendation='Use environment variables or secure credential storage',
                    cwe_id='CWE-798'
                ))
                break  # Only report once

        return issues

    def _check_python_dependencies(self) -> List[SecurityIssue]:
        """Check Python dependencies for security issues."""
        issues = []

        # Check if requirements.txt exists
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            issues.append(SecurityIssue(
                severity='medium',
                category='dependencies',
                title='No Dependency Management',
                description='requirements.txt file not found',
                impact='Unmanaged dependencies may contain vulnerabilities',
                recommendation='Create requirements.txt and manage dependencies properly'
            ))
        else:
            # Check for known vulnerable packages (simplified check)
            vulnerable_packages = ['insecure-package', 'old-version-package']
            try:
                with open(requirements_file, 'r') as f:
                    content = f.read()

                for package in vulnerable_packages:
                    if package in content:
                        issues.append(SecurityIssue(
                            severity='high',
                            category='dependencies',
                            title=f'Vulnerable Package: {package}',
                            description=f'Known vulnerable package {package} found in dependencies',
                            impact='System compromise through known vulnerabilities',
                            recommendation='Update to secure version or replace with secure alternative'
                        ))
            except Exception as e:
                issues.append(SecurityIssue(
                    severity='low',
                    category='dependencies',
                    title='Dependency File Read Error',
                    description=f'Could not read requirements.txt: {str(e)}',
                    impact='Cannot verify dependency security',
                    recommendation='Ensure requirements.txt is readable and properly formatted'
                ))

        return issues

    def _check_outdated_dependencies(self) -> List[SecurityIssue]:
        """Check for outdated dependencies."""
        issues = []

        # This would require running pip-outdated or similar tools
        issues.append(SecurityIssue(
            severity='info',
            category='dependencies',
            title='Dependency Update Check',
            description='Manual dependency update check required',
            impact='Unknown outdated dependencies',
            recommendation='Run pip list --outdated and update dependencies regularly'
        ))

        return issues

    def _search_codebase(self, pattern: str) -> bool:
        """Search codebase for a pattern."""
        try:
            for file_path in self.project_root.rglob('*.py'):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if re.search(pattern, content, re.IGNORECASE):
                            return True
                    except (UnicodeDecodeError, OSError):
                        continue
        except Exception:
            pass

        return False

    def _calculate_scan_score(self, issues: List[SecurityIssue]) -> float:
        """Calculate compliance score for a scan."""
        if not issues:
            return 100.0

        # Weight issues by severity
        weights = {
            'critical': 20,
            'high': 10,
            'medium': 5,
            'low': 2,
            'info': 1
        }

        total_penalty = sum(weights.get(issue.severity, 1) for issue in issues)
        max_penalty = 100  # Maximum expected penalty

        return max(0.0, 100.0 - (total_penalty / max_penalty * 100))

    def _calculate_security_score(self, issues: List[SecurityIssue]) -> float:
        """Calculate overall security compliance score."""
        if not issues:
            return 100.0

        # Group by severity
        severity_counts = {}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        # Calculate weighted score
        weights = {
            'critical': 25,
            'high': 15,
            'medium': 8,
            'low': 3,
            'info': 1
        }

        total_penalty = 0
        for severity, count in severity_counts.items():
            total_penalty += weights.get(severity, 1) * count

        max_penalty = 200  # Maximum expected penalty

        return max(0.0, 100.0 - (total_penalty / max_penalty * 100))

    def _generate_security_recommendations(self, issues: List[SecurityIssue]) -> List[str]:
        """Generate security improvement recommendations."""
        recommendations = []

        if not issues:
            recommendations.append("Security audit passed! No critical security issues found.")
            return recommendations

        # Group issues by category
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

        # Generate category-specific recommendations
        if category_counts.get('authentication', 0) > 0:
            recommendations.append("Authentication: Implement MFA, strengthen password policies, and improve session management")

        if category_counts.get('encryption', 0) > 0:
            recommendations.append("Encryption: Implement proper data encryption at rest and in transit using strong algorithms")

        if category_counts.get('access_control', 0) > 0:
            recommendations.append("Access Control: Implement proper authorization, prevent privilege escalation, and fix IDOR vulnerabilities")

        if category_counts.get('https', 0) > 0:
            recommendations.append("HTTPS: Configure SSL/TLS properly, implement HSTS, and prevent mixed content")

        if category_counts.get('firewall', 0) > 0:
            recommendations.append("Firewall: Review firewall rules, close unnecessary ports, and implement network segmentation")

        if category_counts.get('code_security', 0) > 0:
            recommendations.append("Code Security: Fix SQL injection, XSS vulnerabilities, and remove hardcoded secrets")

        if category_counts.get('dependencies', 0) > 0:
            recommendations.append("Dependencies: Update vulnerable packages and implement dependency scanning")

        # Overall recommendations
        critical_count = len([i for i in issues if i.severity == 'critical'])
        high_count = len([i for i in issues if i.severity == 'high'])

        if critical_count > 0:
            recommendations.append(f"URGENT: Address {critical_count} critical security issues immediately")

        if high_count > 0:
            recommendations.append(f"Address {high_count} high-severity security issues")

        recommendations.append(f"Security compliance: {self._calculate_security_score(issues):.1f}%")

        return recommendations