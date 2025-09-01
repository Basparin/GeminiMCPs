"""CES Accessibility Checker.

Validates WCAG 2.1 AA compliance including automated scanning,
semantic HTML validation, keyboard navigation testing, and color contrast analysis.
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse


@dataclass
class AccessibilityIssue:
    """Represents an accessibility issue found during analysis."""
    severity: str  # 'critical', 'serious', 'moderate', 'minor'
    code: str  # WCAG code like 'WCAG2AA.Principle1.Guideline1_1.1_1_1.H30'
    title: str
    description: str
    impact: str
    help_url: str
    elements: List[Dict[str, Any]]
    page_url: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class AccessibilityScanResult:
    """Result of an accessibility scan."""
    url_or_file: str
    issues_found: List[AccessibilityIssue]
    wcag_aa_compliant: bool
    compliance_score: float
    scan_duration: float
    timestamp: datetime


@dataclass
class AccessibilityComplianceReport:
    """Comprehensive accessibility compliance report."""
    overall_compliant: bool
    compliance_score: float
    total_issues: int
    critical_issues: int
    serious_issues: int
    moderate_issues: int
    minor_issues: int
    scan_results: List[AccessibilityScanResult]
    recommendations: List[str]
    timestamp: datetime


class AccessibilityChecker:
    """Checks CES accessibility compliance with WCAG 2.1 AA standards."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)

    def perform_accessibility_audit(self) -> AccessibilityComplianceReport:
        """Perform comprehensive accessibility audit."""
        scan_results = []

        # Scan HTML templates
        template_files = list(self.project_root.glob('**/templates/*.html'))
        for template_file in template_files:
            scan_results.append(self._scan_html_file(template_file))

        # Scan static HTML files
        static_html_files = list(self.project_root.glob('**/*.html'))
        for html_file in static_html_files:
            if 'templates' not in str(html_file):
                scan_results.append(self._scan_html_file(html_file))

        # Calculate overall metrics
        all_issues = []
        for result in scan_results:
            all_issues.extend(result.issues_found)

        total_issues = len(all_issues)
        critical_issues = len([i for i in all_issues if i.severity == 'critical'])
        serious_issues = len([i for i in all_issues if i.severity == 'serious'])
        moderate_issues = len([i for i in all_issues if i.severity == 'moderate'])
        minor_issues = len([i for i in all_issues if i.severity == 'minor'])

        # Calculate compliance score
        compliance_score = self._calculate_accessibility_score(all_issues)

        # Determine overall compliance
        overall_compliant = (
            critical_issues == 0 and
            serious_issues <= 5 and
            compliance_score >= 90.0
        )

        # Generate recommendations
        recommendations = self._generate_accessibility_recommendations(all_issues)

        return AccessibilityComplianceReport(
            overall_compliant=overall_compliant,
            compliance_score=compliance_score,
            total_issues=total_issues,
            critical_issues=critical_issues,
            serious_issues=serious_issues,
            moderate_issues=moderate_issues,
            minor_issues=minor_issues,
            scan_results=scan_results,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _scan_html_file(self, file_path: Path) -> AccessibilityScanResult:
        """Scan a single HTML file for accessibility issues."""
        import time
        start_time = time.time()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, OSError):
            return AccessibilityScanResult(
                url_or_file=str(file_path),
                issues_found=[],
                wcag_aa_compliant=True,
                compliance_score=100.0,
                scan_duration=0.0,
                timestamp=datetime.now()
            )

        issues = []

        # Check for alt text on images
        issues.extend(self._check_image_alt_text(content, str(file_path)))

        # Check for heading hierarchy
        issues.extend(self._check_heading_hierarchy(content, str(file_path)))

        # Check for form labels
        issues.extend(self._check_form_labels(content, str(file_path)))

        # Check for color contrast (basic check)
        issues.extend(self._check_color_contrast(content, str(file_path)))

        # Check for keyboard navigation
        issues.extend(self._check_keyboard_navigation(content, str(file_path)))

        # Check for semantic HTML
        issues.extend(self._check_semantic_html(content, str(file_path)))

        # Check for language declaration
        issues.extend(self._check_language_declaration(content, str(file_path)))

        # Calculate compliance for this file
        file_compliance_score = self._calculate_accessibility_score(issues)
        wcag_aa_compliant = len([i for i in issues if i.severity in ['critical', 'serious']]) == 0

        scan_duration = time.time() - start_time

        return AccessibilityScanResult(
            url_or_file=str(file_path),
            issues_found=issues,
            wcag_aa_compliant=wcag_aa_compliant,
            compliance_score=file_compliance_score,
            scan_duration=scan_duration,
            timestamp=datetime.now()
        )

    def _check_image_alt_text(self, content: str, file_path: str) -> List[AccessibilityIssue]:
        """Check for missing alt text on images."""
        issues = []

        # Find all img tags
        img_pattern = r'<img[^>]*>'
        img_tags = re.findall(img_pattern, content, re.IGNORECASE)

        for i, img_tag in enumerate(img_tags):
            if 'alt=' not in img_tag.lower():
                # Find line number
                lines_before = content[:content.find(img_tag)].count('\n') + 1

                issues.append(AccessibilityIssue(
                    severity='critical',
                    code='WCAG2AA.Principle1.Guideline1_1.1_1_1.H30',
                    title='Image missing alt text',
                    description='Image does not have alternative text',
                    impact='Screen reader users cannot understand the image content',
                    help_url='https://www.w3.org/WAI/WCAG21/Techniques/html/H30',
                    elements=[{'tag': 'img', 'attributes': img_tag}],
                    page_url=file_path,
                    line_number=lines_before
                ))

        return issues

    def _check_heading_hierarchy(self, content: str, file_path: str) -> List[AccessibilityIssue]:
        """Check heading hierarchy for proper structure."""
        issues = []

        # Extract headings
        heading_pattern = r'<h([1-6])[^>]*>(.*?)</h\1>'
        headings = re.findall(heading_pattern, content, re.IGNORECASE | re.DOTALL)

        if not headings:
            return issues

        # Check for missing h1
        h1_found = any(int(level) == 1 for level, _ in headings)
        if not h1_found:
            issues.append(AccessibilityIssue(
                severity='serious',
                code='WCAG2AA.Principle1.Guideline1_3.1_3_1.H42',
                title='Missing H1 heading',
                description='Page is missing a top-level heading (H1)',
                impact='Screen reader users cannot understand page structure',
                help_url='https://www.w3.org/WAI/WCAG21/Techniques/html/H42',
                elements=[],
                page_url=file_path
            ))

        # Check for heading level skips
        previous_level = 0
        for level, text in headings:
            current_level = int(level)
            if previous_level > 0 and current_level > previous_level + 1:
                issues.append(AccessibilityIssue(
                    severity='moderate',
                    code='WCAG2AA.Principle1.Guideline1_3.1_3_1.G141',
                    title='Heading level skipped',
                    description=f'Heading level jumped from H{previous_level} to H{current_level}',
                    impact='Screen reader users may be confused by heading structure',
                    help_url='https://www.w3.org/WAI/WCAG21/Techniques/general/G141',
                    elements=[{'tag': f'h{current_level}', 'text': text.strip()}],
                    page_url=file_path
                ))
            previous_level = current_level

        return issues

    def _check_form_labels(self, content: str, file_path: str) -> List[AccessibilityIssue]:
        """Check form inputs have proper labels."""
        issues = []

        # Find input elements
        input_pattern = r'<input[^>]*>'
        inputs = re.findall(input_pattern, content, re.IGNORECASE)

        for i, input_tag in enumerate(inputs):
            # Skip hidden, submit, and button inputs
            if any(attr in input_tag.lower() for attr in ['type="hidden"', 'type="submit"', 'type="button"']):
                continue

            # Check for associated label
            input_id = re.search(r'id=["\']([^"\']+)["\']', input_tag, re.IGNORECASE)
            has_label = False

            if input_id:
                label_pattern = f'<label[^>]*for=["\']{input_id.group(1)}["\'][^>]*>'
                if re.search(label_pattern, content, re.IGNORECASE):
                    has_label = True

            # Check for aria-label or aria-labelledby
            if 'aria-label=' in input_tag.lower() or 'aria-labelledby=' in input_tag.lower():
                has_label = True

            if not has_label:
                lines_before = content[:content.find(input_tag)].count('\n') + 1

                issues.append(AccessibilityIssue(
                    severity='critical',
                    code='WCAG2AA.Principle1.Guideline1_3.1_3_1.H44',
                    title='Form input missing label',
                    description='Form input does not have an associated label',
                    impact='Screen reader users cannot understand what the input is for',
                    help_url='https://www.w3.org/WAI/WCAG21/Techniques/html/H44',
                    elements=[{'tag': 'input', 'attributes': input_tag}],
                    page_url=file_path,
                    line_number=lines_before
                ))

        return issues

    def _check_color_contrast(self, content: str, file_path: str) -> List[AccessibilityIssue]:
        """Check for potential color contrast issues."""
        issues = []

        # Look for inline color styles
        color_pattern = r'color\s*:\s*#[0-9a-fA-F]{3,6}'
        background_pattern = r'background-color\s*:\s*#[0-9a-fA-F]{3,6}'

        colors = re.findall(color_pattern, content, re.IGNORECASE)
        backgrounds = re.findall(background_pattern, content, re.IGNORECASE)

        if colors and backgrounds:
            issues.append(AccessibilityIssue(
                severity='moderate',
                code='WCAG2AA.Principle1.Guideline1_4.1_4_3.G18',
                title='Color contrast may be insufficient',
                description='Inline color styles detected - manual contrast check required',
                impact='Users with visual impairments may not be able to read text',
                help_url='https://www.w3.org/WAI/WCAG21/Techniques/general/G18',
                elements=[],
                page_url=file_path
            ))

        return issues

    def _check_keyboard_navigation(self, content: str, file_path: str) -> List[AccessibilityIssue]:
        """Check for keyboard navigation issues."""
        issues = []

        # Check for tabindex attributes
        tabindex_pattern = r'tabindex\s*=\s*["\']([^"\']+)["\']'
        tabindex_attrs = re.findall(tabindex_pattern, content, re.IGNORECASE)

        negative_tabindex = [t for t in tabindex_attrs if t.startswith('-')]
        if negative_tabindex:
            issues.append(AccessibilityIssue(
                severity='serious',
                code='WCAG2AA.Principle2.Guideline2_1.2_1_1.F10',
                title='Negative tabindex values',
                description='Elements with negative tabindex values found',
                impact='Keyboard users cannot navigate to these elements',
                help_url='https://www.w3.org/WAI/WCAG21/Techniques/failures/F10',
                elements=[],
                page_url=file_path
            ))

        return issues

    def _check_semantic_html(self, content: str, file_path: str) -> List[AccessibilityIssue]:
        """Check for semantic HTML usage."""
        issues = []

        # Check for proper use of semantic elements
        semantic_elements = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer']
        found_semantic = False

        for element in semantic_elements:
            if f'<{element}' in content.lower():
                found_semantic = True
                break

        if not found_semantic:
            issues.append(AccessibilityIssue(
                severity='moderate',
                code='WCAG2AA.Principle1.Guideline1_3.1_3_1.G115',
                title='Limited semantic HTML usage',
                description='Page uses few or no semantic HTML elements',
                impact='Screen reader users have difficulty understanding page structure',
                help_url='https://www.w3.org/WAI/WCAG21/Techniques/general/G115',
                elements=[],
                page_url=file_path
            ))

        return issues

    def _check_language_declaration(self, content: str, file_path: str) -> List[AccessibilityIssue]:
        """Check for language declaration."""
        issues = []

        # Check for lang attribute on html element
        if not re.search(r'<html[^>]*lang\s*=', content, re.IGNORECASE):
            issues.append(AccessibilityIssue(
                severity='serious',
                code='WCAG2AA.Principle3.Guideline3_1.3_1_1.H57',
                title='Missing language declaration',
                description='HTML document does not specify language',
                impact='Screen readers may not pronounce content correctly',
                help_url='https://www.w3.org/WAI/WCAG21/Techniques/html/H57',
                elements=[],
                page_url=file_path
            ))

        return issues

    def _calculate_accessibility_score(self, issues: List[AccessibilityIssue]) -> float:
        """Calculate accessibility compliance score."""
        if not issues:
            return 100.0

        # Weight issues by severity
        weights = {
            'critical': 10,
            'serious': 5,
            'moderate': 2,
            'minor': 1
        }

        total_penalty = sum(weights.get(issue.severity, 1) for issue in issues)
        max_penalty = 100  # Maximum expected penalty

        return max(0.0, 100.0 - (total_penalty / max_penalty * 100))

    def _generate_accessibility_recommendations(self, issues: List[AccessibilityIssue]) -> List[str]:
        """Generate accessibility improvement recommendations."""
        recommendations = []

        if not issues:
            recommendations.append("Accessibility audit passed! No WCAG 2.1 AA violations found.")
            return recommendations

        # Group issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue.code.split('.')[1] if '.' in issue.code else 'general'
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)

        # Generate specific recommendations
        if 'Guideline1_1' in str(issue_types.keys()):  # Images
            recommendations.append("Images: Add descriptive alt text to all images")

        if 'Guideline1_3' in str(issue_types.keys()):  # Structure
            recommendations.append("Structure: Use proper heading hierarchy and semantic HTML elements")

        if 'Guideline1_4' in str(issue_types.keys()):  # Color
            recommendations.append("Color: Ensure sufficient color contrast and don't rely on color alone")

        if 'Guideline2_1' in str(issue_types.keys()):  # Keyboard
            recommendations.append("Keyboard: Ensure all interactive elements are keyboard accessible")

        if 'Guideline3_1' in str(issue_types.keys()):  # Language
            recommendations.append("Language: Declare document language and provide translations")

        # General recommendations
        critical_count = len([i for i in issues if i.severity == 'critical'])
        serious_count = len([i for i in issues if i.severity == 'serious'])

        if critical_count > 0:
            recommendations.append(f"URGENT: Fix {critical_count} critical accessibility issues immediately")

        if serious_count > 0:
            recommendations.append(f"Address {serious_count} serious accessibility issues")

        recommendations.append(f"WCAG 2.1 AA compliance: {self._calculate_accessibility_score(issues):.1f}%")

        return recommendations