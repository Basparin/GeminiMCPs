"""CES Context Validator.

Validates context management implementation including working memory, task history,
user preferences, and semantic memory against Phase 1 requirements.
"""

import os
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging


@dataclass
class ContextValidationResult:
    """Result of a context validation check."""
    component: str
    check_name: str
    success: bool
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class ContextValidationReport:
    """Comprehensive context validation report."""
    overall_compliance: bool
    compliance_score: float
    total_validations: int
    passed_validations: int
    failed_validations: int
    component_results: Dict[str, List[ContextValidationResult]]
    recommendations: List[str]
    timestamp: datetime


class ContextValidator:
    """Validates CES context management implementation."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)
        self.logger = logging.getLogger(__name__)

        # Phase 1 context requirements
        self.requirements = {
            'working_memory': {
                'persistence': 'sqlite',
                'max_size': 100 * 1024 * 1024,  # 100MB
                'retention_days': 1,  # 24 hours
                'required_fields': ['user_id', 'session_id', 'data', 'timestamp', 'expires_at']
            },
            'task_history': {
                'persistence': 'sqlite',
                'max_size': 1 * 1024 * 1024 * 1024,  # 1GB
                'retention_days': 90,  # 90 days
                'indexing': True,
                'required_fields': ['task_id', 'user_id', 'description', 'status', 'created_at', 'updated_at']
            },
            'user_preferences': {
                'persistence': 'sqlite',
                'max_size': 100 * 1024 * 1024,  # 100MB
                'retention_days': -1,  # Indefinite
                'compression': True,
                'required_fields': ['user_id', 'preference_key', 'preference_value', 'updated_at']
            },
            'semantic_memory': {
                'engine': 'faiss',
                'max_size': 500 * 1024 * 1024,  # 500MB
                'retention_days': 180,  # 6 months
                'vector_dimension': 128,
                'required_fields': ['content', 'embedding', 'metadata', 'timestamp']
            }
        }

    def validate_all_context_management(self) -> ContextValidationReport:
        """Validate all context management components."""
        results = {}

        # Validate working memory
        results['working_memory'] = self._validate_working_memory()

        # Validate task history
        results['task_history'] = self._validate_task_history()

        # Validate user preferences
        results['user_preferences'] = self._validate_user_preferences()

        # Validate semantic memory
        results['semantic_memory'] = self._validate_semantic_memory()

        # Calculate overall metrics
        all_results = []
        for component_results in results.values():
            all_results.extend(component_results)

        total_validations = len(all_results)
        passed_validations = len([r for r in all_results if r.success])
        failed_validations = total_validations - passed_validations

        # Calculate compliance score
        compliance_score = (passed_validations / total_validations * 100) if total_validations > 0 else 0

        # Determine overall compliance
        overall_compliance = failed_validations == 0 and compliance_score >= 90.0

        # Generate recommendations
        recommendations = self._generate_context_recommendations(results)

        return ContextValidationReport(
            overall_compliance=overall_compliance,
            compliance_score=compliance_score,
            total_validations=total_validations,
            passed_validations=passed_validations,
            failed_validations=failed_validations,
            component_results=results,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _validate_working_memory(self) -> List[ContextValidationResult]:
        """Validate working memory implementation."""
        results = []

        # Check database file exists
        db_path = self.project_root / "ces_memory.db"
        if not db_path.exists():
            results.append(ContextValidationResult(
                component='working_memory',
                check_name='database_exists',
                success=False,
                details={'expected_path': str(db_path)},
                error_message='Working memory database file not found'
            ))
            return results

        try:
            # Connect to database
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check working memory table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='working_memory'")
            table_exists = cursor.fetchone() is not None

            results.append(ContextValidationResult(
                component='working_memory',
                check_name='table_exists',
                success=table_exists,
                details={'table_name': 'working_memory'},
                error_message='Working memory table not found' if not table_exists else None
            ))

            if table_exists:
                # Check table schema
                cursor.execute("PRAGMA table_info(working_memory)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]

                required_fields = self.requirements['working_memory']['required_fields']
                missing_fields = [field for field in required_fields if field not in column_names]

                results.append(ContextValidationResult(
                    component='working_memory',
                    check_name='schema_compliance',
                    success=len(missing_fields) == 0,
                    details={
                        'required_fields': required_fields,
                        'present_fields': column_names,
                        'missing_fields': missing_fields
                    },
                    error_message=f'Missing fields: {missing_fields}' if missing_fields else None
                ))

                # Check data retention
                cursor.execute("SELECT COUNT(*) FROM working_memory WHERE expires_at < datetime('now')")
                expired_count = cursor.fetchone()[0]

                results.append(ContextValidationResult(
                    component='working_memory',
                    check_name='data_retention',
                    success=expired_count == 0,
                    details={'expired_records': expired_count},
                    error_message=f'{expired_count} expired records found' if expired_count > 0 else None
                ))

                # Check database size
                db_size = os.path.getsize(str(db_path))
                max_size = self.requirements['working_memory']['max_size']

                results.append(ContextValidationResult(
                    component='working_memory',
                    check_name='size_limit',
                    success=db_size <= max_size,
                    details={
                        'current_size': db_size,
                        'max_size': max_size,
                        'usage_percent': (db_size / max_size) * 100
                    },
                    error_message=f'Database size {db_size} exceeds limit {max_size}' if db_size > max_size else None
                ))

            conn.close()

        except Exception as e:
            results.append(ContextValidationResult(
                component='working_memory',
                check_name='database_access',
                success=False,
                details={},
                error_message=f'Database access error: {str(e)}'
            ))

        return results

    def _validate_task_history(self) -> List[ContextValidationResult]:
        """Validate task history implementation."""
        results = []

        db_path = self.project_root / "ces_memory.db"
        if not db_path.exists():
            results.append(ContextValidationResult(
                component='task_history',
                check_name='database_exists',
                success=False,
                details={'expected_path': str(db_path)},
                error_message='Task history database file not found'
            ))
            return results

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check task history table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_history'")
            table_exists = cursor.fetchone() is not None

            results.append(ContextValidationResult(
                component='task_history',
                check_name='table_exists',
                success=table_exists,
                details={'table_name': 'task_history'},
                error_message='Task history table not found' if not table_exists else None
            ))

            if table_exists:
                # Check table schema
                cursor.execute("PRAGMA table_info(task_history)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]

                required_fields = self.requirements['task_history']['required_fields']
                missing_fields = [field for field in required_fields if field not in column_names]

                results.append(ContextValidationResult(
                    component='task_history',
                    check_name='schema_compliance',
                    success=len(missing_fields) == 0,
                    details={
                        'required_fields': required_fields,
                        'present_fields': column_names,
                        'missing_fields': missing_fields
                    },
                    error_message=f'Missing fields: {missing_fields}' if missing_fields else None
                ))

                # Check indexing
                cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='task_history'")
                indexes = cursor.fetchall()

                has_user_index = any('user_id' in str(index) for index in indexes)
                has_created_index = any('created_at' in str(index) for index in indexes)

                results.append(ContextValidationResult(
                    component='task_history',
                    check_name='indexing',
                    success=has_user_index and has_created_index,
                    details={
                        'indexes_found': len(indexes),
                        'user_index': has_user_index,
                        'created_index': has_created_index
                    },
                    error_message='Missing required indexes' if not (has_user_index and has_created_index) else None
                ))

                # Check data retention (90 days)
                retention_days = self.requirements['task_history']['retention_days']
                cursor.execute(f"SELECT COUNT(*) FROM task_history WHERE created_at < datetime('now', '-{retention_days} days')")
                old_records = cursor.fetchone()[0]

                results.append(ContextValidationResult(
                    component='task_history',
                    check_name='data_retention',
                    success=old_records == 0,
                    details={'old_records': old_records, 'retention_days': retention_days},
                    error_message=f'{old_records} records older than {retention_days} days' if old_records > 0 else None
                ))

            conn.close()

        except Exception as e:
            results.append(ContextValidationResult(
                component='task_history',
                check_name='database_access',
                success=False,
                details={},
                error_message=f'Database access error: {str(e)}'
            ))

        return results

    def _validate_user_preferences(self) -> List[ContextValidationResult]:
        """Validate user preferences implementation."""
        results = []

        db_path = self.project_root / "ces_memory.db"
        if not db_path.exists():
            results.append(ContextValidationResult(
                component='user_preferences',
                check_name='database_exists',
                success=False,
                details={'expected_path': str(db_path)},
                error_message='User preferences database file not found'
            ))
            return results

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check user preferences table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
            table_exists = cursor.fetchone() is not None

            results.append(ContextValidationResult(
                component='user_preferences',
                check_name='table_exists',
                success=table_exists,
                details={'table_name': 'user_preferences'},
                error_message='User preferences table not found' if not table_exists else None
            ))

            if table_exists:
                # Check table schema
                cursor.execute("PRAGMA table_info(user_preferences)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]

                required_fields = self.requirements['user_preferences']['required_fields']
                missing_fields = [field for field in required_fields if field not in column_names]

                results.append(ContextValidationResult(
                    component='user_preferences',
                    check_name='schema_compliance',
                    success=len(missing_fields) == 0,
                    details={
                        'required_fields': required_fields,
                        'present_fields': column_names,
                        'missing_fields': missing_fields
                    },
                    error_message=f'Missing fields: {missing_fields}' if missing_fields else None
                ))

                # Check data integrity (no indefinite retention check needed for preferences)
                cursor.execute("SELECT COUNT(*) FROM user_preferences")
                total_preferences = cursor.fetchone()[0]

                results.append(ContextValidationResult(
                    component='user_preferences',
                    check_name='data_integrity',
                    success=True,
                    details={'total_preferences': total_preferences},
                    error_message=None
                ))

            conn.close()

        except Exception as e:
            results.append(ContextValidationResult(
                component='user_preferences',
                check_name='database_access',
                success=False,
                details={},
                error_message=f'Database access error: {str(e)}'
            ))

        return results

    def _validate_semantic_memory(self) -> List[ContextValidationResult]:
        """Validate semantic memory implementation."""
        results = []

        # Check for FAISS index files
        faiss_files = list(self.project_root.glob("**/*.faiss"))
        faiss_exists = len(faiss_files) > 0

        results.append(ContextValidationResult(
            component='semantic_memory',
            check_name='faiss_index_exists',
            success=faiss_exists,
            details={'faiss_files_found': len(faiss_files), 'files': [str(f) for f in faiss_files]},
            error_message='No FAISS index files found' if not faiss_exists else None
        ))

        # Check for semantic memory metadata
        metadata_file = self.project_root / "semantic_memory_metadata.json"
        metadata_exists = metadata_file.exists()

        results.append(ContextValidationResult(
            component='semantic_memory',
            check_name='metadata_exists',
            success=metadata_exists,
            details={'metadata_file': str(metadata_file)},
            error_message='Semantic memory metadata file not found' if not metadata_exists else None
        ))

        if metadata_exists:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Check metadata structure
                required_keys = ['total_vectors', 'dimension', 'created_at', 'last_updated']
                missing_keys = [key for key in required_keys if key not in metadata]

                results.append(ContextValidationResult(
                    component='semantic_memory',
                    check_name='metadata_structure',
                    success=len(missing_keys) == 0,
                    details={
                        'metadata_keys': list(metadata.keys()),
                        'missing_keys': missing_keys,
                        'total_vectors': metadata.get('total_vectors', 0)
                    },
                    error_message=f'Missing metadata keys: {missing_keys}' if missing_keys else None
                ))

                # Check vector dimension
                dimension = metadata.get('dimension', 0)
                expected_dimension = self.requirements['semantic_memory']['vector_dimension']

                results.append(ContextValidationResult(
                    component='semantic_memory',
                    check_name='vector_dimension',
                    success=dimension == expected_dimension,
                    details={
                        'current_dimension': dimension,
                        'expected_dimension': expected_dimension
                    },
                    error_message=f'Vector dimension mismatch: {dimension} != {expected_dimension}' if dimension != expected_dimension else None
                ))

            except Exception as e:
                results.append(ContextValidationResult(
                    component='semantic_memory',
                    check_name='metadata_access',
                    success=False,
                    details={},
                    error_message=f'Metadata access error: {str(e)}'
                ))

        # Check semantic memory size
        total_size = 0
        for faiss_file in faiss_files:
            try:
                total_size += os.path.getsize(str(faiss_file))
            except OSError:
                pass

        max_size = self.requirements['semantic_memory']['max_size']

        results.append(ContextValidationResult(
            component='semantic_memory',
            check_name='size_limit',
            success=total_size <= max_size,
            details={
                'current_size': total_size,
                'max_size': max_size,
                'usage_percent': (total_size / max_size) * 100 if max_size > 0 else 0
            },
            error_message=f'Semantic memory size {total_size} exceeds limit {max_size}' if total_size > max_size else None
        ))

        return results

    def _generate_context_recommendations(self, results: Dict[str, List[ContextValidationResult]]) -> List[str]:
        """Generate context management improvement recommendations."""
        recommendations = []

        failed_checks = []
        for component_results in results.values():
            failed_checks.extend([r for r in component_results if not r.success])

        if not failed_checks:
            recommendations.append("Context management validation passed! All components are compliant.")
            return recommendations

        # Group failures by component
        component_failures = defaultdict(list)
        for check in failed_checks:
            component_failures[check.component].append(check)

        # Generate component-specific recommendations
        if 'working_memory' in component_failures:
            recommendations.append("Working Memory: Fix database schema, retention policy, and size limits")

        if 'task_history' in component_failures:
            recommendations.append("Task History: Ensure proper indexing, schema compliance, and data retention")

        if 'user_preferences' in component_failures:
            recommendations.append("User Preferences: Validate database schema and data integrity")

        if 'semantic_memory' in component_failures:
            recommendations.append("Semantic Memory: Set up FAISS indexing, metadata management, and size monitoring")

        # Overall recommendations
        total_failed = len(failed_checks)
        recommendations.append(f"Address {total_failed} context management validation failures")

        # Calculate compliance score
        all_results = []
        for component_results in results.values():
            all_results.extend(component_results)

        compliance_score = len([r for r in all_results if r.success]) / len(all_results) * 100 if all_results else 0
        recommendations.append(f"Context management compliance: {compliance_score:.1f}%")

        return recommendations