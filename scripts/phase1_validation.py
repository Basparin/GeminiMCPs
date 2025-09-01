#!/usr/bin/env python3
"""
CES Phase 1 Performance Validation Script

This script validates all Phase 1 performance benchmarks and generates
a comprehensive optimization summary with before/after metrics.

Usage:
    python scripts/phase1_validation.py

Requirements:
    - CES system running
    - All dependencies installed
    - Performance monitoring enabled
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Add CES to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ces.core.performance_optimizer import run_phase1_optimization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main validation function"""
    logger.info("Starting CES Phase 1 Performance Validation")
    logger.info("=" * 60)

    try:
        # Run Phase 1 optimization and validation
        logger.info("Running Phase 1 optimization sequence...")
        results = await run_phase1_optimization()

        # Save detailed results
        output_file = Path("benchmark_results/phase1_validation_results.json")
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Detailed results saved to {output_file}")

        # Generate summary report
        summary = generate_summary_report(results)
        print("\n" + "=" * 80)
        print("CES PHASE 1 PERFORMANCE OPTIMIZATION SUMMARY")
        print("=" * 80)
        print(summary)
        print("=" * 80)

        # Save summary report
        summary_file = Path("benchmark_results/phase1_summary_report.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)

        logger.info(f"Summary report saved to {summary_file}")
        logger.info("Phase 1 validation completed successfully")

    except Exception as e:
        logger.error(f"Phase 1 validation failed: {e}")
        raise


def generate_summary_report(results: dict) -> str:
    """Generate a comprehensive summary report"""
    lines = []

    # Header
    lines.append("CES PHASE 1 PERFORMANCE OPTIMIZATION RESULTS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overall achievement
    overall_achievement = results.get('overall_achievement', 0)
    lines.append(f"OVERALL ACHIEVEMENT: {overall_achievement:.1f}%")
    lines.append("")

    # Benchmark validation results
    lines.append("PHASE 1 BENCHMARK VALIDATION:")
    lines.append("-" * 50)

    validation_results = results.get('validation_results', {})
    targets = {
        'response_time_p50_simple': 'P50 Simple Response Time',
        'response_time_p95_complex': 'P95 Complex Response Time',
        'throughput_sustained': 'Sustained Throughput',
        'throughput_peak': 'Peak Throughput',
        'memory_normal': 'Normal Memory Usage',
        'memory_peak': 'Peak Memory Usage',
        'cpu_normal': 'Normal CPU Usage',
        'cpu_peak': 'Peak CPU Usage',
        'ai_grok_response': 'Grok Response Time',
        'ai_gemini_response': 'Gemini Response Time',
        'ai_qwen_response': 'Qwen Response Time',
        'memory_search_latency': 'Memory Search Latency',
        'memory_utilization': 'Memory Utilization',
        'cache_hit_rate': 'Cache Hit Rate'
    }

    for metric_key, display_name in targets.items():
        if metric_key in validation_results:
            result = validation_results[metric_key]
            status = "✓ ACHIEVED" if result['achieved'] else "✗ FAILED"
            target = result['target']
            current = result['current']
            variance = result['variance_percent']

            if metric_key in ['memory_normal', 'memory_peak', 'memory_search_latency']:
                unit = "MB" if "memory" in metric_key and "latency" not in metric_key else "ms"
                lines.append(f"{display_name}: {status}")
                lines.append(f"  Target: {target}{unit}, Current: {current:.1f}{unit}, Variance: {variance:+.1f}%")
            elif metric_key in ['cpu_normal', 'cpu_peak', 'memory_utilization', 'cache_hit_rate']:
                unit = "%"
                lines.append(f"{display_name}: {status}")
                lines.append(f"  Target: {target}{unit}, Current: {current:.1f}{unit}, Variance: {variance:+.1f}%")
            elif "throughput" in metric_key:
                unit = " req/min"
                lines.append(f"{display_name}: {status}")
                lines.append(f"  Target: {target}{unit}, Current: {current:.1f}{unit}, Variance: {variance:+.1f}%")
            else:
                unit = "ms"
                lines.append(f"{display_name}: {status}")
                lines.append(f"  Target: {target}{unit}, Current: {current:.1f}{unit}, Variance: {variance:+.1f}%")
            lines.append("")

    # Optimization results summary
    lines.append("OPTIMIZATION IMPLEMENTATIONS:")
    lines.append("-" * 50)

    optimization_results = results.get('optimization_results', {})

    if 'database_optimizations' in optimization_results:
        lines.append("✓ Database Operations Optimized:")
        db_opts = optimization_results['database_optimizations']
        lines.append(f"  - Connection pool size: {db_opts.get('connection_pool_size', 'N/A')}")
        lines.append(f"  - WAL mode: {'Enabled' if db_opts.get('wal_mode_enabled') else 'Disabled'}")
        lines.append(f"  - Cache size: {db_opts.get('cache_size_mb', 'N/A')}MB")
        lines.append("")

    if 'memory_optimizations' in optimization_results:
        lines.append("✓ Memory Management Optimized:")
        mem_opts = optimization_results['memory_optimizations']
        if 'memory_optimization_results' in mem_opts:
            mem_results = mem_opts['memory_optimization_results']
            reduction = mem_results.get('reduction_percentage', 0)
            lines.append(f"  - Memory reduction: {reduction:.1f}%")
        lines.append("  - FAISS indexing optimized")
        lines.append("")

    if 'ai_optimizations' in optimization_results:
        lines.append("✓ AI Integration Optimized:")
        ai_opts = optimization_results['ai_optimizations']
        lines.append(f"  - Groq target: {ai_opts.get('grok_optimizations', {}).get('target_response_time_ms', 'N/A')}ms")
        lines.append(f"  - Gemini target: {ai_opts.get('gemini_optimizations', {}).get('target_response_time_ms', 'N/A')}ms")
        lines.append(f"  - Qwen target: {ai_opts.get('qwen_optimizations', {}).get('target_response_time_ms', 'N/A')}ms")
        lines.append("")

    if 'concurrency_optimizations' in optimization_results:
        lines.append("✓ Concurrent Operations Optimized:")
        conc_opts = optimization_results['concurrency_optimizations']
        lines.append(f"  - Max concurrent operations: {conc_opts.get('max_concurrent_operations', 'N/A')}")
        lines.append(f"  - Throughput target: {conc_opts.get('throughput_target_req_per_min', 'N/A')} req/min")
        lines.append("")

    if 'cache_optimizations' in optimization_results:
        lines.append("✓ Caching System Optimized:")
        cache_opts = optimization_results['cache_optimizations']
        lines.append(f"  - Cache hit rate target: {cache_opts.get('cache_hit_rate_target', 'N/A')}%")
        lines.append(f"  - Adaptive caching: {'Enabled' if cache_opts.get('adaptive_cache_enabled') else 'Disabled'}")
        lines.append("")

    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        lines.append("OPTIMIZATION RECOMMENDATIONS:")
        lines.append("-" * 50)
        for rec in recommendations:
            lines.append(f"• {rec}")
        lines.append("")

    # Final status
    lines.append("PHASE 1 COMPLETION STATUS:")
    lines.append("-" * 50)

    if overall_achievement >= 90:
        lines.append("🎉 EXCELLENT: Phase 1 targets achieved with outstanding performance!")
    elif overall_achievement >= 75:
        lines.append("✅ GOOD: Phase 1 targets largely achieved with good performance.")
    elif overall_achievement >= 60:
        lines.append("⚠️  ACCEPTABLE: Phase 1 targets partially achieved, minor optimizations needed.")
    else:
        lines.append("❌ NEEDS IMPROVEMENT: Phase 1 targets not fully achieved, significant optimizations required.")

    lines.append("")
    lines.append(f"Next Steps: Address the {len(recommendations)} optimization recommendations above.")
    lines.append("For detailed metrics, see: benchmark_results/phase1_validation_results.json")

    return "\n".join(lines)


if __name__ == "__main__":
    # Run the validation
    asyncio.run(main())