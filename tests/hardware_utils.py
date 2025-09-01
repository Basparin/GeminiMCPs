"""
Hardware Detection Utility Module

This module provides functions to detect system hardware capabilities and generate
adaptive configuration profiles for testing. It includes safety mechanisms and
resource monitoring to ensure tests run appropriately based on available resources.

Usage:
    from tests.hardware_utils import detect_cpu_cores, detect_available_ram, get_hardware_profile

    cpu_count = detect_cpu_cores()
    ram_gb = detect_available_ram()
    profile = get_hardware_profile()
"""

import logging
import os
import psutil
from typing import Dict, Literal, Optional

# Configure logging for safety warnings
logger = logging.getLogger(__name__)

HardwareProfile = Literal['light', 'medium', 'full']

# Thresholds for hardware profiles (adjustable)
LIGHT_CPU_THRESHOLD = 2
LIGHT_RAM_THRESHOLD_GB = 4
MEDIUM_CPU_THRESHOLD = 8
MEDIUM_RAM_THRESHOLD_GB = 16

# Safety thresholds (minimum required for basic operation)
MIN_CPU_CORES = 1
MIN_RAM_GB = 1

def detect_cpu_cores() -> int:
    """
    Detect the number of logical CPU cores available.

    Returns:
        int: Number of logical CPU cores. Returns 1 if detection fails.

    Raises:
        RuntimeError: If CPU detection fails and no fallback is available.
    """
    try:
        cpu_count = psutil.cpu_count(logical=True)
        if cpu_count is None:
            cpu_count = os.cpu_count()
        if cpu_count is None:
            raise RuntimeError("Unable to detect CPU cores")
        return cpu_count
    except Exception as e:
        logger.warning(f"CPU detection failed: {e}. Using fallback value of 1.")
        return 1

def detect_available_ram() -> float:
    """
    Detect the available RAM in gigabytes.

    Returns:
        float: Available RAM in GB. Returns 1.0 if detection fails.

    Raises:
        RuntimeError: If RAM detection fails and no fallback is available.
    """
    try:
        available_ram_bytes = psutil.virtual_memory().available
        available_ram_gb = available_ram_bytes / (1024 ** 3)
        return round(available_ram_gb, 2)
    except Exception as e:
        logger.warning(f"RAM detection failed: {e}. Using fallback value of 1.0 GB.")
        return 1.0

def detect_total_ram() -> float:
    """
    Detect the total RAM in gigabytes.

    Returns:
        float: Total RAM in GB. Returns 1.0 if detection fails.
    """
    try:
        total_ram_bytes = psutil.virtual_memory().total
        total_ram_gb = total_ram_bytes / (1024 ** 3)
        return round(total_ram_gb, 2)
    except Exception as e:
        logger.warning(f"Total RAM detection failed: {e}. Using fallback value of 1.0 GB.")
        return 1.0

def get_hardware_profile() -> HardwareProfile:
    """
    Determine the hardware profile based on CPU cores and available RAM.

    Profiles:
        - 'light': Low-end hardware (≤2 CPU cores or ≤4GB RAM)
        - 'medium': Mid-range hardware (≤8 CPU cores or ≤16GB RAM)
        - 'full': High-end hardware (>8 CPU cores and >16GB RAM)

    Returns:
        HardwareProfile: The determined hardware profile.
    """
    cpu_cores = detect_cpu_cores()
    available_ram = detect_available_ram()

    if cpu_cores <= LIGHT_CPU_THRESHOLD or available_ram <= LIGHT_RAM_THRESHOLD_GB:
        return 'light'
    elif cpu_cores <= MEDIUM_CPU_THRESHOLD or available_ram <= MEDIUM_RAM_THRESHOLD_GB:
        return 'medium'
    else:
        return 'full'

def monitor_resources() -> Dict[str, float]:
    """
    Monitor current system resource usage.

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'cpu_percent': Current CPU usage percentage
            - 'ram_percent': Current RAM usage percentage
            - 'ram_available_gb': Available RAM in GB
            - 'cpu_cores': Number of CPU cores
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_available_gb = ram.available / (1024 ** 3)
        cpu_cores = detect_cpu_cores()

        return {
            'cpu_percent': cpu_percent,
            'ram_percent': ram_percent,
            'ram_available_gb': round(ram_available_gb, 2),
            'cpu_cores': cpu_cores
        }
    except Exception as e:
        logger.error(f"Resource monitoring failed: {e}")
        return {
            'cpu_percent': 0.0,
            'ram_percent': 0.0,
            'ram_available_gb': 0.0,
            'cpu_cores': 1
        }

def check_safety_requirements(profile: Optional[HardwareProfile] = None) -> bool:
    """
    Check if the system meets minimum safety requirements for testing.

    Args:
        profile: Optional hardware profile to check against. If None, uses detected profile.

    Returns:
        bool: True if requirements are met, False otherwise.

    Logs warnings if requirements are not met.
    """
    if profile is None:
        profile = get_hardware_profile()

    cpu_cores = detect_cpu_cores()
    available_ram = detect_available_ram()

    is_safe = True

    if cpu_cores < MIN_CPU_CORES:
        logger.warning(f"Insufficient CPU cores: {cpu_cores} (minimum: {MIN_CPU_CORES})")
        is_safe = False

    if available_ram < MIN_RAM_GB:
        logger.warning(f"Insufficient RAM: {available_ram}GB (minimum: {MIN_RAM_GB}GB)")
        is_safe = False

    # Additional checks based on profile
    if profile == 'light':
        if available_ram < LIGHT_RAM_THRESHOLD_GB:
            logger.warning(f"Light profile requires at least {LIGHT_RAM_THRESHOLD_GB}GB RAM, detected: {available_ram}GB")
            is_safe = False
    elif profile == 'medium':
        if available_ram < MEDIUM_RAM_THRESHOLD_GB:
            logger.warning(f"Medium profile requires at least {MEDIUM_RAM_THRESHOLD_GB}GB RAM, detected: {available_ram}GB")
            is_safe = False

    return is_safe

def get_adaptive_config(profile: HardwareProfile) -> Dict[str, any]:
    """
    Get adaptive configuration settings based on hardware profile.

    Args:
        profile: The hardware profile to base configuration on.

    Returns:
        Dict[str, any]: Configuration dictionary with adaptive settings.
    """
    base_config = {
        'max_workers': 1,
        'chunk_size': 1000,
        'cache_size_mb': 100,
        'timeout_seconds': 30,
        'memory_limit_mb': 512
    }

    if profile == 'light':
        base_config.update({
            'max_workers': 1,
            'chunk_size': 500,
            'cache_size_mb': 50,
            'timeout_seconds': 60,
            'memory_limit_mb': 256
        })
    elif profile == 'medium':
        base_config.update({
            'max_workers': 4,
            'chunk_size': 2000,
            'cache_size_mb': 200,
            'timeout_seconds': 45,
            'memory_limit_mb': 1024
        })
    elif profile == 'full':
        base_config.update({
            'max_workers': 8,
            'chunk_size': 5000,
            'cache_size_mb': 500,
            'timeout_seconds': 30,
            'memory_limit_mb': 2048
        })

    return base_config

def log_system_info():
    """
    Log detailed system information for debugging and monitoring.
    """
    cpu_cores = detect_cpu_cores()
    available_ram = detect_available_ram()
    total_ram = detect_total_ram()
    profile = get_hardware_profile()

    logger.info("System Hardware Detection:")
    logger.info(f"  CPU Cores: {cpu_cores}")
    logger.info(f"  Available RAM: {available_ram}GB")
    logger.info(f"  Total RAM: {total_ram}GB")
    logger.info(f"  Hardware Profile: {profile}")

# Convenience function to get all hardware info at once
def get_hardware_info() -> Dict[str, any]:
    """
    Get comprehensive hardware information.

    Returns:
        Dict[str, any]: Dictionary with all hardware detection results.
    """
    return {
        'cpu_cores': detect_cpu_cores(),
        'available_ram_gb': detect_available_ram(),
        'total_ram_gb': detect_total_ram(),
        'profile': get_hardware_profile(),
        'is_safe': check_safety_requirements(),
        'adaptive_config': get_adaptive_config(get_hardware_profile())
    }