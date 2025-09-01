#!/usr/bin/env python3
"""
CES Dashboard Runner

Starts the CES web dashboard with real-time monitoring and collaborative features.
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Add CES to path
sys.path.insert(0, str(Path(__file__).parent))

from ces.web.dashboard import dashboard


def main():
    """Main entry point for CES Dashboard"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting CES Dashboard...")

    try:
        # Run the dashboard
        dashboard.run(host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.info("CES Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error running CES Dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


if __name__ == "__main__":
    main()