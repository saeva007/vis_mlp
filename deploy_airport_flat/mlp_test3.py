#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin business entrypoint for the airport Static-RNN visibility forecast.

Keep the old command line:
    python mlp_test3.py --config forecast_config.json --process-existing

Put this file in the same directory as:
    xiahang_forecast_system.py
    airport_visibility_common.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_runtime():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    try:
        from xiahang_forecast_system import (  # type: ignore
            VisibilityForecastSystem,
            create_default_config,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to import xiahang_forecast_system.py. "
            "Please put xiahang_forecast_system.py and "
            "airport_visibility_common.py next to mlp_test3.py."
        ) from exc
    return VisibilityForecastSystem, create_default_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Airport visibility forecast")
    parser.add_argument("--config", "-c", default="forecast_config.json")
    parser.add_argument("--create-config", action="store_true")
    parser.add_argument("--process-existing", action="store_true")
    parser.add_argument("--scan-all", action="store_true")
    parser.add_argument("--cleanup-days", type=int, default=None)
    args = parser.parse_args()

    VisibilityForecastSystem, create_default_config = _load_runtime()

    if args.create_config:
        create_default_config()
        return 0

    if not os.path.exists(args.config):
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        return 2

    system = VisibilityForecastSystem(args.config)
    if not system.load_model():
        print("ERROR: model loading failed; see visibility_forecast.log", file=sys.stderr)
        return 3

    if args.cleanup_days is not None:
        system.cleanup_old_forecasts(days_to_keep=args.cleanup_days)

    try:
        system.monitor_and_forecast(
            process_existing=args.process_existing,
            scan_only_new=not args.scan_all,
        )
    except KeyboardInterrupt:
        print("Interrupted, exiting.")
        return 130
    except Exception as exc:
        print(f"ERROR: forecast runtime failed: {exc}", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
