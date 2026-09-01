#!/usr/bin/env python3
"""Compatibility entry point for saved-program evaluation."""

from __future__ import annotations

import sys

from train_gepa_v3 import main


if __name__ == "__main__":
    main(["--eval-only", *sys.argv[1:]])
