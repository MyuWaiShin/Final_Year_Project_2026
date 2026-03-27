"""Shortcut: run from 13_03_2026/ root as `python 00_tcp_orient_test.py`"""
import runpy, sys, os
sys.argv[0] = os.path.join(os.path.dirname(__file__), "scripts", "00_tcp_orient_test.py")
runpy.run_path(sys.argv[0], run_name="__main__")
