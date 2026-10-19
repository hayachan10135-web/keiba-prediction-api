# -*- coding: utf-8 -*-
"""
backend-render/run_tests.py

全自動テストスイート一括実行スクリプト
"""

import unittest
import sys
import os

# backend-render をパスに追加
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


def main():
    print("=" * 60)
    print("🏇 KEIBA AI AUTOMATED TEST SUITE RUNNER")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(backend_dir, "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! (DoD 適合)")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print(f"❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
