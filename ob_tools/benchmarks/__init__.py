"""
This __init__.py file enables auto-discovery of benchmarks.

It iterates through all Python files in the 'benchmarks' directory and imports
them, which triggers the @register_benchmark decorator for any benchmark
classes defined in those files.
"""

import os
import importlib
import pkgutil

# Discover and import all modules in the current package
for _, name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    importlib.import_module(f'.{name}', __package__)




