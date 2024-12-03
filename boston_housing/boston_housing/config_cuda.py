
# import sys
# from pathlib import Path

# # Get the root project directory (parent directory of boston-housing)
# project_root = Path(__file__).resolve().parent.parent

# # Add the src directory to the Python path
# src_path = project_root / "src"

# print(f"Adding to sys.path: {src_path}")

# sys.path.insert(0, str(src_path))

# print("sys.path:", sys.path)
# Now import from src.config_cuda
from src.config_cuda import *