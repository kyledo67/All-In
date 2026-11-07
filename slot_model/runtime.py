"""Select the vendored source explicitly, never an ambient Dynamax install."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/vendor/dynamax'))
sys.path.insert(0, str(ROOT / 'organizer'))
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import jax
jax.config.update('jax_enable_x64', True)
