#!/usr/bin/env python3
"""sd_create_simulated_image -- interactive editor for simulated-image scenes.

This is the console-script dispatch entry point; the editor itself lives in the
:mod:`spindoctor.ui.sim_editor` package (one module per schema block).
"""

import os
import sys

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.ui.sim_editor import CreateSimulatedImageModel, main

__all__ = ['CreateSimulatedImageModel', 'main']


if __name__ == '__main__':
    main()
