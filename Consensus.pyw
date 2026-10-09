"""Double-click launcher: opens the desktop app without a console window."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.chdir(here)

from consensus_app import main  # noqa: E402

main()
