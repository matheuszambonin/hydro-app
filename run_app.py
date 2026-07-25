import inspect
import sys
from pathlib import Path

# 1. Force non-interactive Agg backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")

from streamlit.web import bootstrap

def main() -> None:
    # Resolve root directory path safely for PyInstaller frozen environment
    if getattr(sys, "frozen", False):
        base_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base_dir = Path(__file__).parent

    app_script = str(base_dir / "app.py")

    flag_options = {
        "global.developmentMode": False,
        "server.headless": True,
        "browser.gatherUsageStats": False,
    }

    # Safe introspection call to protect against Streamlit minor version updates
    params = inspect.signature(bootstrap.run).parameters
    if "is_hello" in params:
        bootstrap.run(app_script, False, [], flag_options)
    else:
        bootstrap.run(app_script, [], flag_options)

if __name__ == "__main__":
    main()