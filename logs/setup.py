import logging
import os
import sys

from utils import ROOT, load_config

_configured = False


def setup(level: int = logging.INFO) -> logging.Logger:
    global _configured
    logger = logging.getLogger("pgia")
    if _configured:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        cfg = load_config()
        log_dir = os.path.join(ROOT, cfg["paths"]["out"], "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "pgia.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as exc:  # logging must never break the app
        logger.warning("file logging disabled: %s", exc)

    _configured = True
    return logger
