import logging
from pathlib import Path

LOG_DIR = Path("logs/run_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "logs.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
