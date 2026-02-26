import logging
import os
from datetime import datetime, timezone

# Generate a timestamped log file name
LOG_FILE_NAME = f"{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log"

# Define logs directory relative to project root
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

# Configure root logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)