# %%
# Scoring of experts in relation to the number of validated reports

import uuid

from log import get_logger
from utils import get_config, get_config_filenames
from ingest import Ingest

# %load_ext autoreload
# %autoreload 2

# Get configuration parameters and logger for the current session
path = "./config/"
config = get_config(["config.ini"], path=path)
config = get_config(get_config_filenames(config["params"]["env"], path))

config.set("logging", "UIID", uuid.uuid4().hex)
with open("./config/.config.session.ini", "w") as fh:
    config.write(fh)

logger = get_logger(config, "main-pipeline")
logger_alert = get_logger(
    config,
    "main-pipeline-alert",
    email_alert=config.getboolean("logging", "send_email"),
)

try:
    # Get raw data from production server
    logger.info("Data extraction step...")
    data = Ingest(config, logger=get_logger(config, "ingest"))
    data.extract(conn_type=config["params"]["conn_type"])

    # Transform the extracted data
    logger.info("Data transformation step...")
    data.transform(refresh_codes=False)

except Exception as exc:
    logger_alert.error(f"Data ingestion terminated with error: {exc}")
