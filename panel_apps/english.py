from src import app

from src.utils import get_config
from src.log import get_logger

config_logger = get_config([".config.session.ini"], path="./config/")
logger = get_logger(config_logger, __name__)

title = "MA Analytic Dashboard"
logger.info(f"Start the {title} application.")
covidEN = app.MosquitoAlertExplorer().view
covidEN.servable(title=title)
