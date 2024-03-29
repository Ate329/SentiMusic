import logging
import logging.config
from lib.streamlit_log import StreamlitLogHandler


def config():
    logging.config.fileConfig(fname='config.ini', disable_existing_loggers=False)

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    return logger

# logger.info('This is an INFO message')
