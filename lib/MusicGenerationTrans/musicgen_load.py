from transformers import MusicgenForConditionalGeneration
import torch
import logging
import logging.config
from lib.streamlit_log import StreamlitLogHandler


def config():
    logging.config.fileConfig(fname='config.ini', disable_existing_loggers=False)

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    return logger


logger = config()
handler = StreamlitLogHandler()
logger.addHandler(handler)


def load_model(size='small'):
    logger.info("Loading Music Generation model through Transformers...")

    model = MusicgenForConditionalGeneration.from_pretrained(
        f"facebook/musicgen-{size}")

    logger.info("Success!")

    return model


# place the model on the accelerator device (if available)
def accelerator(model):
    logger.info("Enabling accelerator...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info("Success! Using device: {}".format(device))

    return device
