from audiocraft.models import musicgen
import logging
import torch


def config():
    logging.config.fileConfig(
        fname='config.ini', disable_existing_loggers=False)

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    return logger


logger = config()


def load_model(size='small'):
    device = accelerator()

    logger.info("Loading Music Generation model through AudioCraft...")
    model = musicgen.MusicGen.get_pretrained(f"facebook/musicgen-{size}", device=device)

    logger.info("Success!")

    return model


def accelerator():
    logger.info("Enabling accelerator...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Success! Using device: {}".format(device))

    return device
