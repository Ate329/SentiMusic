from transformers import MusicgenForConditionalGeneration
from lib.logging_config import config
import torch

logger = config()

def load_model(size='small'):
    logger.info("Loading Music Generation model...")
    
    model = MusicgenForConditionalGeneration.from_pretrained(
        f"facebook/musicgen-{size}")

    logger.info("Success!")
    
    return model


# place the model on the accelerator device (if available)
def accelerator(model):
    logger.info("Enabling accelerator...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info("Success!")
    
    return device
