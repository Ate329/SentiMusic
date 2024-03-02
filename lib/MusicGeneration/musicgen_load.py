from transformers import MusicgenForConditionalGeneration
import torch


def load_model(size='small'):
    model = MusicgenForConditionalGeneration.from_pretrained(f"facebook/musicgen-{size}")

    return model

# place the model on the accelerator device (if available)
def accelerator(model):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device);

    return device
