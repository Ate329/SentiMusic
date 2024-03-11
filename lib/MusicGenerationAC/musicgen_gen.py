from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
from audiocraft.data.audio import audio_write
from IPython.display import Audio
import librosa
import logging
import torch
import scipy
import torchaudio


def config():
    logging.config.fileConfig(
        fname='config.ini', disable_existing_loggers=False)

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    return logger


logger = config()


def text_conditional_gen(model, music_parameters, length, temperature=1.0, progress=True, top_k=250, top_p=0.0):
    model.set_generation_params(
        duration=length, temperature=temperature, top_k=top_k, top_p=top_p)

    audio = model.generate(descriptions=music_parameters, progress=progress)

    sampling_rate = model.sample_rate
    Audio(audio[0].cpu().numpy(), rate=sampling_rate)

    scipy.io.wavfile.write(
        "generated_music.wav", rate=sampling_rate, data=audio[0, 0].cpu().numpy())
