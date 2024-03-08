from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
from audiocraft.data.audio import audio_write
import librosa
import logging
import torch
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
    
    if len(generated_audio.shape) == 1:
        generated_audio = generated_audio.unsqueeze(0)
    
    filename = "generated_music.wav"
    torchaudio.save(filename, audio, model.sample_rate)
    
    y, sr = librosa.load(filename)
    y_normalized = librosa.effects.normalize(y)
    librosa.output.write_wav(filename, y_normalized, sr)