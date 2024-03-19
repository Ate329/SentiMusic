from IPython.display import Audio
from lib.streamlit_log import StreamlitLogHandler
import logging
import logging.config
import scipy


def config():
    logging.config.fileConfig(fname='config.ini', disable_existing_loggers=False)

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    return logger

logger = config()
handler = StreamlitLogHandler()
logger.addHandler(handler)


def text_conditional_gen(model, music_parameters, filename, length, temperature=1.0, progress=True, top_k=250, top_p=0.0):
    logger.info("Setting parameters for musicgen model...")
    model.set_generation_params(
        duration=length, temperature=temperature, top_k=top_k, top_p=top_p)

    logger.info("Generating audio...")
    audio = model.generate(descriptions=music_parameters, progress=progress)

    sampling_rate = model.sample_rate
    Audio(audio[0].cpu().numpy(), rate=sampling_rate)

    logger.info("Saving file as .wav...")
    scipy.io.wavfile.write(
        filename, rate=sampling_rate, data=audio[0, 0].cpu().numpy())

    logger.info("Success")

    return audio