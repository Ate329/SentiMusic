import scipy
from datasets import load_dataset
from IPython.display import Audio
from transformers import AutoProcessor
'''
# when you run the modlue individually
import sys
sys.path.append("lib/MusicGeneration")
'''
import logging
import logging.config


def config():
    logging.config.fileConfig(
        fname='config.ini', disable_existing_loggers=False)

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    return logger


# model = load_model(size='small')
logger = config()


def unconditional_gen(model):
    logger.info("Generating music unconditionally...")

    unconditional_inputs = model.get_unconditional_inputs(num_samples=1)

    audio_values = model.generate(
        **unconditional_inputs, do_sample=True, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)

    scipy.io.wavfile.write(
        "musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

    audio_length_in_s = 256 / model.config.audio_encoder.frame_rate

    print(f"Audio length: {audio_length_in_s}")

    logger.info("Success! Music saved as .wav")


def text_conditional_gen(model, music_parameters, lengeth, size='small'):
    device = accelerator()

    max_new_tokens = lengeth * model.config.audio_encoder.frame_rate

    logger.info("Generating music by text...")

    processor = AutoProcessor.from_pretrained(f"facebook/musicgen-{size}")

    inputs = processor(
        text=music_parameters,
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(
        **inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens)

    sampling_rate = model.config.audio_encoder.sampling_rate
    Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)

    logger.info("Success! Music saved as .wav")


def audio_prompted_gen(model, music_parameters, size='small'):
    from musicgen_load import accelerator
    device = accelerator()

    logger.info("Generating music by audio...")

    dataset = load_dataset("sanchit-gandhi/gtzan",
                           split="train", streaming=True)
    sample = next(iter(dataset))["audio"]

    processor = AutoProcessor.from_pretrained(f"facebook/musicgen-{size}")

    # take the first half of the audio sample
    sample["array"] = sample["array"][: len(sample["array"]) // 2]

    inputs = processor(
        audio=sample["array"],
        sampling_rate=sample["sampling_rate"],
        text=music_parameters,
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(
        **inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)

    logger.info("Success! Music saved as .wav")


def batched_audio_prompted_gen(model, music_parameters, size='small'):
    from musicgen_load import accelerator
    device = accelerator()

    logger.info("Generating music by batched audio...")

    dataset = load_dataset("sanchit-gandhi/gtzan",
                           split="train", streaming=True)
    sample = next(iter(dataset))["audio"]

    # take the first quater of the audio sample
    sample_1 = sample["array"][: len(sample["array"]) // 4]

    # take the first half of the audio sample
    sample_2 = sample["array"][: len(sample["array"]) // 2]

    processor = AutoProcessor.from_pretrained(f"facebook/musicgen-{size}")

    inputs = processor(
        audio=[sample_1, sample_2],
        sampling_rate=sample["sampling_rate"],
        text=music_parameters,
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(
        **inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=256)

    # post-process to remove padding from the batched audio
    audio_values = processor.batch_decode(
        audio_values, padding_mask=inputs.padding_mask)

    sampling_rate = model.config.audio_encoder.sampling_rate
    Audio(audio_values[0], rate=sampling_rate)

    logger.info("Success! Music saved as .wav")
