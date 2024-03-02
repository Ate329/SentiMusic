from lib.MusicGeneration.musicgen_load import load_model, accelerator
from transformers import AutoProcessor
from IPython.display import Audio
from datasets import load_dataset
import scipy

'''
model = load_model(size='medium')
accelerator()
'''

def unconditional_gen(model):
    unconditional_inputs = model.get_unconditional_inputs(num_samples=1)

    audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)

    scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

    audio_length_in_s = 256 / model.config.audio_encoder.frame_rate

    print(f"Audio length: {audio_length_in_s}")


def text_conditional_gen(model, size='small'):
    device = accelerator()

    processor = AutoProcessor.from_pretrained(f"facebook/musicgen-{size}")

    inputs = processor(
        text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)


def audio_prompted_gen(model, size='small'):
    device = accelerator()

    dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
    sample = next(iter(dataset))["audio"]

    processor = AutoProcessor.from_pretrained(f"facebook/musicgen-{size}")

    # take the first half of the audio sample
    sample["array"] = sample["array"][: len(sample["array"]) // 2]

    inputs = processor(
        audio=sample["array"],
        sampling_rate=sample["sampling_rate"],
        text=["80s blues track with groovy saxophone"],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)


def batched_audio_prompted_gen(model, size='small'):
    device = accelerator()

    dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
    sample = next(iter(dataset))["audio"]

    # take the first quater of the audio sample
    sample_1 = sample["array"][: len(sample["array"]) // 4]

    # take the first half of the audio sample
    sample_2 = sample["array"][: len(sample["array"]) // 2]

    processor = AutoProcessor.from_pretrained(f"facebook/musicgen-{size}")

    inputs = processor(
        audio=[sample_1, sample_2],
        sampling_rate=sample["sampling_rate"],
        text=["80s blues track with groovy saxophone", "90s rock song with loud guitars and heavy drums"],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=256)

    # post-process to remove padding from the batched audio
    audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
    
    sampling_rate = model.config.audio_encoder.sampling_rate
    Audio(audio_values[0], rate=sampling_rate)