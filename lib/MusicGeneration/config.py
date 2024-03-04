def change_config(model, inputs):
    from lib.MusicGeneration.musicgen_load import load_model, accelerator

    guidiance_scale = float(input("Enter the guidiance scale: "))
    model.generation_config.guidance_scale = guidiance_scale

    max_tokens = int(input("Enter the max tokens (default 256): "))
    model.generation_config.max_new_tokens = max_tokens

    temperature = float(input("Enter new model temperature (randomness): "))
    model.generation_config.temperature = temperature

    generate = str(
        input("Do you want to generate music according to the new config again? [y/N] "))
    if generate.lower() == 'y':
        from IPython.display import Audio

        device = accelerator()
        audio_values = model.generate(**inputs.to(device))

        sampling_rate = model.config.audio_encoder.sampling_rate
        Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)
    elif generate.lower() == 'n':
        pass
