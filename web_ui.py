import streamlit as st
from lib.sentiment_analyser import sentiment_analyser
from lib import MusicGenerationTrans
from lib import music_parameters_phi2
from lib import MusicGenerationAC


st.title("Music Generation App")


def use_transformers(filename, length, size="small"):
    # size can be small, medium and large
    model = MusicGenerationTrans.musicgen_load.load_model(size=size)

    labeled_scores = sentiment_analyser()

    music_parameters = music_parameters_phi2.generate(labeled_scores)

    music_parameters_list = [music_parameters]

    audio = MusicGenerationTrans.musicgen_gen.text_conditional_gen(
        filename=filename, model=model, music_parameters=music_parameters_list, lengeth=length, size=size)

    return audio


def use_audiocraft(filename, length, temperature, size="small", top_k=250, top_p=0.0):
    model = MusicGenerationAC.musicgen_load.load_model(size=size)

    labeled_scores = sentiment_analyser()

    music_parameters = music_parameters_phi2.generate(labeled_scores)

    music_parameters_list = [music_parameters]

    audio = MusicGenerationAC.musicgen_gen.text_conditional_gen(
        filename=filename, model=model, music_parameters=music_parameters_list, top_k=top_k, top_p=top_p, temperature=temperature, length=length)

    return audio


def generate_music(package, filename, length, temperature, size="small", top_k=250, top_p=0.0):
    if package == "transformers":
        use_transformers(filename, length, size="small")

    elif package == "audiocraft":
        use_audiocraft(filename, length, temperature,
                       size="small", top_k=250, top_p=0.0)
    else:
        st.error(
            "Invalid package selection. Please choose Transformers or Audiocraft.")
        return None

    for i in range(100):
        progress.progress(i + 1)
        output.write(f"Generating music... {i+1}% complete")

    return generated_music


package_select = st.selectbox(
    "Generation Package", ("Transformers", "Audiocraft"))
col1, col2 = st.columns(2)

# UI for parameters
with col1:
    filename_input = st.text_input("Filename (Optional)", key="filename")
    length_input = st.slider("Length (in seconds)",
                             min_value=1, max_value=60, key="length")
    temperature_input = st.slider("Temperature (Controls randomness)",
                                  min_value=0.1, max_value=1.0, step=0.1, key="temperature")
    size_select = st.selectbox(
        "Model Size", ("small", "medium", "large"), key="size")
    top_k_input = st.number_input(
        "Top K (Sampling parameter)", min_value=1, max_value=1000, key="top_k")
    top_p_input = st.number_input(
        "Top P (Nucleus sampling)", min_value=0.0, max_value=1.0, step=0.1, key="top_p")

with col2:
    # Create a progress bar
    progress = st.empty()

    # Create a terminal for output
    output = st.empty()

# Generate button
if st.button("Generate Music"):
    # Clear previous output
    output.empty()

    # Call the generate_music function with user-provided parameters
    generated_music = generate_music(
        filename_input, length_input, temperature_input, size=size_select, top_k=top_k_input, top_p=top_p_input)

    if generated_music is not None:
        st.audio(generated_music)
        st.success("Music generation complete!")
