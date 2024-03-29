import streamlit as st
import logging
from lib.sentiment_analyser import sentiment_analyser
from lib.MusicGenerationTrans.musicgen_load import load_model as trans_load
from lib.MusicGenerationTrans.musicgen_gen import text_conditional_gen as trans_gen
from lib import music_parameters_phi2
from lib.MusicGenerationAC.musicgen_gen import text_conditional_gen as AC_gen
from lib.MusicGenerationAC.musicgen_load import load_model as AC_load
from lib.streamlit_log import StreamlitLogHandler


def config():
    logging.config.fileConfig(
        fname='config.ini', disable_existing_loggers=False)

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    return logger


logger = config()
handler = StreamlitLogHandler()
logger.addHandler(handler)


st.title("SentiMusic")


def use_transformers(text, filename, length, size="small"):
    # size can be small, medium and large
    model = trans_load(size=size)

    labeled_scores = sentiment_analyser(text)

    music_parameters = music_parameters_phi2.generate(labeled_scores)

    music_parameters_list = [music_parameters]

    audio = trans_gen(filename=filename, model=model,
                      music_parameters=music_parameters_list, lengeth=length, size=size)

    return audio


def use_audiocraft(text, filename, length, temperature, size="small", top_k=250, top_p=0.0):
    model = AC_load(size=size)

    labeled_scores = sentiment_analyser(text)

    music_parameters = music_parameters_phi2.generate(labeled_scores)

    music_parameters_list = [music_parameters]

    audio = AC_gen(filename=filename, model=model, music_parameters=music_parameters_list,
                   top_k=top_k, top_p=top_p, temperature=temperature, length=length)

    return audio


def generate_music(package, text, filename, length, temperature, size="small", top_k=250, top_p=0.0):
    if package == "Transformers":
        use_transformers(text, filename, length, size=size)

    elif package == "Audiocraft":
        use_audiocraft(text, filename, length, temperature,
                       size=size, top_k=top_k, top_p=top_p)
    else:
        st.error(
            "Invalid package selection. Please choose Transformers or Audiocraft.")
        return None

    return generated_music


package_select = st.selectbox(
    "Generation Package", ("Audiocraft", "Transformers"))
col1, col2 = st.columns(2)

# UI for parameters
with col1:
    filename_input = st.text_input("Filename", key="filename")
    text_input = st.text_input(
        "Enter a sentence, a paragraph or a article", key="text")
    length_input = st.slider("Length (in seconds)",
                             min_value=1, max_value=5000, key="length")
    temperature_input = st.slider("Temperature (Controls randomness)",
                                  min_value=0.1, max_value=1.0, step=0.1, key="temperature")
    size_select = st.selectbox(
        "Model Size", ("small", "medium", "large"), key="size")
    top_k_input = st.number_input(
        "Top K (Sampling parameter, defualt=250)", min_value=1, max_value=1000, key="top_k")
    top_p_input = st.number_input(
        "Top P (Nucleus sampling, defualt=0.0)", min_value=0.0, max_value=1.0, step=0.1, key="top_p")

# Display the logs
st.text_area("Logs", value=st.session_state.get('logs', []), height=200)


# Generate button
if st.button("Generate Music"):
    # Call the generate_music function with user-provided parameters
    generated_music = generate_music(package=package_select, filename=filename_input, length=length_input,
                                     temperature=temperature_input, text=text_input, size=size_select, top_k=top_k_input, top_p=top_p_input)

    if generated_music is not None:
        st.audio(generated_music)
        st.success("Music generation complete!")
