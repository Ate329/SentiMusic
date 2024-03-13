from lib.sentiment_analyser import sentiment_analyser
from lib import MusicGenerationTrans
from lib import music_parameters_phi2
from lib import MusicGenerationAC


def use_transformers(filename, length, size="small"):
    # size can be small, medium and large
    model = MusicGenerationTrans.musicgen_load.load_model(size=size)

    labeled_scores = sentiment_analyser()

    music_parameters = music_parameters_phi2.generate(labeled_scores)

    music_parameters_list = [music_parameters]

    MusicGenerationTrans.musicgen_gen.text_conditional_gen(
        filename=filename, model=model, music_parameters=music_parameters_list, lengeth=length, size=size)


def use_audiocraft(filename, length, temperature, size="small", top_k=250, top_p=0.0):
    model = MusicGenerationAC.musicgen_load.load_model(size=size)

    labeled_scores = sentiment_analyser()

    music_parameters = music_parameters_phi2.generate(labeled_scores)

    music_parameters_list = [music_parameters]

    MusicGenerationAC.musicgen_gen.text_conditional_gen(
        filename=filename, model=model, music_parameters=music_parameters_list, top_k=top_k, top_p=top_p, temperature=temperature, length=length)
