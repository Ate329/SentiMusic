from lib.sentiment_analyser import sentiment_analyser
from lib import MusicGenerationTrans
from lib import music_parameters_phi2
from lib import MusicGenerationAC

def use_transformers():
    # size can be small, medium and large
    model = MusicGenerationTrans.musicgen_load.load_model(size='small')

    labeled_scores = sentiment_analyser()

    music_parameters = music_parameters_phi2.generate(labeled_scores)

    music_parameters_list = [music_parameters]

    MusicGenerationTrans.musicgen_gen.text_conditional_gen(model, music_parameters=music_parameters_list, lengeth=100, size='small')


def use_audiocraft():
    model = MusicGenerationAC.musicgen_load.load_model(size="small")

    labeled_scores = sentiment_analyser()

    music_parameters = music_parameters_phi2.generate(labeled_scores)

    music_parameters_list = [music_parameters]

    MusicGenerationAC.musicgen_gen.text_conditional_gen(model=model, music_parameters=music_parameters_list, temperature=1.0, length=60)
