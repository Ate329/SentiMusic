from lib.MusicGeneration.musicgen_gen import *
from lib.MusicGeneration.musicgen_load import load_model, accelerator
from lib.sentiment_analyser import sentiment_analyser
from lib.music_parameters_phi2 import generate

# size can be small, medium and large
model = load_model(size='small')

labeled_scores = sentiment_analyser()

music_parameters = generate(labeled_scores)

music_parameters_list = [music_parameters]

text_conditional_gen(model, music_parameters=[music_parameters_list], size='small')
    