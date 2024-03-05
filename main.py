from lib.MusicGeneration.musicgen_gen import *
from lib.MusicGeneration.musicgen_load import load_model
from lib.sentiment_analyser import sentiment_analyser
from lib.music_parameters_phi2 import generate

model = load_model(size='small')

labeled_scores = sentiment_analyser()

music_parameters = generate(labeled_scores)

text_conditional_gen(model, music_parameters, size='small')
