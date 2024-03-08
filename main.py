from lib.MusicGenerationTrans.musicgen_gen import *
from lib.MusicGenerationTrans.musicgen_load import load_model, accelerator
from lib.sentiment_analyser import sentiment_analyser
from lib.music_parameters_phi2 import generate
from lib import MusicGenerationTrans

# size can be small, medium and large
model = MusicGenerationTrans.musicgen_load(size='small')

labeled_scores = sentiment_analyser()

music_parameters = generate(labeled_scores)

music_parameters_list = [music_parameters]

text_conditional_gen(model, music_parameters=music_parameters_list, lengeth=100, size='small')
