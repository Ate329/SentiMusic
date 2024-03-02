from magenta.models import music_transformer as MusicTransformer
from magenta.music import ChordProgression as chords  # Import for chord representation
from magenta.music import QuantizedStep  # Import for musical notes
from magenta.models.music_vae import MusicVAE
import music21 as m21

def generate_music():
    # this is just an example for testing purpose
    '''
    Length: Specify the desired length of the generated music piece (in number of steps).
    Tempo: Set the desired tempo (beats per minute).
    Temperature: This value controls the randomness of the generation. Higher values lead to more surprising and potentially less coherent music.
    Start_text (Optional): Provide a starting text prompt to influence the style or content (e.g., "happy melody").
    Chords (Optional): If you want specific chords (harmony), define a chords.Progression object to specify the chord sequence.
    '''
    length = 512
    tempo = 120
    temperature = 1.0
    start_text = ""  # Optional starting text prompt (empty for now)
    chords_progression = None  # Optional chord progression (empty for now)

    # Create a MusicTransformer instance
    model = MusicTransformer()

    # Generate music
    generated_music = model.generate(length, temperature=temperature, start_text=start_text, chords_progression=chords_progression, qpm=tempo)

    return generate_music

# Function to convert QuantizedStep sequence to music21 stream
def quantized_steps_to_stream(qsteps):
    stream = m21.stream.Stream()
    for qstep in qsteps:
        # Convert each QuantizedStep to a music21 note
        note = m21.note.Note(qstep.pitch, quarterLength=qstep.duration)
        stream.append(note)
    return stream

'''
generate_music = generate_music()
# Convert the generated music
music_stream = quantized_steps_to_stream(generated_music)
# Display the music notation in a graphical window
music_stream.show()
# Alternatively, save the music notation as a MIDI file
music_stream.write('midi', fp='generated_music.midi')
'''

music_generated = generate_music()