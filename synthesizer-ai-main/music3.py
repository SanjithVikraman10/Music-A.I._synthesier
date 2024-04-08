import click
from datetime import datetime
from typing import List, Dict
from midiutil import MIDIFile
from pyo import *
from music21 import stream, meter, key as kkey, note
from genetic2 import genome_generation, Genome, pair_selection, single_point_crossover, mutationfunc
import csv
import joblib
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score


from music21 import environment
environment.UserSettings()['lilypondPath'] = r'lilypond-2.24.2\bin\lilypond.exe'


bitspernote = 4
keys = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
scales = ["major", "minorM", "majorBlues", "minorBlues", "dorian"]
newpop = []

def binary_to_float32(binary_sequence):
    # Convert binary to decimal
    try:
        decimal_value = int(binary_sequence, 2)
    except ValueError:
        new = ''
        for i in binary_sequence:
            if i.isdigit():
                new += str(i)
            else:
                pass
        decimal_value = int(new, 2)
    
    # Normalize decimal value (adjust as needed)
    normalized_value = decimal_value / 1e9  # Example normalization, adjust as needed
    
    # Convert to float32
    float32_value = np.float32(normalized_value)
    
    return float32_value

df = pd.read_csv('genome_dataset.csv')

df['float_genome'] = df['Genome'].apply(binary_to_float32)
df['Pauses'] = df['Pauses'].astype(int)
df['log_genome'] = np.log1p(df['float_genome'])

categorical_cols = ['log_genome', 'Pauses', 'Key', 'Scale']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Rating']]

X = df.drop(['Rating','float_genome','Genome'], axis=1)
y = df['Rating']
#print(X,y)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df.drop(['Rating','float_genome','Genome'], axis=1)
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)
from sklearn.metrics import explained_variance_score

# Create and fit the model
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)
print(regressor)


# Predict on the test set
y_pred = regressor.predict(X_test)
print(y_pred)
# Evaluate the model (you can use different metrics based on your problem)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {abs(r2)}')
print(f'Explained Variance Score: {explained_var}')




def int_from_bits(bits: List[int]) -> int:
    """Converts list of bits into integer"""
    return int(sum([bit*pow(2, index) for index, bit in enumerate(bits)]))


def genome_to_melody(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: int, key: str, scale: str, root: int) -> Dict[str, list]:

    """Converts a genome(list of bits) into music"""

    notes = [genome[i * bitspernote:i * bitspernote + bitspernote] for i in range(num_bars * num_notes)]

    note_length = 4/float(num_notes)

    scl = EventScale(root=key, scale=scale, first=root)

    melody = {
        "notes": [],
        "velocity": [],
        "beat": []
    }

    for note in notes:
        integer = int_from_bits(note)

        if pauses:
                melody["notes"] += [0]
                melody["velocity"] += [0]
                melody["beat"] += [note_length]
        if not pauses:
            integer = int(integer % pow(2, bitspernote - 1))

        if integer >= pow(2, bitspernote - 1):
            melody["notes"] += [0]
            melody["velocity"] += [0]
            melody["beat"] += [note_length]
        else:
            if len(melody["notes"]) > 0 and melody["notes"][-1] == integer:
                melody["beat"][-1] += note_length
            else:
                melody["notes"] += [integer]
                melody["velocity"] += [127]
                melody["beat"] += [note_length]

    steps = []
    for step in range(num_steps):
        steps.append([scl[(note+step*2) % len(scl)] for note in melody["notes"]])

    melody["notes"] = steps
    return melody


def genome_to_events(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: bool, key: str, scale: str, root: int, bpm: int) -> [Events]:

    """creates Pyo Events objects that can be played in the Pyo Server"""

    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    return [
        Events(
            midinote=EventSeq(step, occurrences=1),
            midivel=EventSeq(melody["velocity"], occurrences=1),
            beat=EventSeq(melody["beat"], occurrences=1),
            attack=0.001,
            decay=0.05,
            sustain=0.5,
            release=0.005,
            bpm=bpm
        ) for step in melody["notes"]
    ]

def objective(weights, *args):
    user_rating, rounded_rating = args
    user_rating = float(user_rating)
    rounded_rating = float(rounded_rating)
    combined_rating = weights[0] * user_rating + weights[1] * rounded_rating
    return abs(user_rating - combined_rating)

def gradient_descent(objective, initial_weights, args, learning_rate=0.01, num_iterations=100):
    weights = initial_weights.copy()

    for _ in range(num_iterations):
        grad = np.zeros_like(weights)

        for i in range(len(weights)):
            weights[i] += 1e-5  
            loss_plus = objective(weights, *args)
            weights[i] -= 2e-5  
            loss_minus = objective(weights, *args)
            weights[i] += 1e-5 

            grad[i] = (loss_plus - loss_minus) / (2e-5)

        weights -= learning_rate * grad

    return weights
    
def fitness(genome: Genome, s: Server, num_bars: int, num_notes: int, num_steps: int,
            pauses: bool, key: str, scale: str, root: int, bpm: int, initial_weights: np.ndarray, 
            user_rating = None) -> int:
    m = metronome(bpm)

    events = genome_to_events(genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    s.start()
    
    genstr=''
    for i in genome:
        genstr += str(i)
    print(genstr)
    
    key_mapping = {key: i for i, key in enumerate(keys)}
    scale_mapping = {scale: i for i, scale in enumerate(scales)}

    # Get the model's predicted rating
    new_df = pd.DataFrame({
    'Genome': [genstr],
    'Bars': [num_bars],
    'Notes': [num_notes],
    'Steps': [num_steps],
    'Pauses': [pauses],
    'Key': [key_mapping[key]],
    'Scale': [scale_mapping[scale]],
    'ScaleRoot': [root],
    'bpm': [bpm]
    })

    print(new_df['Genome'])
    # Convert categorical variables using label encoder
    new_df['float_genome'] = new_df['Genome'].apply(binary_to_float32)
    new_df['Pauses'] = new_df['Pauses'].astype(int)
    new_df['log_genome'] = np.log1p(new_df['float_genome'])
    new_df = new_df.drop(['Genome','float_genome'],axis=1)
    print(new_df)

    predicted_rating = regressor.predict(new_df)
    rounded_rating = np.round(predicted_rating)
    rounded_rating = rounded_rating.astype(int) 
   

    if user_rating is None:
        user_rating = input("How good did that sound on a scale of [0-5]? ")
    print(f'Predicted Rating for the new genome: {predicted_rating}')
    

    # Perform gradient descent to optimize weights
    optimized_weights = gradient_descent(objective, initial_weights, (user_rating, rounded_rating))

    print("Optimized Weights:", optimized_weights)

    user_rating_float = float(user_rating)

    # Calculate combined rating using the optimized weights
    combined_rating = optimized_weights[0] * user_rating_float + optimized_weights[1] * predicted_rating
    print("Combined Rating with Optimized Weights:", combined_rating)
    
    newpop.append([genstr,num_bars, num_notes, num_steps, pauses, key_mapping[key], scale_mapping[scale], root, bpm, user_rating])
    for e in events:
        e.stop()
    s.stop()
    time.sleep(1)

    try:
        user_rating = int(user_rating)
    except ValueError:
        user_rating = 0

    return combined_rating


def metronome(bpm: int):
    met = Metro(time=1 / (bpm / 60.0)).play()
    t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
    amp = TrigEnv(met, table=t, dur=.25, mul=1)
    freq = Iter(met, choice=[660, 440, 440, 440])
    return Sine(freq=freq, mul=amp).mix(2).out()


def save_genome_to_midi(filename: str, genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                        pauses: bool, key: str, scale: str, root: int, bpm: int):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    if len(melody["notes"][0]) != len(melody["beat"]) or len(melody["notes"][0]) != len(melody["velocity"]):
        raise ValueError

    mf = MIDIFile(1)

    track = 0
    channel = 0

    time = 0.0
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, bpm)

    for i, vel in enumerate(melody["velocity"]):
        if vel > 0:
            for step in melody["notes"]:
                mf.addNote(track, channel, step[i], time, melody["beat"][i], vel)

        time += melody["beat"][i]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        mf.writeFile(f)


def save_genome_to_pdf(filename: str, genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                       pauses: bool, keyy: str, scale: str, root: int, bpm: int):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, keyy, scale, root)

    # Convert melody to music21 stream
    music21_stream = stream.Score()
    part = stream.Part()

    for i, step in enumerate(melody["notes"][0]):
        if step != 0:
            n = note.Note()
            n.pitch.midi = step
            n.quarterLength = melody["beat"][i]
            part.append(n)

    music21_stream.insert(0, part)

    # Set key and time signature
    ks = kkey.Key(keyy)
    music21_stream.append(ks)
    music21_stream.append(meter.TimeSignature('4/4'))

    # Export to PDF
    pdf_path = f"{os.path.splitext(filename)[0]}.pdf"
    music21_stream.write('lily.pdf', fp=pdf_path)

@click.command()
@click.option("--num-bars", default=8, prompt='Number of bars:', type=int)
@click.option("--num-notes", default=4, prompt='Notes per bar:', type=int)
@click.option("--num-steps", default=1, prompt='Number of steps:', type=int)
@click.option("--pauses", default=True, prompt='Pauses?', type=bool)
@click.option("--key", default="C", prompt='Key:', type=click.Choice(keys, case_sensitive=False))
@click.option("--scale", default="major", prompt='Scale:', type=click.Choice(scales, case_sensitive=False))
@click.option("--root", default=4, prompt='Scale Root:', type=int)
# @click.option("--population-size", default=10, prompt='Population size:', type=int) 


def main(num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int):

    population_size=5
    num_mutationfuncs=2
    mutationfunc_probability=0.5
    bpm=128
    folder = 'MusicSynthAI'+ str(int(datetime.now().timestamp()))

    initial_weights = [0.7, 0.3]

    population = [genome_generation(num_bars * num_notes * bitspernote) for _ in range(population_size)]
    print(population)
    s = Server().boot()

    population_id = 0

    running = True
    while running:
        random.shuffle(population)

        population_fitness = [(genome, fitness(genome, s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm, initial_weights)) for genome in population]
        print(population_fitness)
        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)

        population = [e[0] for e in sorted_population_fitness]

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):

            def fitness_lookup(genome):
                for e in population_fitness:
                    if e[0] == genome:
                        return e[1]
                return 0

            parents = pair_selection(population, fitness_lookup)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutationfunc(offspring_a, num=num_mutationfuncs, prob=mutationfunc_probability)
            offspring_b = mutationfunc(offspring_b, num=num_mutationfuncs, prob=mutationfunc_probability)
            next_generation += [offspring_a, offspring_b]

        print(f"Population {population_id} done")

        events = genome_to_events(population[0], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        for e in events:
            e.play()
        s.start()
        input("The best rated, playing now. Press Enter when done.")
        s.stop()
        for e in events:
            e.stop()

        time.sleep(1)

        events = genome_to_events(population[1], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        for e in events:
            e.play()
        s.start()
        input("The next best liked, playing now. Press Enter when done.")
        s.stop()
        for e in events:
            e.stop()

        time.sleep(1)

        print(newpop)
        with open('genome_dataset.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerows(newpop)
        print("*-"*20)
        print("Saving MIDI files...")
        print("*-"*20)
        for i, genome in enumerate(population):
            save_genome_to_midi(f"{folder}/{population_id}/{scale}-{key}-{i}.mid", genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
            save_genome_to_pdf(f"{folder}/{population_id}/{scale}-{key}-{i}.mid", genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        print("Files saved. Go check out new folder.")

        running = input("Want to try making music again? [Y/n]") != "n"
        population = next_generation
        population_id += 1


if __name__ == '__main__':
    main()





""" 
@click.option("--num-mutationfuncs", default=2, prompt='Number of mutationfuncs:', type=int)
@click.option("--mutationfunc-probability", default=0.5, prompt='mutationfuncs probability:', type=float)
@click.option("--bpm", default=128, type=int) 
 """