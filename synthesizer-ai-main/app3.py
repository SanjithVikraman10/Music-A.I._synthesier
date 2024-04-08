
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime
from genetic2 import genome_generation, pair_selection, single_point_crossover, mutationfunc
import music3
from music3 import Server

from typing import List, Dict
from midiutil import MIDIFile
from pyo import *
from music21 import stream, meter, key as kkey, note
from genetic2 import genome_generation, Genome, pair_selection, single_point_crossover, mutationfunc
import csv
import joblib
import pandas as pd
import numpy as np
from random import choices
import time


from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

s = Server().boot()
app = Flask(__name__, template_folder="templates", static_url_path="/static")
newpop = []

from music21 import environment
environment.UserSettings()['lilypondPath'] = r'lilypond-2.24.2\bin\lilypond.exe'


bitspernote = 4
keys = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
scales = ["major", "minorM", "majorBlues", "minorBlues", "dorian"]

df = pd.read_csv('genome_dataset.csv')

df['float_genome'] = df['Genome'].apply(music3.binary_to_float32)
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

@app.route('/')
def index():
    return render_template('index.html')

def run_genetic_algorithm(num_bars, num_notes, num_steps, pauses, key, scale, root):
    global newpop
    population_size=5
    num_mutationfuncs=2
    mutationfunc_probability=0.5
    bpm=128
    folder = 'MusicSynthAI'+ str(int(datetime.now().timestamp()))

    population = [genome_generation(num_bars * num_notes * 4) for _ in range(population_size)]
    print(population)
    

    population_id = 0

    running = True

    while running:
        
        random.shuffle(population)

        population_fitness = [(genome, fitness(genome, s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)) for genome in population]

        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)

        population = [e[0] for e in sorted_population_fitness]

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):

            def fitness_lookup(genome):
                for e in population_fitness:
                    if e[0] == genome:
                        return e[1]
                return 0

            
            def pair_selection(population, fitness_lookup):
                weights = [max(1, int(fitness_lookup(gene))) for gene in population]
                # Ensure the total weight is greater than zero
                while sum(weights) == 0:
                    weights = [max(1, int(fitness_lookup(gene))) for gene in population]

                selected_pair = choices(population=population, weights=weights, k=2)
                return selected_pair


            parents = pair_selection(population, fitness_lookup)
            offspring_a, offspring_b = music3.single_point_crossover(parents[0], parents[1])
            offspring_a = music3.mutationfunc(offspring_a, num=num_mutationfuncs, prob=mutationfunc_probability)
            offspring_b = music3.mutationfunc(offspring_b, num=num_mutationfuncs, prob=mutationfunc_probability)
            next_generation += [offspring_a, offspring_b]

        print(f"Population {population_id} done")

        events = music3.genome_to_events(population[0], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        for e in events:
            e.play()
        s.start()
        input("The best rated, playing now. Press Enter when done.")
        s.stop()
        for e in events:
            e.stop()

        time.sleep(1)

        events = music3.genome_to_events(population[1], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
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
            for i, genome in enumerate(population):
               music3.save_genome_to_midi(f"{folder}/{population_id}/{scale}-{key}-{str(i)}.mid", genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
               music3.save_genome_to_pdf(f"{folder}/{population_id}/{scale}-{key}-{str(i)}.pdf", genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        print("Files saved. Go check out new folder.")


# Update the generate_music route in app/main.py
# ... (existing imports)

@app.route('/generate_music', methods=['POST'])
def generate_music():
    if request.method == 'POST':
        num_bars = int(request.form['num-bars'])
        num_notes = int(request.form['num-notes'])
        num_steps = int(request.form['num-steps'])
        pauses = bool(request.form.get('pauses', False))
        key = request.form['key']
        scale = request.form['scale']

        # Validate and handle the 'root' field
        root_str = request.form['root']
        root = int(root_str) if root_str.isdigit() else 0  # Default to 0 if not a valid integer
        s.start()
        run_genetic_algorithm(num_bars, num_notes, num_steps, pauses, key, scale, root)
        s.stop()

    return redirect(url_for('index'))

selected_rating = 0
@app.route('/set_rating', methods=['POST'])
def set_rating():
    global selected_rating
    selected_rating = request.form.get('rating')
    return jsonify(success=True)


def metronome(bpm: int):
    met = Metro(time=1 / (bpm / 60.0)).play()
    t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
    amp = TrigEnv(met, table=t, dur=.25, mul=1)
    freq = Iter(met, choice=[660, 440, 440, 440])
    return Sine(freq=freq, mul=amp).mix(2).out()
    
def fitness(genome: Genome, s: Server, num_bars: int, num_notes: int, num_steps: int,
            pauses: bool, key: str, scale: str, root: int, bpm: int) -> int:

    global selected_rating,newpop
    m = metronome(bpm)

    events = music3.genome_to_events(genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    s.start()
    time.sleep(10)

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
    
    new_df['float_genome'] = new_df['Genome'].apply(music3.binary_to_float32)
    new_df['Pauses'] = new_df['Pauses'].astype(int)
    new_df['log_genome'] = np.log1p(new_df['float_genome'])
    new_df = new_df.drop(['Genome','float_genome'],axis=1)
    print(new_df)
    
    predicted_rating = regressor.predict(new_df)
    rounded_rating = np.round(predicted_rating)
    rounded_rating = rounded_rating.astype(int) 
   
    if selected_rating:
        user_rating = selected_rating
    else:
        user_rating = 0
    print("User Rating:", selected_rating)
    print(f'Predicted Rating for the new genome: {predicted_rating}')
    
    optimized_weights = music3.gradient_descent(music3.objective, [0.7,0.3], (user_rating, predicted_rating))

    print("Optimized Weights:", optimized_weights)

    user_rating_float = float(user_rating)

    combined_rating = optimized_weights[0] * user_rating_float + optimized_weights[1] * predicted_rating
    print("Combined Rating with Optimized Weights/Fitness:", combined_rating)
    
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


if __name__ == '__main__':
    app.run(debug=True)