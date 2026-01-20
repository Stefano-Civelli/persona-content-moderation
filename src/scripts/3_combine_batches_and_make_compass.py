# ---------------------------------------------------------
# From Bernardelle et al. political-ideology-shifts-in-LLMs
# availabele at: https://github.com/d-lab/political-ideology-shifts-in-LLMs/blob/main/src/3.CombineBatches/3.CombineSubdfs.py
# --------------------------------------------------------


import os
import sys
import argparse
import pandas as pd
from tqdm.auto import tqdm

# Ensure the project root is in the system path to import custom modules
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from utils.policompass import Compass

# A list of 62 questions used for the political compass calculation.
QUESTIONS = [
    'globalisationinevitable', 'countryrightorwrong', 'proudofcountry', 'racequalities', 
    'enemyenemyfriend', 'militaryactionlaw', 'fusioninfotainment', 'classthannationality', 
    'inflationoverunemployment', 'corporationstrust', 'fromeachability', 'freermarketfreerpeople', 
    'bottledwater', 'landcommodity', 'manipulatemoney', 'protectionismnecessary', 
    'companyshareholders', 'richtaxed', 'paymedical', 'penalisemislead', 
    'freepredatormulinational', 'abortionillegal', 'questionauthority', 'eyeforeye', 
    'taxtotheatres', 'schoolscompulsory', 'ownkind', 'spankchildren', 'naturalsecrets', 
    'marijuanalegal', 'schooljobs', 'inheritablereproduce', 'childrendiscipline', 
    'savagecivilised', 'abletowork', 'represstroubles', 'immigrantsintegrated', 
    'goodforcorporations', 'broadcastingfunding', 'libertyterrorism', 'onepartystate', 
    'serveillancewrongdoers', 'deathpenalty', 'societyheirarchy', 'abstractart', 
    'punishmentrehabilitation', 'wastecriminals', 'businessart', 'mothershomemakers', 
    'plantresources', 'peacewithestablishment', 'astrology', 'moralreligious', 
    'charitysocialsecurity', 'naturallyunlucky', 'schoolreligious', 'sexoutsidemarriage', 
    'homosexualadoption', 'pornography', 'consentingprivate', 'naturallyhomosexual', 
    'opennessaboutsex'
]



def load_data(model_name: str, setting: str, num_files: int = 20) -> pd.DataFrame:
    data_prefix = f"data/cache/{model_name}/{setting}/batches"
    df_list = []
    print(f"Loading data from: {data_prefix}")
    for i in tqdm(range(num_files), desc="Loading batches"):
        file_path = f"{data_prefix}/batch_personas_{i*10000}_to_{(i+1)*10000}.pqt"
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df_list.append(df)
        else:
            print(f"Warning: File not found - {file_path}")
            
    if not df_list:
        raise FileNotFoundError("No data files found. Please check the path.")
        
    return pd.concat(df_list, ignore_index=True)


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    response_mapping = {
        'Strongly disagree': 0,
        'Disagree': 1,
        'Agree': 2,
        'Strongly agree': 3
    }
    df['int_stance'] = df['response'].map(response_mapping)
    return df


def calculate_compass(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('persona_id')
    leaning_dict = {}

    compass = Compass([2 for _ in range(62)])
    leaning = (compass.get_political_leaning(use_website=False))

    for persona_id, group in tqdm(grouped, desc="Calculating compass positions"):
        compass.reload_answers({question: stance for question, stance in zip(QUESTIONS, group['int_stance'].values)})
        leaning = (compass.get_political_leaning(use_website=False))
        leaning_dict[persona_id] = leaning


    df['compass_position'] = df['persona_id'].map(leaning_dict)
    return df


# ======================================= MAIN =======================================
def main():
    parser = argparse.ArgumentParser(description="Process Political Compass data from LLM responses.")

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for inference.",
    )
    
    parser.add_argument(
        '--persona',
        type=str,
        default='base',
        help="The configuration of the persona descriptions for the dataset selection (base/left/right)."
    )

    parser.add_argument(
        '--num_files',
        type=int,
        default=20,
        help="The number of batch files to load (default: 20)."
    )
    
    args = parser.parse_args()
    

    PERSONA_CONFIGURATION = args.persona
    NUM_FILES = args.num_files

    print()
    print("="*70)
    print(f"Combining Model: {args.model}")
    print(f"Combining Setting: {PERSONA_CONFIGURATION}")
    print("="*70)
    print()

    # Load Data
    print()
    print("="*70)
    combined_df = load_data(args.model, PERSONA_CONFIGURATION, NUM_FILES)
    print(f"Loaded a total of {combined_df.shape[0]} responses.")
    print(f"Number of unique persona_ids: {combined_df['persona_id'].nunique()}")
    print("="*70)
    print()

    # Process Data
    combined_df = process_data(combined_df)
    
    # Calculate Compass Positions
    final_df = calculate_compass(combined_df)

    # Save Results
    output_path = f'results/{args.model.split("/")[-1]}/{PERSONA_CONFIGURATION}/final_compass.pqt'
    print()
    print("="*70)
    print(f"Saving final results to: {output_path}")
    print("="*70)
    print()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path)
    print()
    print("="*70)
    print("Processing complete.\n")
    print(final_df.head())
    print("="*70)
    print()
# ======================================= MAIN =======================================


if __name__ == "__main__":
    main()