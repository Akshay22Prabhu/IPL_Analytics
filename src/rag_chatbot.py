from transformers import pipeline
import pandas as pd

llm = pipeline("text-generation", model="distilgpt2")

def generate_player_summary(player_name):
    df = pd.read_csv("data/processed/final_dataset.csv")

    player_df = df[df['batter'] == player_name]

    avg_runs = player_df['runs'].mean()
    avg_sr = player_df['strike_rate'].mean()
    last_5 = player_df.tail(5)['runs'].tolist()

    prompt = f"""
    Player: {player_name}
    Avg Runs: {avg_runs}
    Strike Rate: {avg_sr}
    Last 5: {last_5}

    Analyze performance, strengths and weaknesses.
    """

    response = llm(prompt, max_length=120)

    return response[0]['generated_text']