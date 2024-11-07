from dotenv import load_dotenv
import os
import pandas as pd
 # core vs legacy
from llama_index.legacy.query_engine import PandasQueryEngine

from prompts import new_prompt, instruction_str

load_dotenv()

population_path = os.path.join("data","population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str)  # interface for LLM to read the data with RAG
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
population_query_engine.query("what is the population of canada")