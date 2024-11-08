from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

 # core vs legacy
from llama_index.legacy.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import  note_engine
from llama_index.legacy.tools import QueryEngineTool, ToolMetadata
from llama_index.legacy.agent import ReActAgent
from llama_index.legacy.llms import OpenAI
from pdf import canada_engine


population_path = os.path.join("data","population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str)  # interface for LLM to read the data with RAG
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, metadata=ToolMetadata(
        name="population_data",
        description="this gives information at the world population and demographics"
        ),
    ),
    QueryEngineTool(query_engine=canada_engine, metadata=ToolMetadata(
        name="Canada_data",
        description="this gives information about Canada, the country"
        ),
    ),
]

# set an agent that will have access to these tools
llm = OpenAI(model="gpt-3.5-turbo-1106")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit)")) !="q:":
    result = agent.query(prompt)
    print(result)