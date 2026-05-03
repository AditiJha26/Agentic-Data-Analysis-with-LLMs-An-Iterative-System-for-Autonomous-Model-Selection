from tools.data_tool import load_dataset
from agent.graph import run_agent

# Load dataset
df = load_dataset("../data/housing.csv")

# Run agent
output = run_agent(df)

print("Final Output:")
print(output)