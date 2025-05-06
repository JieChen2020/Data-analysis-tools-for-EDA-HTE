# **Data analysis tools for EDA HTE**

Some data analysis tools for high-throughput experimentation using the encapsulated droplet array.

## **Tool 1: LangChain**

#### Requirements

1. Python 3.12
2. rdkit 2024.9.6
3. scikit-learn 1.6.1
4. langchain 0.3.24
5. langchain-community 0.3.23
6. langchain-core 0.3.56
7. langchain-openai 0.3.14
8. langsmith 0.3.38

#### How to use

1. Fill in the LangSmith project information to monitor the LangChain execution process.

```python
os.environ["LANGCHAIN_TRACING_V2"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""
```

2. Fill in the LLM information.

```python
def run_langchain(prompt, tools):
    llm = ChatOpenAI(
        model="",
        api_key="",
        base_url=""
    )
```

3. Fill in the prompt (task input) to start the LangChain process.

```python
if __name__ == "__main__":
    global RADIUS, SIZE

    RADIUS = 3
    SIZE = 512
    #
    y_label_index = 4
    #
    prompt = ()
    tools = ['ChemicalTools']

    run_langchain(prompt, tools)
```

## **Tool 2: ROC_curve**

#### Requirements

1. Python 3.12
2. scikit-learn 1.6.1

#### How to use

1. Load 'acidamine_result_w_probs.csv' or CSV file with the same format.

```python
# Reading a CSV file
file_path = 'acidamine_result_w_probs.csv'
df = pd.read_csv(file_path)

# Use a 0.5 threshold to convert probabilities to classification results
df['predicted'] = (df['prob'] >= 0.5).astype(int)

# Calculate ROC curve data
fpr, tpr, thresholds = roc_curve(df['gt'], df['prob'])
```

## **Tool 3: Reaction kinetic tool**

#### Requirements

1. Python 3.12
2. scipy 1.15.2

#### How to use

1. Load 'Reaction kinetic data 60.csv' or CSV file with the same format.

```python
# Load the csv file containing the data
file_path = 'Reaction kinetic data 60.csv'
data = pd.read_csv(file_path)
```

## **Tool 4: Substrate_selection**

#### Requirements

1. Python 3.12
2. rdkit 2024.9.6

#### How to use

1. Download Mcule In Stock (mcule_purchasable_in_stock_250309.smi) from https://mcule.com/database/.
2. Load 'mcule_purchasable_in_stock_250309.smi' or SMI file with the same format.

```python
smi_file_path = 'mcule_purchasable_in_stock_250309.smi'
acid = extract_acid_structures(smi_file_path)
```

3. 
