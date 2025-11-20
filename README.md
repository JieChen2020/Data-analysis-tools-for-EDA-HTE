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

1. Load 'acidamine_result_w_probs.csv' or CSV file with the same format, you can get the ROC result (ROC_curve.png and roc_curve_data.csv).

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

1. Load 'SI 9.1 Reaction kinetic data 60.csv' or CSV file with the same format to obtain the results of the four kinetic models.

```python
# Load the csv file containing the data
file_path = 'SI 9.1 Reaction kinetic data 60.csv'
data = pd.read_csv(file_path)
```

## **Tool 4: Substrate_selection**

Matching molecules using regular expressions of SMILES and rdkit (mol.HasSubstructMatch).

#### Requirements

1. Python 3.12
2. rdkit 2024.9.6

#### How to use

1. Download Mcule In Stock ('mcule_purchasable_in_stock_250309.smi') from https://mcule.com/database/.
2. Load 'mcule_purchasable_in_stock_250309.smi' or SMI file with the same format.

```python
smi_file_path = 'mcule_purchasable_in_stock_250309.smi'
acid = extract_acid_structures(smi_file_path)
```

3. Change the pattern according to the target molecular structure.

```python
pattern1 = Chem.MolFromSmarts('[NH2]')
pattern = Chem.MolFromSmarts('C(=O)[OH]')
```

```python
N_pattern = r'N'
pattern1 = Chem.MolFromSmarts('[NH2]')
pattern2 = Chem.MolFromSmarts('C1CCCC[NH]1')
pattern3 = Chem.MolFromSmarts('C1C=CC=C([NH]C)C=1')
pattern4 = Chem.MolFromSmarts('C1(CCCCC1)[NH]C')
pattern5 = Chem.MolFromSmarts('[NH]1CCCC1')
pattern6 = Chem.MolFromSmarts('C1CN(C)CC[NH]1')
pattern7 = Chem.MolFromSmarts('C1C=CC=C(C[NH]C)C=1')
pattern = Chem.MolFromSmarts('C(=O)[OH]')
```

## **Tool 5: TSNE_curve**

For more information, see: https://chemplot.readthedocs.io/en/latest/index.html

#### Requirements

1. Python 3.12
2. chemplot 1.3.1

#### How to use

1. Load 'filtered_molecules.csv' or CSV file with the same format.

```python
data_BBBP = read_csv('filtered_molecules.csv')
```
