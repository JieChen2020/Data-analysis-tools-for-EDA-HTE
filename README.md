# **Data analysis tools for EDA HTE**

Some data analysis tools for high-throughput experimentation using the encapsulated droplet array.

## **Tool 1: LangChain**

#### Requirements

1. Python 3.12
2. rdkit 2024.9.6
3. scikt-learn 1.6.1
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

