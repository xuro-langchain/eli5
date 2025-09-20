# Explain Like I'm 5 (ELI5)


## Introduction
In this notebook, we'll set up a simple application to trace with LangSmith. 

## Context

At LangChain, we aim to make it easy to build LLM applications. One type of LLM application you can build is an agent. There’s a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible. 

In practice though, it is incredibly difficult to build systems that reliably execute on these tasks. As we’ve worked with our users to put agents into production, we’ve learned that more control is often necessary. You might need an agent to always call a specific tool first or use different prompts based on its state.

To tackle this problem, we’ve built [LangGraph](https://langchain-ai.github.io/langgraph/) — a framework for building agent and multi-agent applications. Separate from the LangChain package, LangGraph’s core design philosophy is to help developers add better precision and control into agent workflows, suitable for the complexity of real-world systems.

## Pre-work

### Clone the ELI5 repo
```
git clone https://github.com/langchain-ai/adaptive-rag-101.git
```

### Create .env file

Follow the example in .env.example to fill in the necessary information to run the application.

### Install dependencies

Create a virtual enviornment
```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

## Running the Project

Now you're ready to run the notebooks! Use the command
```
jupyter notebook
```
in the root directory to open up the notebooks.

The notebooks are designed to be used in the following order:
1. eli5_tracing
2. eli5_types
2. eli5_prompting
3. eli5_experiment
