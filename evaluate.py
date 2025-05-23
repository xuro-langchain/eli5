from langsmith import evaluate, Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from eli5 import eli5

load_dotenv(dotenv_path=".env", override=True)

# 1. Create and/or select your dataset
client = Client()
dataset_name = "eli5-golden"


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 2. Define a custom code evaluator
def conciseness(outputs: dict) -> bool:
    print(outputs)
    words = outputs["output"].split(" ")
    return len(words) <= 200


# 3. Define an LLM-as-a-judge evaluator
class CorrectnessScore(BaseModel):
    """Correctness score of the answer when compared to the reference answer."""
    score: int = Field(description="The score of the correctness of the answer, from 0 to 1")
    
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    prompt = """
    You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

    <Rubric>
        A correct answer:
        - Provides accurate information
        - Uses suitable analogies and examples
        - Contains no factual errors
        - Is logically consistent

        When scoring, you should penalize:
        - Factual errors
        - Incoherent analogies and examples
        - Logical inconsistencies
    </Rubric>

    <Instructions>
        - Carefully read the input and output
        - Use the reference output to determine if the model output contains errors
        - Focus whether the model output uses accurate analogies and is logically consistent
    </Instructions>

    <Reminder>
        The analogies in the output do not need to match the reference output exactly. Focus on logical consistency.
    </Reminder>

    <input>
        {}
    </input>

    <output>
        {}
    </output>

    Use the reference outputs below to help you evaluate the correctness of the response:
    <reference_outputs>
        {}
    </reference_outputs>
    """.format(inputs["question"], outputs["output"], reference_outputs["output"])
    structured_llm = llm.with_structured_output(CorrectnessScore)
    generation = structured_llm.invoke([HumanMessage(content=prompt)])
    return generation.score == 1


# 4. Define a function to run your application
def run(inputs: dict):
    return eli5(inputs["question"])

# 4. Run an evaluation
# For more info on evaluators, see: https://docs.smith.langchain.com/concepts/evaluation#evaluators
evaluate(
    run,
    data=dataset_name,
    evaluators=[correctness, conciseness],
    experiment_prefix="eli5-gpt4o"
)