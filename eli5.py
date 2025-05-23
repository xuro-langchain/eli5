from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables
load_dotenv(dotenv_path=".env", override=True)


# Initialize web search tool
web_search_tool = TavilySearchResults(max_results=1)

# Define prompt template
prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
Your job is to answer the provided question so that even a 5 year old can understand it. 
You have provided with relevant background context to answer the question.

Question: {question} 

Context: {context}

Answer:"""
# print("Prompt Template: ", prompt)


# Create Application
openai_client = wrap_openai(OpenAI())

@traceable
def search(question):
    web_docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in web_docs])
    return web_results
    
@traceable
def explain(question, context):
    formatted = prompt.format(question=question, context=context)
    
    completion = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": formatted},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )
    return completion.choices[0].message.content

@traceable
def eli5(question):
    context = search(question)
    answer = explain(question, context)
    return answer

# Run the application
question = "What is complexity economics?"
print(eli5(question))