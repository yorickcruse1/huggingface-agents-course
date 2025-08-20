from llama_index.core.agent import AgentWorkflow
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.workflow import Context
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm with token from env
llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key=os.getenv("HF_TOKEN")  # Changed 'token' to 'api_key'
)

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)

async def run_stateless_example():
    # stateless example
    response = await agent.run("What is 2 times 2?")
    print("Stateless response:", response)

async def run_stateful_example():
    # remembering state example

    ctx = Context(agent)

    response1 = await agent.run("My name is Bob.", ctx=ctx)
    response2 = await agent.run("What was my name again?", ctx=ctx)
    print("Stateful responses:", response1, response2)

query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
import asyncio

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


async def main():
    # Create agent configs
    calculator_agent = ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant. Use your tools for any math operation.",
        tools=[add, subtract],
        llm=llm,
    )

    query_agent = ReActAgent(
        name="info_lookup",
        description="Looks up information about XYZ",
        system_prompt="Use your tool to query a RAG system to answer information about XYZ",
        tools=[query_engine_tool],
        llm=llm
    )

    # Create and run the workflow
    agent = AgentWorkflow(
        agents=[calculator_agent, query_agent], root_agent="calculator"
    )

    # Run the system
    response = await agent.run(user_msg="Can you add 5 and 3?")
    print(response)

if __name__ == "__main__":
    # Run both examples and the main calculator example
    async def run_all():
        await run_stateless_example()
        await run_stateful_example()
        await main()
    
    asyncio.run(run_all())