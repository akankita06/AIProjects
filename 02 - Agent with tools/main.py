from re import search
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tools import search_tool, wikipedia_tool
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

class SearchResponse(BaseModel):
    search_query: str = Field(description="The search query to be used to search the internet")
    best_result: str = Field(description="The best search result to be used to answer the user query")
    source: str = Field(description="The source of the best search result")
    tools_used: list[str] = Field(description="The tools used to answer the user query from the tools provided")

parser = PydanticOutputParser(pydantic_object=SearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        You are a search assistant that will help search the most updated information on the internet. 
        Answer the user query and use necessary and appropriatetools depending on the user query.
        Wrap the output in this format - \n{format_instructions} and provide no other text,
        You are only allowed to use the tools provided to you. If you don't find the relevant tool, say you don't know.
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wikipedia_tool]
agent= create_tool_calling_agent(
    llm=llm, prompt=prompt, tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# response = agent_executor.invoke({"query": "What is the latest news regarding H1b visa?"})
response = agent_executor.invoke({"query": "What is capital of India?"})

try:
    structured_response = parser.parse(response["output"])
    print(structured_response.model_dump_json(indent=2))
except Exception as e:
    print(f"Error: {e}", "Raw response: ", response)
print(response)