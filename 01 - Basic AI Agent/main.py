from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel, Field
import os

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str = Field(description="The topic of the research")
    summary: str = Field(description="A summary of the research")
    sources: list[str] = Field(description="A list of sources used to answer the question")
    tools: list[str] = Field(description="A list of tools used to answer the question")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        You are a research assistant that will help generate a research paper.
        Answer the user query and use necessary tools.
        Wrap the output in this format - \n{format_instructions} and provide no other text 
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm, prompt=prompt  , tools=[])
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=False)

response = agent_executor.invoke({"query": "Why am I always so anxious and under information overload?"})

try:
    structured_response = parser.parse(response["output"])      
    print(structured_response.model_dump_json(indent=2))
except Exception as e:
    print(f"Error: {e}", "Raw response: ", response)