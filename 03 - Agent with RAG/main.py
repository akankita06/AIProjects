# connect to database
# split the data documents into chunks
# embed the chunk with embeddings
# store the embedded chunks in a vector database

# query the vector database
# retrieve the most relevant chunks
# combine the chunks with the prompt
# generate the response

from pathlib import Path
import time
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import langchain
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


folder_path = Path("database")
vectorstore_path = Path("vectorstore")

load_dotenv()

### INDEXING

# 1. load the data documents

loader = loader = DirectoryLoader(
    str(folder_path),
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
docs = loader.load()

# 2. split the text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# 3. embed the chunks with embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 4. store the embedded chunks in a vector database
# To avoid rate limting exceptions by sending too many chunks at the same time, we batch the chunks
batch_size = 100
delay_seconds = 30
vectorstore = None
for i in range(0, len(chunks), batch_size):
    print("Processing batch", i//batch_size + 1, "of", len(chunks)//batch_size + 1)
    if i == 0:
        vectorstore = Chroma.from_documents(chunks[i: i+batch_size], embeddings, persist_directory=str(vectorstore_path))
    else:
        vectorstore.add_documents(chunks[i:i+batch_size])
    time.sleep(delay_seconds)
    


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


### GENERATION

prompt = hub.pull("rlm/rag-prompt")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

response = chain.invoke("What are my goals for work? List them out in a bullet point format")
print(response)