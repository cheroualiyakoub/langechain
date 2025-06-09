# from fastapi import FastAPI
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI()

# # Connect to Chroma vector DB running in container
# persist_directory = "/data"  # Chroma data directory (inside container)

# embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
# client = Chroma(persist_directory=persist_directory, embedding_function=embedding_model, chroma_api_impl="rest", chroma_server_host="vector_db", chroma_server_http_port="8000")

# llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=client.as_retriever())

# @app.get("/query/")
# async def query(q: str):
#     result = qa.run(q)
#     return {"answer": result}


from src.llm.router import LLMRouter

from fastapi import FastAPI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = LLMRouter(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_BASE"), model="gpt-4.1-nano", provider="openai")

@app.get("/test-token")
async def test_token():
    try:
        response = client.ask("Hello! Can you confirm you're working?")
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
