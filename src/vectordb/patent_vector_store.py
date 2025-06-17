from src.data_pipline.json_loader import load_patents_from_dir
from src.data_pipline.chunker import chunk_patent
from src.embeddings.openai import get_embedder
from langchain.vectorstores import Chroma
from langchain.schema import Document

class PatentVectorStore:
    def __init__(self, data_dir, persist_dir):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embedder = get_embedder()
        self.vectorstore = None

    def build(self):
        # Load + chunk + embed
        patents = load_patents_from_dir(self.data_dir)
        docs = []
        for patent in patents:
            chunks = chunk_patent(patent)
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                ))
        self.vectorstore = Chroma.from_documents(docs, self.embedder, persist_directory=self.persist_dir)
        self.vectorstore.persist()

    def load(self):
        self.vectorstore = Chroma(embedding_function=self.embedder, persist_directory=self.persist_dir)

    def search(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)
