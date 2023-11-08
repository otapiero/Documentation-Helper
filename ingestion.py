import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import dotenv

dotenv.load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8"
    )
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", "", " "]
    )
    documents = splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} documents")

    # Add url to metadata
    for doc in documents:
        path = doc.metadata["source"]
        url = path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": url})

    # Embedding documents
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_id, model_kwargs=model_kwargs)
    Pinecone.from_documents(
        documents=documents, embedding=embeddings, index_name="langchain-doc-index"
    )

    print("Done ingesting documents")


if __name__ == "__main__":
    print("Ingesting documents")
    ingest_docs()
