import os
from typing import Any, List, Tuple

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone
import dotenv

from consts import INDEX_NAME

dotenv.load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(question: str, chat_history: List[Tuple[str, Any]]) -> Any:
    embeddings = get_embeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat_model = ChatOpenAI(verbose=True, temperature=0)
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return chain({"question": question, "chat_history": chat_history})


def get_embeddings():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_id, model_kwargs=model_kwargs)
    return embeddings


if __name__ == "__main__":
    query = "what are the chain type of string other than \"stuff\" 'chain_type' can receive in  RetrievalQA.from_chain_type?"
    result = run_llm(query, [])
    print()
