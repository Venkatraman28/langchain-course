from typing import Any, Dict

from state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("Retrieving...")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
