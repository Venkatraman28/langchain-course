import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

MODEL = "gpt-4o-mini"

"""
    Used langchain to load the documents
    Used text splitters to split the loaded documents into smaller chunks
    Embed it and store them in the Vector DB (Pinecone)
"""

def main():
    loader = TextLoader("mediumblog1.txt",  encoding='UTF-8')
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()

    print("ingesting...")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("finished ingestion...")


if __name__ == "__main__":
    main()