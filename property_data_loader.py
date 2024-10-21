from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings_model_name = 'all-MiniLM-L6-v2'

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def property_data_ingest(file_path, persist_directory):

    # file_path = r"C:\Users\user\Downloads\dummy_data.csv"
    loader = CSVLoader(file_path, encoding="windows-1252")
    documents = loader.load()

    vectorstore = Chroma.from_documents(documents, embedding_function, persist_directory=persist_directory)

    print(f"Ingestion complete!")