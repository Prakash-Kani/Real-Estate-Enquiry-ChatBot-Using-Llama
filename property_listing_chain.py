from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



model_name = 'llama3.1:latest'

embeddings_model_name = 'all-MiniLM-L6-v2'

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_retriever(persist_directory):


    db = Chroma(persist_directory= persist_directory, embedding_function =embeddings)

    retriever = db.as_retriever()
    return retriever


def create_chain(persist_directory):
    template = """You are an intelligent assistant designed to provide information about real estate properties based on the provided context.

                        ### Instructions:
                        - If the user input is a property inquiry (e.g., asking for plots in a specific location), retrieve and list the top available properties that match the criteria from the CSV file.
                        # - If no properties are available in the context for the user inquiry, respond with "No properties match the specified criteria."
                        - If the user input does not pertain to a property inquiry, respond with "I cannot assist with that."
                        - Use only the information retrieved from the context to formulate your response without any additional commentary.

                        ### Context:
                        {context}

                        ### User Input:
                        {question}

                        ### Response:
                        """
    template = """You are an intelligent assistant designed to provide information about real estate properties based on the provided context.

                    ### Instructions:
                    - If the user input is a property inquiry (e.g., asking for properties based on size, price, or location), retrieve and list the top 10 available properties that match the criteria from the CSV file.
                    - If no properties are available in the context for the user inquiry, respond with "No properties match the specified criteria."
                    - If the user input does not pertain to a property inquiry, respond with "I cannot assist with that."
                    - Use only the information retrieved from the context to formulate your response. 

                    ### Context:
                    {context}

                    ### User Input:
                    {question}

                    ### Response:
                    """


    prompt = ChatPromptTemplate.from_template(template)

    retriever = create_retriever(persist_directory= persist_directory)
    model = Ollama(model=model_name)
    # Set up the chain
    chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | prompt
                | model
                | StrOutputParser()
            )
    return chain