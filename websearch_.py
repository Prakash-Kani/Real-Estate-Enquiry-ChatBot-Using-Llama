from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

api_key = "Enter your api key here"
cse_id = "Enter your CSE id here"

# Function to create a retriever using GoogleSearchAPIWrapper
def create_google_search_retriever(api_key, cse_id):
    # Initialize the Google Search API Wrapper
    search = GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cse_id)
    return search
# Define a function to handle the query and search
def search_with_query_(search, user_query):
    if user_query.strip():
        print(user_query)
        return search.result(num_results=20,  query=user_query)
    else:
        return "No properties match the specified criteria."

# Function to create the chain with Google Search retriever
def rusult_create_chain(api_key, cse_id):
    # Define the prompt template for the chain
    template = """You are an intelligent assistant designed to provide information about real estate properties based on the provided context.
                ### Instructions:
                - If the user input is a property inquiry (e.g., asking for properties based on size, price, or location), retrieve and list the top 5 available properties that match the criteria from 99acres or MagicBricks.
                - If no properties are available for the user inquiry, respond with "No properties match the specified criteria."
                - If the user input does not pertain to a property inquiry, respond with "I cannot assist with that."
                - Use only the information retrieved from the context to formulate your response.
                ### Context:
                {context}
                ### User Input:
                {question}
                ### Response:
                """
    # Set up the prompt
    prompt = ChatPromptTemplate.from_template(template)
    # Create the Google Search retriever
    retriever = create_google_search_retriever(api_key=api_key, cse_id=cse_id)
    # Initialize the language model (using Llama 3.1 in this case)
    model = Ollama(model="llama3.1:latest")
    # Set up the chain
    chain = (
        {
            "context": RunnableLambda(lambda x: search_with_query_(search = retriever, user_query = x)),  # The retriever fetching the context (property listings)
            # "context": RunnableLambda(lambda x: search_with_query(search = retriever, user_query = x)), 
            "question": RunnablePassthrough()  # The user query
        }
        | prompt  # Apply the prompt template
        | model   # Use the model to generate a response based on the context
        | StrOutputParser()  # Parse the final output as a string
    )
    return chain


