from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model_name = 'llama3.2:latest'
embeddings_model_name =  "all-MiniLM-L6-v2"

model = Ollama(model = model_name)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

store = {}

def History_Chain(retriever):
    contextualize_q_system_prompt = (
        "Given the previous chat history and the latest user message, reformulate the latest message into a clear and standalone question related to real estate in Chennai. "
        "Ensure that the reformulated question includes all necessary context so that it can be understood without referring to the chat history. "
        "Do NOT answer the question; just reformulate it if needed. If the latest message is not related to real estate, indicate that it is outside the scope."
    )
   
   

# #     contextualize_q_system_prompt = (
# #     "Given the user's initial input, which includes the position, role, and required skills, generate an interview-style question "
# #     "that is unique and relevant to the specified role and skills. Ensure that the question has not been asked previously in this interview. "
# #     "After the user provides an answer, generate up to two follow-up questions based on that answer, maintaining relevance to the topic. "
# #     "Avoid repetitive or overly lengthy questions to mimic a natural, real-time interview flow."
# # )
# #     contextualize_q_system_prompt = (
# #     "Given the user's initial input, which includes the position, role, and required skills, generate a single interview-style question "
# #     "relevant to the specified role and skills. After the user answers, generate one follow-up question at a time based on their response. "
# #     "Each follow-up question should stay focused on the current topic and not shift to a new topic until the follow-up questions are exhausted."
# #     "Ensure that the interview remains engaging, realistic, and that questions are concise without being repetitive."
# # )
#     contextualize_q_system_prompt = (
#         "You are an AI assistant conducting a mock interview tailored to a specific job role. Your task is to generate relevant interview questions"
#         "based on the role description and required skills provided by the user. You will ask one question at a time and dynamically generate"
#         "two follow-up questions based on the interviewee's responses to dive deeper into specific topics. After each follow-up, wait for the candidate’s response before proceeding.\n"
#         "Your goal is to simulate a real interview scenario by:\n"
#         "1. Asking questions in a logical flow, where each question is relevant to the previous response or topic."
#         "2. Adapting your follow-up questions to address key points mentioned by the interviewee."
#         # "3. Refraining from indicating how many questions are left or summarizing at any point, focusing solely on continuing the conversation naturally."
#         "4. Avoid listing multiple questions at once; you should ask them individually."
#         "5. Avoid  multiple sentence question ask questions using maximum three sentence."
#         "\n\n"
#         "Ensure the conversation feels fluid, engaging, and focused on the candidate's problem-solving abilities and understanding of the role. After each question, allow the user to respond before asking the next one."
#     )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                                                                [   
                                                                    ("system", contextualize_q_system_prompt),
                                                                    MessagesPlaceholder("chat_history"),
                                                                    ("human", "{input}"),
                                                                ]
                                                            )
    history_aware_retriever = create_history_aware_retriever(model,
                                                             retriever, 
                                                             contextualize_q_prompt
                                                            )
    return history_aware_retriever


def Question_Answer_Chain():

    system_prompt = (
        "You are a specialized real estate assistant for Chennai. Your task is to help users find properties based on their requirements. "
        "When a user provides a message, perform the following steps:\n\n"
        "1. **Determine Relevance:** Check if the user's input is related to real estate. If not, respond with 'I'm sorry, I can only assist with real estate inquiries.'\n\n"
        "2. **Extract Information:** Identify and extract the following categories from the user's input if present:\n"
        "   - **Type:** Residential / Commercial\n"
        "   - **City:** Chennai\n"
        "   - **Location:** Specific area within Chennai\n"
        "   - **Sub Type:** Buy / Rent\n"
        "   - **Buy Type:** Individual House / Flat / Plot (for Residential)\n"
        "   - **Rent Type:** Full House / PG / Flatmates (for Residential)\n"
        "   - **Property Type:** Office Space, Coworking, Warehouse, etc. (for Commercial)\n"
        "   - **BHK Type:** 1, 2, 3\n"
        "   - **Price Range:**Any price or price range specified by the user\n\n"
        "3. **Identify Missing Categories:** If any of the above categories are missing, ask the user a simple, single question to obtain the missing information. Only one question should be asked at a time.\n\n"
        "4. **Provide Response:** Once all necessary categories are gathered, provide a relevant response based on the user's requirements, such as listing available properties that match the criteria.\n\n"
        "5. **Format:** Ensure all responses are clear, concise, and user-friendly."
        "\n\n"
        "{context}"
    )

    system_prompt = (
    "You are a specialized real estate assistant designed to help users find properties based on their specific requirements. "
    "When a user provides a message, follow these steps:\n\n"
    "1. **Determine Relevance:** Assess if the user's input is related to real estate. If not, respond with 'I'm sorry, I can only assist with real estate inquiries.'\n\n"
    "2. **Extract Information:** Identify and extract the following categories from the user's input if present:\n"
    "   - **Property Type:** Residential / Commercial\n"
    "   - **City:** Any city specified by the user\n"
    "   - **Area:** Specific area within the specified city\n"
    "   - **Transaction Type:** Buy / Rent / Lease\n"
    "   - **Property Subtype:** \n"
    "       - **For Residential:** Individual House / Flat / Plot\n"
    "       - **For Commercial:** Office Space / Coworking / Warehouse\n"
    "   - **BHK Type:** 1 BHK / 2 BHK / 3 BHK / etc.\n"
    "   - **Price Range:** Any price or price range specified by the user\n\n"
    "3. **Identify Missing Information:** If any of the above categories are missing, ask the user a simple, single question to obtain the missing information. Only one question should be asked at a time without mentioning the missing categories explicitly.\n\n"
    "4. **Provide Response:** Once all necessary categories are gathered, provide a relevant response based on the user's requirements, such as listing available properties that match the criteria. Do not include the extracted information in the response.\n\n"
    "5. **Format:** Ensure all responses are clear, concise, and user-friendly.\n\n"
    "6. **Fallback:** If the user's input doesn't match any of the expected patterns or if it's unrelated to real estate, respond with 'I don't know.'"
    "7. Do not include the extracted information in the response"
    "9. **Final Output Format:** Once all necessary categories are gathered Respond in JSON format as follows:\n"
                "{{\n"
                    "'categories': {{'category_name': 'value', ...}},\n"
                    
                    # "'missing_categories': ['missing_category_name'],\n"
                    "'ai_message': 'ai response',"
                    "'final_response': true/false"
                "}}\n"

        "10. **Final Response:** If all categories are identified and extracted, set 'final_response' to true. If any categories are missing, set it to false."
        ""
    "\n\n"
        "{context}"

)
    
    # system_prompt = (
    #     "You are a specialized real estate assistant designed to help users find properties based on their specific requirements.\n"
    #     "When a user provides a message, follow these steps:\n"

    #     "1. **Determine Relevance:** Assess if the user's input is related to real estate. If not, respond with 'I'm sorry, I can only assist with real estate inquiries.'"

    #     "2. **Extract Information:** Identify and extract the following categories from the user's input if present:\n"
    #             "- **Property Type:** Residential / Commercial"
    #             "- **City:** Any city specified by the user"
    #             "- **Area:** Specific area within the specified city"
    #             "- **Transaction Type:** Buy / Rent / Lease"
    #             "- **Property Subtype:**"
    #                 "- **For Residential:** Individual House / Flat / Plot"
    #                 "- **For Commercial:** Office Space / Coworking / Warehouse"
    #             "- **BHK Type:** 1 BHK / 2 BHK / 3 BHK / etc."
    #             "- **Price Range:** Any price or price range specified by the user"

    #     "3. **Identify Missing Information:** If any of the above categories are missing, ask the user a simple, single question to obtain the missing information. Only one question should be asked at a time without mentioning the missing categories explicitly."

    #     "4. **Provide Response:** Once all necessary categories are gathered, provide a relevant response based on the user's requirements, such as listing available properties that match the criteria."

    #     "5. **Format:** Ensure all responses are clear, concise, and user-friendly."

    #     "6. **Fallback:** If the user's input doesn't match any of the expected patterns or if it's unrelated to real estate, respond with 'I don't know.'"

    #     "7. If any of the above categories are missing, ask the user a simple, single question to obtain the missing information. Only one question should be asked at a time without mentioning the missing categories explicitly."
    #     # "8. **Update Information:** Based on the user's response to the question generated for missing categories, update the `extracted_categories` and `missing_categories` accordingly, reflecting any new information provided by the user."

    #     "9. **Final Output Format:** Respond in JSON format as follows:\n"
    #             "{{\n"
    #                 "'categories': {{'category_name': 'value', ...}},\n"
                    
    #                 # "'missing_categories': ['missing_category_name'],\n"
    #                 "'ai_message': 'ai response',"
    #                 "'final_response': true/false"
    #             "}}\n"

    #     "10. **Final Response:** If all categories are identified and extracted, set 'final_response' to true. If any categories are missing, set it to false."
    #     ""
    #     "\n\n"
    #     "{context}"

    # )

    qa_prompt = ChatPromptTemplate.from_messages(
                                                    [
                                                        ("system", system_prompt),
                                                        MessagesPlaceholder("chat_history"),
                                                        ("human", "{input}"),
                                                    ]
                                                )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    return question_answer_chain

def RAG_Chain(retriever):
    history_aware_retriever = History_Chain(retriever)
    question_answer_chain = Question_Answer_Chain()
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def Enquiry_Chain(filename):

    db = Chroma(persist_directory= filename, embedding_function=embeddings)

    retriever = db.as_retriever()
    rag_chain = RAG_Chain(retriever)
    return RunnableWithMessageHistory(
                                        rag_chain,
                                        get_session_history,
                                        input_messages_key="input",
                                        history_messages_key="chat_history",
                                        output_messages_key="answer",
                                    )
