import os
import uuid
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import chainlit as cl
import redis
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from sklearn.metrics.pairwise import cosine_similarity

# Retrieve API key and Redis URL from environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "dburl-cingst")

llm = Ollama(
    model="llama3.1:8b-instruct-fp16",
    verbose=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# Initialize Redis client
redis_client = redis.Redis.from_url(REDIS_URL)

model_kwargs = {'trust_remote_code': True}

embd = HuggingFaceEmbeddings(model_name="nvidia/NV-Embed-v2", model_kwargs=model_kwargs)

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Answer question system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. ONLY USE THE PROVIDED CONTEXT to answer the question. "
    "If the context does not contain the information needed to answer the question, respond with 'The provided context does not contain the necessary information to answer this question.' "
    "DO NOT USE ANY OTHER knowledge or information."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

def is_relevant_to_query(query, docs):
    relevant_docs = []
    for doc in docs:
        # Basic keyword matching for now, replace with semantic similarity if needed
        if query.lower() in doc.page_content.lower():
            relevant_docs.append(doc)
    return relevant_docs


@cl.on_chat_start
async def on_chat_start():
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embd, collection_name="rag-chroma")
    cl.user_session.set("vectorstore", vectorstore)
    
    # Create a retriever and store it in the session
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    cl.user_session.set("retriever", retriever)
    
    # Create and store session ID
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content="You can create a new session or switch between existing sessions:",
    ).send()
    print("Your session started!")


@cl.action_callback("name")
async def on_select_session(action: cl.Action):
    # Switch to the selected session
    session_id = action.value
    cl.user_session.set("session_id", session_id)
    print(f"Switched to session ID: {session_id}")

    return f"Switched to session ID: {session_id}"


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    
    # Retrieve the vectorstore from the session (it is guaranteed to be set in `on_chat_start`)
    vectorstore = cl.user_session.get("vectorstore")
    
    # Force the retriever to be recreated each time to ensure the context is updated
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})   # Recreate the retriever dynamically

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create a conversational RAG chain with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Check if the question has been asked before
    question = message.content
    cached_response = redis_client.hget(f"session:{session_id}", question)
    
    retrieved_docs = retriever.get_relevant_documents(question)
    print(f"Retrieved Documents: {retrieved_docs}")
    
    # First filter by keyword relevance
    relevant_docs = is_relevant_to_query(question, retrieved_docs)

    if not relevant_docs:
        # If keyword matching failed, apply embedding-based similarity filtering
        query_embedding = embd.embed_query(question) 
        for doc in retrieved_docs:
            doc_embedding = embd.embed_query(doc.page_content)  # Use embedding function for document content
            similarity_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            print(f"Similarity score for document '{doc.page_content[:100]}...': {similarity_score}")
            if similarity_score > 0.2:  # Threshold for relevance
                relevant_docs.append(doc)

    if not relevant_docs:
        await cl.Message(content="The provided context does not contain the necessary information to answer this question.").send()
        return
    
    if cached_response:
        await cl.Message(content=cached_response.decode()).send()
    else:
        # Set the session ID in the configuration
        config = {"configurable": {"session_id": session_id}}

        # Invoke the conversational RAG chain and get the result
        result = conversational_rag_chain.invoke({"input": question}, config=config)

        # Cache the result in Redis
        redis_client.hset(f"session:{session_id}", question, result["answer"])
        print("Result Context is:", result)
        await cl.Message(content=result["answer"]).send()


# Run the Chainlit app
if __name__ == "__main__":
    cl.run()
