import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """Helper function to format retrieved documents."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain():
    """
    This function contains all the core logic for your RAG chatbot.
    It sets up the models, indexes the data, and returns the runnable chain.
    """
    print("Setting up LLM and Embedding Model...")

    os.environ["GROQ_API_KEY"] = "ENTER API KEY"

    # The ChatGroq() function will now automatically find the key
    # that we just set in os.environ.
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- Phase 1: Indexing (Build the Knowledge Base) ---
    print("Phase 1: Indexing data from LangChain docs...")
    loader = WebBaseLoader("https://docs.langchain.com/oss/python/langchain/overview")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("Indexing complete.")

    # --- Phase 2: Create RAG Chain ---
    print("Phase 2: Setting up RAG chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the user's question based only on the following context. 
If the answer is not in the context, say 'I don't know.'

Context:
{context}

Question:
{question}
"""
    )

    # Create the final, runnable chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain setup complete.")
    return rag_chain


if __name__ == "__main__":
    # This allows you to run `python rag_logic.py` to test the chain
    print("Testing RAG logic file directly...")
    
    # Make sure to set your GROQ_API_KEY in your terminal first!
    try:
        test_chain = create_rag_chain()
        
        print("\n--- Chatbot is ready for testing ---")
        
        question_1 = "What is LangChain?"
        print(f"User Query 1: {question_1}")
        response = test_chain.invoke(question_1)
        print(f"Chatbot Answer 1: {response}")
        
        print("-" * 20)
        
        question_2 = "What is the capital of France?"
        print(f"User Query 2: {question_2}")
        response = test_chain.invoke(question_2)
        print(f"Chatbot Answer 2: {response}")
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        print("Make sure your GROQ_API_KEY is set in your environment.")