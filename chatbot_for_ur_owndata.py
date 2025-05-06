from dotenv import load_dotenv
import os
import requests

load_dotenv()

# Set environment variables
os.environ['USER_AGENT'] = 'RAGUserAgent'
os.environ['LLAMA_API_KEY'] = 'ee18e8e2-fe77-4bcc-8bb1-1bf81ed0ba6d'  # Replace with actual API key
os.environ['LLAMA_API_URL'] = 'https://console.llmapi.com/en/dashboard/api-token'  # Replace with actual API request URL

LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')
LLAMA_API_URL = os.getenv('LLAMA_API_URL')

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Load and process the document
loader = WebBaseLoader("https://www.bing.com/search?q=llama+geeks+for+geeks&form=ANNTH1&refig=9870a60703ff424fac39f31a231d1a34&pc=HCTS")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Create conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the conversation prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can answer questions about the US Constitution.
                 Use the provided context to answer the question and reference previous conversations 
                 when relevant. If you are unsure of the answer, say 'I don't know'."""), 
    ("system", "Previous conversation:\n{chat_history}"), 
    ("user", "Question: {question}\nContext: {context}")
])

def ask_llama(question, context, chat_history):
    """Send query to Llama API and return the response"""
    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-2-7b-chat",  # Ensure this matches the actual model name in API docs
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"Question: {question}\nContext: {context}\nChat History: {chat_history}"}
        ]
    }
    
    response = requests.post(LLAMA_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return "Error: API returned an unexpected format."
    else:
        return f"Error {response.status_code}: {response.text}"

def main():
    print("Welcome to the Constitutional Q&A Assistant!")
    print("Ask questions about the US Constitution. Type 'exit' to end the conversation.\n")

    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break

        chat_history = memory.load_memory_variables({})['chat_history']
        docs = retriever.invoke(query)

        response = ask_llama(query, docs, chat_history)
        print(f"\nAssistant: {response}")

        memory.save_context({"input": query}, {"output": response})

    print("\nFull Conversation History:")
    print(memory.load_memory_variables({})['chat_history'])

if __name__ == "__main__":
    main()