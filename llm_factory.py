import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def get_llm(provider="ollama", model_name="llama3"):
    """
    Returns the Chat Model based on the provider.
    """
    if provider == "ollama":
        print(f"--- Using Ollama (Model: {model_name}) ---")
        return ChatOllama(model=model_name, temperature=0)
    
    elif provider == "openai":
        print(f"--- Using OpenAI (Model: {model_name}) ---")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        return ChatOpenAI(model=model_name, temperature=0)
    
    elif provider == "gemini":
        print(f"--- Using Google Gemini (Model: {model_name}) ---")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_embeddings(provider="ollama", model_name="llama3"):
    """
    Returns the Embeddings Model based on the provider.
    """
    if provider == "ollama":
        return OllamaEmbeddings(model=model_name)
    
    elif provider == "openai":
        # Usually text-embedding-3-small or text-embedding-ada-002
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    elif provider == "gemini":
        # usually models/embedding-001
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    else:
        raise ValueError(f"Unknown provider: {provider}")