from langchain.chat_models import init_chat_model


from llama_index.llms.openai import OpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI

#from llama_index.llms.google_genai import GoogleGenAI

import os 
if not os.environ.get("GROQ_API_KEY"):
  print("API Erro")
else:
  groq_api = os.environ.get("GROQ_API_KEY")

from llama_index.llms.groq import Groq



def get_llama_index_llm(model_name: str):

    return "Service Not Availabe"

    '''
    """
    Returns an initialized chat LLM. 
    Supported: "openai", "groq", "gemini", etc.
    """

    if "gpt" in model_name:

        return OpenAI(model=model_name)
    
    elif model_name == "llama3-8b-8192":
        return Groq(model="llama3-70b-8192", api_key=groq_api)
    
    elif model_name == "gemma2-9b-it":
        return Groq(model="gemma2-9b-it", api_key=groq_api)
    
    
    
    
    
    
    
    
    elif model_name == "gemini-2.0-flash":
        # Replace "gemini-llm-name" with actual Gemini identifier
        return Groq(model="gemma2-9b-it", api_key=groq_api)
        #return GoogleGenAI("gemini-2.0-flash", model_provider="google_genai")
    
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")

    '''
    




def get_llm(model_name: str):
    """
    Returns an initialized chat LLM. 
    Supported: "openai", "groq", "gemini", etc.
    """

    if "gpt" in model_name:
        return init_chat_model(model_name, model_provider="openai")

    elif model_name == "llama3-8b-8192":
        return init_chat_model("llama3-8b-8192", model_provider="groq")
    
    elif model_name == "gemma2-9b-it":
        return init_chat_model("gemma2-9b-it", model_provider="groq")
    
    
    
    
    elif model_name == "llama-3.3-70b-versatile":
        # Replace "groq-llm-name" with actual Groq model identifier
        return init_chat_model("llama-3.3-70b-versatile", model_provider="groq")


    elif model_name == "gemini-2.0-flash":
        # Replace "gemini-llm-name" with actual Gemini identi
        return init_chat_model("llama-3.3-70b-versatile", model_provider="groq")  ## 
        #return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")
