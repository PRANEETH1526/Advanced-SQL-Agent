from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
#from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import os
from dotenv import load_dotenv

load_dotenv()

mini_llm = AzureChatOpenAI(
    model_name="gpt-4o-mini",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4O_MINI"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

llm = AzureChatOpenAI(
    model_name="gpt-4o",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4O"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_4O"),
    temperature=0,
    cache=False,
)

advanced_llm = AzureChatOpenAI(
    model_name="o4-mini",
    api_version=os.getenv("AZURE_OPENAI_ADVANCED_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ADVANCED_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_ADVANCED_API_KEY"),
    temperature=0,
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
)

"""


sparse_embeddings = BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3',
        device='cpu',          # Use 'cuda:0' for GPU
        use_fp16=False,        # Set to True if using GPU and want to use half-precision
        return_dense=False,    # Disable dense embeddings
        return_sparse=True     # Enable sparse embeddings
    )

"""
