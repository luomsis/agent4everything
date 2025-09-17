import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

# Load environment variables from .env file
load_dotenv()

# Get environment variables
llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

if llm_provider == "deepseek":
    # DeepSeek configuration
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    if deepseek_api_key:
        llm = init_chat_model(
            model=deepseek_model,
            model_provider="openai",  # DeepSeek uses OpenAI-compatible API
            api_key=deepseek_api_key,
            base_url=deepseek_base_url
        )
    else:
        # Fallback to fake model if no API key
        llm = FakeMessagesListChatModel(responses=[AIMessage(content="DeepSeek API key not configured")])

elif llm_provider == "openai":
    # OpenAI configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if openai_api_key:
        llm = init_chat_model(
            model=openai_model,
            model_provider="openai",
            api_key=openai_api_key,
            base_url=openai_base_url
        )
    else:
        # Fallback to fake model if no API key
        llm = FakeMessagesListChatModel(responses=[AIMessage(content="OpenAI API key not configured")])

else:
    # Fallback to fake model for unknown providers
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=f"Unsupported LLM provider: {llm_provider}")])