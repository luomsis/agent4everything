from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    AZURE_OPENAI = "azure"
    ANTHROPIC = "anthropic"

class LLMConfig(BaseModel):
    """LLM 配置模型"""
    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="模型提供商")

    # OpenAI 配置
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API 密钥")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI 模型名称")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API 基础URL")

    # DeepSeek 配置
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API 密钥")
    deepseek_model: str = Field(default="deepseek-chat", description="DeepSeek 模型名称")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", description="DeepSeek API 基础URL")

    # Azure OpenAI 配置
    azure_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API 密钥")
    azure_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI 终端点")
    azure_deployment: Optional[str] = Field(default=None, description="Azure OpenAI 部署名称")

    # 通用配置
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="最大令牌数")
    timeout: int = Field(default=30, description="API 超时时间(秒)")

    class Config:
        use_enum_values = True