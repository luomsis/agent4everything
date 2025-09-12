import os
import json
from openai import OpenAI
from typing import Optional, Dict, Any, Callable
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize OpenAI client with API key, model, and base URL"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
    
    def chat_completion(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send chat completion request to OpenAI API with optional function calling"""
        try:
            params = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            if tools:
                params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
            
            response = self.client.chat.completions.create(**params)
            return response
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response from prompt"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(messages, **kwargs)
        return response.choices[0].message.content
    


def translate_chinese_to_english(text: str) -> str:
    """Translate Chinese text to English using OpenAI API"""
    client = OpenAIClient()
    
    system_prompt = """You are a professional translator specializing in Chinese to English translation. 
    Translate the given Chinese text to accurate, natural-sounding English. 
    Maintain the original meaning, tone, and context.
    Return only the translated text without any additional commentary."""
    
    response = client.generate_response(
        prompt=f"Translate this Chinese text to English: {text}",
        system_message=system_prompt,
        temperature=0.3,
        max_tokens=1000
    )
    
    return response

def main():
    """Main function for testing the OpenAI interface"""
    try:
        client = OpenAIClient()
        
        print("OpenAI Chat Interface Started (type 'quit' to exit)")
        print("=" * 50)
        print("Special commands: 'translate [chinese text]' - Translate Chinese to English")
        
        while True:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Conversation ended")
                break
                
            if not user_input:
                continue
            
            # Check for translation command
            if user_input.lower().startswith('translate '):
                chinese_text = user_input[10:].strip()
                if chinese_text:
                    try:
                        translation = translate_chinese_to_english(chinese_text)
                        print(f"Translation: {translation}")
                    except Exception as e:
                        print(f"Translation error: {str(e)}")
                else:
                    print("Please provide Chinese text to translate after 'translate' command")
                continue
                
            try:
                response = client.generate_response(
                    prompt=user_input,
                    system_message="You are a helpful AI assistant that can get weather information. Use functions when appropriate."
                )
                
                print(f"AI: {response}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
    except ValueError as e:
        print(f"Initialization error: {str(e)}")
        print("Please set OPENAI_API_KEY environment variable")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()