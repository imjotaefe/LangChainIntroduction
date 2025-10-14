from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

gemini = init_chat_model(model="gemini-2.5-flash", model_provider="google-genai")
gemini_answer = gemini.invoke("Hello world")

print(gemini_answer.content)