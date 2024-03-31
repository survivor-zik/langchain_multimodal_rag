from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from src.prompts import SUMMARIZER


class Chatbot:
    def __init__(self):
        prompt = ChatPromptTemplate.from_template(SUMMARIZER)
        model = ChatOpenAI(temperature=0, model="gpt-4")
        self.summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()


