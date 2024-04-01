from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from src.prompts import SUMMARIZER
from src.utils import return_retriever, split_image_text_types, img_prompt_func


class Chatbot:
    def __init__(self):
        prompt = ChatPromptTemplate.from_template(SUMMARIZER)
        model = ChatOpenAI(temperature=0, model="gpt-4")
        self.summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        self.retriever = return_retriever()
        output_model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
        self.chain = (
                {
                    "context": self.retriever | RunnableLambda(split_image_text_types),
                    "question": RunnablePassthrough()
                }
                | RunnablePassthrough(img_prompt_func)
                | output_model
                | StrOutputParser
        )

    def generate_text_summaries(self, texts, tables, summarize_texts=False):
        """
        Summarize text elements
        texts: List of str
        tables: List of str
        summarize_texts: Bool to summarize texts
        """

        text_summaries = []
        table_summaries = []

        # Apply to text if texts are provided and summarization is requested
        if texts and summarize_texts:
            text_summaries = self.summarize_chain.batch(texts, {"max_concurrency": 5})
        elif texts:
            text_summaries = texts
        if tables:
            table_summaries = self.summarize_chain.batch(tables, {"max_concurrency": 5})

        return text_summaries, table_summaries

    def get_docs(self, question: str):
        docs = self.retriever.get_relevant_documents(query=question, limit=6)
        return docs
