from langchain.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.agents import Tool, OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.llms import GPT4All
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any

SYSTEM_MESSAGE_PROMPT = """
    Assistant's name is PDFBot, a help agent for answering questions about a user-provided PDF document. PDFBot
    is able to answer two types of questions:
        1. General questions about the contents of a PDF: These are questions that don't require summarization.
            Instead, these questions require that PDFBot look through the contents of the PDF to answer the user's
            question
        2. Summarization: This kind of question requires PDFBot to summarize the entire document
    No question that isn't about the document, or the document's contents should be answered. If a user asks a question
    that isn't about the document, you should tell them that you can't answer questions that aren't related to the document.
"""
MEMORY_KEY = "chat_history"

#Why split into different agents?
	# Prompt is smaller, incurring less costs and allowing for longer memory between different agents
	# More modularized: E.g. Aren’t restricted to a single model. You can use different models based on the different requirements.
#TODO: PAss system prompt to tools so they don't answer generic questions not related to the document
#TODO: compute diff between full PDF and summarized PDF
class PDFChatMasterBot:
    def __init__(self, pdf_path) -> None:
        full_document = PyPDFLoader(pdf_path).load()
        print(full_document)
        split_documents = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0
        ).split_documents(full_document)

        self.qa_retrival_bot = QARetrievalBot(split_documents)
        self.summarize_bot = SummarizationBot(full_document)

        self.tools = [
            Tool.from_function(
                name="document_QA_tool",
                func=self.qa_retrival_bot.query,
                description="""
                    Used when you need to answer a general question about the document's contents. This is useful for
                    when the user is asking questions about the document, and isn't asking for you to summarize the
                    document.
                    Input:
                        general_question (str): The user's general question concerning the document's contents
                """,
            ),
            Tool.from_function(
                name="document_summarization_tool",
                func=self.summarize_bot.summarize,
                description="""
                    Used when the user is asking for a summary of the document. This tool has no input
                """,
            ),
        ]

        self.llm = ChatOpenAI(
            temperature=0.0,
            top_p=1,
        )

        self.prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=SystemMessage(
                content=SYSTEM_MESSAGE_PROMPT
            ),
            extra_prompt_messages=[
                MessagesPlaceholder(variable_name=MEMORY_KEY),
            ],
        )

        self.memory = ConversationBufferMemory(
            memory_key=MEMORY_KEY,
            return_messages=True,
            input_key="input",
            output_key="output",
        )

        self.agent = OpenAIFunctionsAgent(
            llm=self.llm, tools=self.tools, prompt=self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True,
        )

    def query(self, prompt: str) -> str:
        result = self.agent_executor(prompt)
        return result


class QARetrievalBot:
    def __init__(self, split_documents: List[Document]) -> None:
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.vectorstore = Chroma.from_documents(
            documents=split_documents, embedding=GPT4AllEmbeddings()
        )
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        self.qa_chat = ConversationalRetrievalChain.from_llm(
            self.llm, retriever=self.vectorstore.as_retriever(), memory=self.memory
        )

    def query(self, general_question: str) -> str:
        return self.qa_chat({"question": general_question})


class SummarizationBot:
    def __init__(self, document: List[Document]) -> None:
        self.document = document
        # Define prompt
        self.prompt_template = """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:
        """
        self.prompt = PromptTemplate.from_template(self.prompt_template)

        # Define LLM chain
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

        # Define StuffDocumentsChain
        self.stuff_chain = StuffDocumentsChain(
            llm_chain=self.llm_chain, document_variable_name="text"
        )

    def summarize(self, dummy_input) -> str:
        return self.stuff_chain(self.document)
