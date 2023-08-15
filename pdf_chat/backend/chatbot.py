from difflib import HtmlDiff
from fpdf import FPDF
from langchain.document_loaders import (
    PyPDFLoader,
)
from langchain.agents import Tool, OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from typing import List

SYSTEM_MESSAGE_PROMPT = """
    Assistant's name is PDFBot, a help agent for answering questions about a user-provided PDF document. PDFBot
    is able to answer two types of questions:
        1. General questions about the contents of a PDF: These are questions that don't require summarization.
            Instead, these questions require that PDFBot look through the contents of the PDF to answer the user's
            question
        2. Summarization: This kind of question requires PDFBot to summarize the entire document
    No question that isn't about the document, or the document's contents should be answered. If a user asks a question
    that isn't about the document, you should tell them that you can't answer questions that aren't related to the document.
    Your answer should contain more than 1000 words
"""
MEMORY_KEY = "chat_history"


# Why split into different agents?
    # Prompt is smaller, incurring less costs and allowing for longer memory between different agents
    # More modularized: E.g. Aren’t restricted to a single model. You can use different models based on the different requirements.

class PDFChatMasterBot:
    """
    Master Agent. This agent takes care of delefating the prompt to the right agent. It
    initializes two agents for two different kidns of queries:
        1. General questions about the contents of a PDF: These are questions that don't require summarization.
            Instead, these questions require that PDFBot look through the contents of the PDF to answer the user's
            question.
        2. Summarization: This kind of question requires PDFBot to summarize the entire document
    Any question that doesn't git into these two categories won't be answered e.g. 'How many continents are there?'

    Input:
        pdf_path (str): Path to the PDF file that you want to use for your queries
    """
    def __init__(self, pdf_path: str) -> None:
        self.full_document = PyPDFLoader(pdf_path).load()
        split_documents = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0
        ).split_documents(self.full_document)

        self.qa_retrival_bot = QARetrievalBot(split_documents)
        self.summarize_bot = SummarizationBot(self.full_document)

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
                    Used when the user is asking for a summary of the document. the output always contains at least 1000 words. This tool has no input
                """,
            ),
        ]

        self.llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.0)

        self.prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=SystemMessage(content=SYSTEM_MESSAGE_PROMPT),
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

    def query(self, prompt: str):
        """
        Executes master agent using prompt. If the request is for a summary,
        will reuturn the summary and the diff, in pdf and html format respectively.
        If request is a general question, will simply return the answer as a string.

        Input:
            prompt (str): User's prompt requesting summary or answer to a question
        
        Return:
            String answer, or files, depending on user request
        """
        output = self.agent_executor(
            f"Use the document to answer the following question: {prompt}"
        )
        if len(output["intermediate_steps"]) != 0:
            if output["intermediate_steps"][0][0].tool == "document_summarization_tool":
                try:
                    # Summary
                    summary_pdf = FPDF("P", "mm", "A4")
                    summary_pdf.add_page()
                    summary_pdf.set_font("times", "", 12)
                    summary_pdf.multi_cell(w=190, txt=output["output"])

                    # HTML Diff
                    html_diff = HtmlDiff(wrapcolumn=100).make_file(
                        "\n".join(
                            [doc.page_content for doc in self.full_document]
                        ).splitlines(),
                        output["output"].splitlines(),
                        fromdesc="Original",
                        todesc="Summary",
                    )

                    return bytes(summary_pdf.output()), html_diff
                except:
                    return 'Sorry, there seems to have been a slight error on my part. I can fix myself! Just re-enter the prompt'
            else:
                return output["output"]
        else:
            return output["output"]

class QARetrievalBot:
    """
    Bot that handles general questions about the contents of a PDF.
    Does so using Langchain's ConversationalRetrievalChain, which stores
    different document parts as embeddings in a vectore store, and performs
    a similarity search between these embeddings and the user's prompt.
    See here for more https://python.langchain.com/docs/use_cases/question_answering/

    Input:
        split_documents (List[Document]): List of document split into its different
            parts
    """
    def __init__(self, split_documents: List[Document]) -> None:
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.vectorstore = Chroma.from_documents(
            documents=split_documents, embedding=GPT4AllEmbeddings()
        )
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.qa_chat = ConversationalRetrievalChain.from_llm(
            self.llm, retriever=self.vectorstore.as_retriever(), memory=self.memory
        )

    def query(self, general_question: str) -> str:
        return self.qa_chat({"question": general_question})


class SummarizationBot:
    """
    Bot that handles the summarizatin of the PDF. Does so by using
    langchain's MapReduceDocumentsChain, which uses Map-Reduce to
    summarize all splits of document (Map) before combining them to
    produce one full summary of the entire document (Reduce).
    See here for more https://python.langchain.com/docs/use_cases/summarization
    """
    def __init__(self, document: List[Document]) -> None:
        self.document = document
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

        # Map
        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main themes 
        Helpful Answer:"""

        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # Define prompt
        self.reduce_template = """
        The following is set of summaries:
        {doc_summaries}
        Take these and distill it into a final, consolidated summary of the main themes in minimum 1000 words. 
        Helpful Answer:            
        """

        reduce_prompt = PromptTemplate.from_template(self.reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="doc_summaries"
        )

        # Combines and iteravely reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        self.map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )
        self.split_docs = text_splitter.split_documents(self.document)

    def summarize(self, dummy_input) -> str:
        return self.map_reduce_chain(self.split_docs)
