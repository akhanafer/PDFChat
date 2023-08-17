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
<PLAY AROUND WITH THE SYSTEM PROMPT>
"""
MEMORY_KEY = "chat_history"

class PDFChatMasterBot:
    """
    Master Agent. This agent takes care of delegating the prompt to the right agent. It
    initializes two agents for two different kinds of queries:
        1. General questions about the contents of a PDF: These are questions that don't require summarization.
            Instead, these questions require that PDFBot look through the contents of the PDF to answer the user's
            question.
        2. Summarization: This kind of question requires PDFBot to summarize the entire document
    Any question that doesn't git into these two categories won't be answered e.g. 'How many continents are there?'

    Input:
        pdf_path (str): Path to the PDF file that you want to use for your queries
    """
    def __init__(self, pdf_path: str) -> None:
        # Step 1:
        # Load PDF using an apropriate loader. You can find a list of LangChain loaders at https://integrations.langchain.com/
        self.full_document = ...
        split_documents = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0
        ).split_documents(self.full_document)

        # Step 2:
        # Initialize the QARetrievalBot and the SummarizationBot. The former takes as parameter the split documents
        # while the latter takes in the full document. Both classes should have a function exposed to the master bot
        # that allows us to execute their respective chains
        self.qa_retrival_bot = ...
        self.summarize_bot = SummarizationBot(self.full_document)


        # Step 3:
        # Populate the list of tools using the Tool.from_function method. This method takes three parameters that help the
        # agent understand when it should be call:
            # 1. name: The tool's name
            # 2. func: The function to call when the LLM decides that this is the tool it should use for the given prompt
            # 3. description: The description of what the tool is and when it should be used
        # You can read more about tools here https://python.langchain.com/docs/modules/agents/tools/

        self.tools = [
            Tool.from_function(
                name="document_summarization_tool",
                func=self.summarize_bot.summarize,
                description="""
                    Used when the user is asking for a summary of the document. the output always contains at least 1000 words. This tool has no input
                """,
            ),
        ] # add QA tool exposed by child agents

        # Step 4:
        # Add the right parameters to the ChatOpenAI class below. Since this tutorial uses the OpenAIFunctionsAgent,
        # It will only work with LangChain's GPT class, ChatOpenAI but you can find documentation
        # for all supported models (including ChatOpenAI) here https://python.langchain.com/docs/integrations/chat/
        self.llm = ChatOpenAI() # Add necessary params


        # Defines the system prompt and a memory key (location in the prompt where memory should be apended)
        # This is always the first prompt that will be sent to our agent, since it gives the agent things like
        # context, and your described personality. Play with the system prompt above to see if you can get a good
        # PDF bot!
        self.prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=SystemMessage(content=SYSTEM_MESSAGE_PROMPT),
            extra_prompt_messages=[
                MessagesPlaceholder(variable_name=MEMORY_KEY),
            ],
        )

        # Step 5:
        # Define the memory. The memory will do as it says: give the agent memory
        # so that it can have a more conversational-like feel. more here
        # https://python.langchain.com/docs/modules/memory/
        self.memory = ...

        # Everything you've done so far is to reach these next few lines: Initializing the agent
        # with the proper model, tools, system prompt, and memory
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
        # Step 6:
        # Execute the self.agent_executer with the user's prompt and return its output

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
        # Step 1:
        # Initialize vectore store. You can find all vector stores supported by
        # langchain here https://python.langchain.com/docs/integrations/vectorstores/
        self.vectorstore = ...

        # Step 2:
        # Create the model you would like this chain to use. It can be any model you'd like
        # from this list https://python.langchain.com/docs/integrations/llms/
        self.llm = ...

        # Step 3:
        # Initialize the ConversationalRetrievalChain using the vector store and model you've defined
        # above. More on this chain here https://python.langchain.com/docs/use_cases/question_answering/
        self.qa_chat = ...

    def query(self, general_question: str) -> str:
        # Step 4:
        # Use input the prompt into your chain, so it can go through the sequence of
        # steps that it needs to to provide an output. Return this output to the master bot
        return


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
