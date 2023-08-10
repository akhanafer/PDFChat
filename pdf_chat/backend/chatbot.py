from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class PDFChatBot:
    def __init__(self, pdf_path) -> None:
        self.full_document = UnstructuredPDFLoader(pdf_path).load()
        self.split_documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(self.full_document)
        self.vectorstore = Chroma.from_documents(documents=self.split_documents, embedding=OpenAIEmbeddings())
        self.model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        self.qa_chain = RetrievalQA.from_chain_type(self.model, retriever=self.vectorstore.as_retriever())
    
    def query(self, prompt):
        return self.qa_chain({'query': prompt})