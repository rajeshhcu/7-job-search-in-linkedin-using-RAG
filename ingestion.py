from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

def load_document():
    loader = Docx2txtLoader  ("./resume.docx")
    return loader.load()

def main():
    document = load_document()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    splitted_text = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore.from_documents(splitted_text, embeddings,
                                                     index_name="resume-index")
    print(vectorstore)


if __name__ == "__main__":
    main()