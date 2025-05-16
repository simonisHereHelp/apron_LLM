import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# === Path setup ===
folder_path = "./diy_articles"
file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".md")]

# === Load and split ===
all_chunks = []
for filepath in file_list:
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
    chunks = splitter.split_text(docs[0].page_content)

    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(filepath)
    all_chunks.extend(chunks)

print(f"✅ Loaded {len(file_list)} files and {len(all_chunks)} chunks.")

# === Embed and build vector store ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(all_chunks, embedding=embeddings)
vector_db.save_local("vector_db/diy_articles_faiss")
