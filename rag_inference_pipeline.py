import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import PreTrainedTokenizerBase, BitsAndBytesConfig

# === GPU / Device Info ===
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🧠 Using device:", torch.cuda.get_device_name(0))

# === Step 1: Load Vector DB ===
start_vectordb = time.time()
db_path = "vector_db/diy_articles_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(
    db_path,
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})  # narrow for grounding
end_vectordb = time.time()
print(f"🗃️ VectorDB Load Time: {end_vectordb - start_vectordb:.2f}s (Expected: 0.5–2s)")

# === Step 2: Load LLaMA3-3B-Instruct ===
start_model = time.time()
model_path = "models/llama3-RAG3b2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",
    quantization_config=bnb_config
)
print("\n 📦 Model loaded on:", model.device)

end_model = time.time()
print(f"🧠 LLaMA3 Load Time: {end_model - start_model:.2f}s (Expected: 10–30s, one-time startup)\n")

# === RAG Function with Strict Context Use ===
def rag_qa(query):
    start_total = time.time()

    # Retrieval
    start_retrieve = time.time()
    docs = retriever.get_relevant_documents(query)
    end_retrieve = time.time()
    print(f"🔍 Retrieval Time: {end_retrieve - start_retrieve:.2f}s (Expected: 0.01–0.10s)")

    for doc in docs:
        print(f"📄 Source: {doc.metadata.get('source', 'unknown')}")

    # Prompt Construction
    start_prompt = time.time()

    # include image_URL in context
    context_lines = []
    for doc in docs:
        lines = doc.page_content.split("\n")
        for line in lines:
            if line.strip().startswith("!["):
                # Make image lines more prominent
                context_lines.append(f"🖼️ IMAGE: {line.strip()}")
            else:
                context_lines.append(line.strip())
    context = "\n".join(context_lines)

    # Token length check (after context is built)
    if isinstance(tokenizer, PreTrainedTokenizerBase):
        n_tokens = len(tokenizer(context)["input_ids"])
        if n_tokens < 50:
            return "⚠️ Not enough relevant content retrieved to answer confidently. Please rephrase."

    # === One-shot formatted example ===
    example_output = """
Example:

Step 1: Choose Your Lighting Type

🖼️ IMAGE: ![Fluorescent under cabinet lights installed in a walk-in closet.](https://dam.thdstatic.com/...)

Step 2: Measure your cabinet space...

...

Step X: Final Adjustment

Please note that installing under cabinet lighting may involve electrical safety steps. If you're not confident with wiring, consult a licensed electrician.

Note:
 📄 Source: best-under-cabinet-lighting.md
"""

    prompt = f"""You are a helpful DIY assistant.

Answer the following question using only the information provided in the context below. Base your reasoning on these materials. Do not introduce unrelated advice.

The answer should be in step-by-step format using "Step 1:", "Step 2:", etc., these steps should be based on the content below.

If the context includes markdown images (shown as lines starting with '🖼️ IMAGE:'), include them in your answer. Use them to visually support step-by-step explanations. Do not remove or ignore the images.

If the context does not cover the topic clearly, summarize what is known and acknowledge if anything is missing.




Context:
{context}

Question: {query}
Answer:"""
    end_prompt = time.time()
    print(f"🧱 Prompt Build Time: {end_prompt - start_prompt:.2f}s (Expected: < 0.01s)")

    # Inference
    start_llm = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    end_llm = time.time()
    print(f"🧠 LLM Inference Time: {end_llm - start_llm:.2f}s (Expected: 1.5–4.0s)")
    print(f"⏱ Total RAG Round Time: {end_llm - start_total:.2f}s\n")
    # === Clean and append footnote ===
    output = output.replace("The final answer is:", "").strip()

    sources = sorted(set(doc.metadata.get("source", "unknown") for doc in docs))
    footnote = "\nNote:\n" + "\n".join(f" 📄 Source: {src}" for src in sources)

    return output + "\n" + footnote

# === CLI Loop ===
if __name__ == "__main__":
    print("🛠️ DIY-RAG Assistant. Ask anything...")
    while True:
        print()
        q = input('❓ How can I help with your home improvement project today? (e.g., "How to tile a basement shower"): ')

        if q.lower() in ["exit", "quit"]:
            break
        response = rag_qa(q)
        print("\n\n\n📎 Response:\n", response.split("Answer:")[-1].strip())
