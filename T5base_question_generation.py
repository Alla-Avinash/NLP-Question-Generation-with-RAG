import json
import sys
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
import spacy


# Chunking the textbook ---------------------------------------------------------------------------------------------------------------------------------------

# Load SpaCy model
# nlp = spacy.load("en_core_web_sm")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def split_into_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def create_chunks(sentences, window_size=2):
    return [" ".join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]

def chunk_textbook(pdf_path: str, output_path: str, window_size: int = 2):
    "window_size: represents the number of sentences that have to be present in a each chunk"
    # problem: want to extract the content from 100 pages at a time instead of all the pages so to reduce the memory requirement needed
    print("Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    print("Splitting into sentences...")
    sentences = split_into_sentences(text)

    print("Creating chunks...")
    chunks = create_chunks(sentences, window_size)

    print(f"Saving {len(chunks)} chunks to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)





# Create an Embedding Vector for the Chunks and then create a Faiss Index for the chunks ------------------------------------------------------------------------------------------------

# Step 1: Load Textbook Chunks from JSON file
def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

# Step 2: Generate Embeddings for the Chunks
def generate_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")           
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Step 3: Create and Save the FAISS Index
def create_faiss_index(embeddings, index_path="textbook_index.faiss"):
    dimension = embeddings[0].shape[0]  # Get the dimensionality of the embeddings
    index = faiss.IndexFlatL2(dimension)  # FAISS index with L2 (Euclidean) distance metric
    index.add(np.array(embeddings))  # Add embeddings to the index
    faiss.write_index(index, index_path)  # Save the index to a file
    print(f"FAISS index saved to {index_path}")

# Step 4: Retrieve Relevant Chunks Based on User Query
def retrieve_relevant_chunks(query, chunks, index_path="textbook_index.faiss", top_k=30):
    # Load FAISS index
    index = faiss.read_index(index_path)  # Read the pre-built FAISS index
    
    # Generate embedding for the query
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Same model as for chunks
    query_embedding = model.encode([query])  # Generate embedding for the query
    
    # Search the FAISS index for top_k most similar chunks
    distances, indices = index.search(np.array(query_embedding), top_k)  # Get top-k results
    
    # Get the relevant chunks based on the indices
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks, distances[0]


# Generate questions using T5 base finetuned question generation model
def get_question(tag, difficulty, context, answer="", num_questions=3, max_length=150, tokenizer = None, model = None):
    """
    Generate questions using the fine-tuned T5 model.
    
    Parameters:
    - tag: Type of question (e.g., "short answer", "multiple choice question", "true or false question")
    - difficulty: "easy", "medium", "hard"
    - context: Supporting context or passage
    - answer: Optional — if you want targeted question generation
    - num_questions: Number of diverse questions to generate
    - max_length: Max token length of generated output
    
    Returns:
    - List of generated questions as strings
    """
    # Format input text based on whether answer is provided
    answer_part = f"[{answer}]" if answer else ""
    input_text = f"<extra_id_97>{tag} <extra_id_98>{difficulty} <extra_id_99>{answer_part} {context}"

    # Tokenize
    features = tokenizer([input_text], return_tensors='pt')

    # Generate questions
    output = model.generate(
        input_ids=features['input_ids'],
        attention_mask=features['attention_mask'],
        max_length=max_length,

        # Beam search
        # num_beams = 5,
        # early_stopping=True,              # to stop when the first beam is finished

        # Sampling
        num_return_sequences=num_questions,
        do_sample=True,
        top_p=0.95,
        top_k=50
    )

    # Decode generated questions
    for i, out in enumerate(output):
        question = tokenizer.decode(out, skip_special_tokens=True)
        print(f"Question {i+1}: {question}")
    
    print("-------------------------------------------------------------------------------------------")
    






def main():

    # Compulsary
    # textbook_path = "textbook.pdf"                  # Path to your textbook
    
    while True:
        textbook_path = input("Enter path to the textbook: ").strip()
        if not textbook_path:
            textbook_path = "textbook.pdf"

        if textbook_path.lower() == "quit":
            print("Exiting program.")
            sys.exit()

        if os.path.isfile(textbook_path):
            break
        else:
            print("File not found. Please provide a valid path to the textbook or type 'quit' to exit.")

    # Get base name (without extension) for consistency check
    textbook_base = os.path.splitext(os.path.basename(textbook_path))[0]

    # Optional
    textbook_chunks_path = f"{textbook_base}_chunks.json"   # Path to your chunks JSON file
    index_path = f"{textbook_base}_index.faiss"             # Path to save the FAISS index

    # try: 
    #     # Step 1: Load the chunks
    #     chunks = load_chunks(textbook_chunks_path)
    #     print(f"Loaded {len(chunks)} chunks from {textbook_chunks_path}")
    # except:
    #     chunk_textbook(textbook_path, textbook_chunks_path)
    #     chunks = load_chunks(textbook_chunks_path)
    #     print(f"Loaded {len(chunks)} chunks from {textbook_chunks_path}")


    # Step 2: Generate embeddings and create the FAISS index (if not already created)
    try:
        chunks = load_chunks(textbook_chunks_path)
        print(f"Loaded {len(chunks)} chunks from {textbook_chunks_path}")
        faiss.read_index(index_path)  # Try to read the FAISS index to check if it exists
        print(f"FAISS index already exists at {index_path}. Skipping creation.")
    except:
        try:
            # Step 1: Load the chunks from json --> create embeddings --> store in a Fiass database
            chunks = load_chunks(textbook_chunks_path)
            print(f"Loaded {len(chunks)} chunks from {textbook_chunks_path}")
            embeddings = generate_embeddings(chunks)
            create_faiss_index(embeddings, index_path)
        except:
            # Step 0: Chunk the textbook --> Load the chunks from json --> create embeddings --> store in a Fiass database
            chunk_textbook(textbook_path, textbook_chunks_path)
            chunks = load_chunks(textbook_chunks_path)
            print(f"Loaded {len(chunks)} chunks from {textbook_chunks_path}")
            embeddings = generate_embeddings(chunks)
            create_faiss_index(embeddings, index_path)


    # Load fine-tuned T5 model for question generation
    # model_name = "./T5base_Question_Generation_v6"
    model_name = "Avinash250325/T5BaseQuestionGeneration"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)


    # Step 3: Continuous Query Input
    while True:
        # Accept a user query and Retrieve relevant chunks for the user query
        take_input = input("\nPlease hit the 'Enter' key to continue or type 'exit' to quit\n")
        # Exit condition
        if take_input == 'exit':
            print("Exiting program...")
            break

        tag = input("\nEnter your question type [short answer, long answer, true or false, multiple choice question]: ")
        difficulty = input("\nEnter your question difficulty [easy, medium, hard]: ")
        query = input("\nEnter your query: ")

        if tag == None:
            tag = "short answer question"
        if difficulty == None:
            difficulty = "medium"

        
        # Retrieve relevant chunks based on the query
        relevant_chunks, distances = retrieve_relevant_chunks(query, chunks, index_path, top_k = 20)

        # Filter the top 3 chunks with >20 words
        # filtered_chunks = [chunk for chunk in relevant_chunks if len(chunk.split()) > 10][:3]
        
        print("\nRetrieved Chunks from the textbook:")
        filtered_chunks = []
        for chunk in relevant_chunks:
            if len(chunk.split()) > 20:
                filtered_chunks.append(chunk)
            if len(filtered_chunks) == 3:
                break

        if not filtered_chunks:
            print("No sufficiently long chunks found. Try another query.")
            continue

        context = ""

        # Display all the first 5 relevant chunks to print the distance values of each chunk with the query
        # print("\nTop 5 Relevant Chunks:")
        # for i, (chunk, distance) in enumerate(zip(relevant_chunks[5], distances)):
        #     print(f"Rank {i + 1}: (Distance: {distance:.4f})")
        #     print(f"  {chunk}\n")

        print("\nTop 5 Relevant Chunks:")
        for i, chunk in enumerate(filtered_chunks):
            print(f"Rank {i + 1}: ")
            print(f"  {chunk}\n")
            context += chunk

        get_question(
            tag=tag,
            difficulty=difficulty,
            context=context,
            tokenizer = tokenizer,
            model = model
        )



if __name__ == "__main__":
    main()





