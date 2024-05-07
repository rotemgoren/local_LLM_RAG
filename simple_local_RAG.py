# Requires !pip install PyMuPDF, see: https://github.com/pymupdf/pymupdf
import fitz # (pymupdf, found this is better than pypdf for our use case, note: licence is AGPL-3.0, keep that in mind if you want to use any code commercially)
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm
from spacy.lang.en import English # see https://spacy.io/usage for install instructions
import random
import pandas as pd
import re
from sentence_transformers import SentenceTransformer,util
import torch
import numpy as np
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from key_param import access_token
import argparse

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(pdf_path: str,num_sentence_chunk_size: int) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)

        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]

        # Count the sentences
        item["page_sentence_count_spacy"] = len(item["sentences"])

        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                             slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                           joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)
    return pages_and_texts,pages_and_chunks



def createEmbedding(sentence_chunks):
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                          device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)


    # Create embeddings one by one on the GPU
    for item in tqdm(sentence_chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Turn text chunks into a single list
    text_chunks = [item["sentence_chunk"] for item in tqdm(sentence_chunks)]
    # Embed all texts in batches
    text_chunk_embeddings = embedding_model.encode(text_chunks,
                                                   batch_size=32, # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
                                                   convert_to_tensor=True) # optional to return embeddings as tensor instead of array

    text_chunks_and_embeddings_df = pd.DataFrame(sentence_chunks)
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

def printWrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


def loadingEmeddingDataBase(device):
    # Import texts and embedding df
    text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

    # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Convert texts and embedding df to list of dicts
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
    print(embeddings.shape)
    return embeddings,pages_and_chunks

def retrieveRelevantResources(query: str,embeddings: torch.tensor,pages_and_chunks: list[dict],
                              embedding_model: SentenceTransformer ,n_resources_to_return: int=5,print_time: bool=True):
    # 2.Turn the query string into an embedding.
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)

    # 3.Perform a dot product or cosine similarity function between the text embeddings and the query embedding.
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]

    # 4.Sort the results from 3 in descending order.
    top_results_dot_product = torch.topk(dot_scores, k=5)

    scores  = top_results_dot_product[0]
    indexs = top_results_dot_product[1]
    print(top_results_dot_product)
    print(query)
    print("Results:")
    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indexs):
        print(f"Score: {score:.4f}")
        print("Text:")
        printWrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")

    return scores,indexs
    # checkout for an open-source reranking model: https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1

def promptFormatter(query: str,
                     context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    Context items:
    {context}
    Query: {query}
    Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt



def ask(query,tokenizer,
        llm_model,embeddings,
        embedding_model,pages_and_chunks,
        temperature=0.7,max_new_tokens=512,
        format_answer_text=True,return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

    # Get just the scores and indices of top related results

    scores, indices = retrieveRelevantResources(query=query,
                                                embeddings=embeddings,
                                                pages_and_chunks=pages_and_chunks,
                                                embedding_model=embedding_model,
                                                n_resources_to_return=5,
                                                print_time=True)

                              # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()  # return score back to CPU

    # Format the prompt with context items
    prompt = promptFormatter(query=query,
                              context_items=context_items)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace(
            "Sure, here is the answer to the user query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text,_

    return output_text, context_items


def makeDataBaseEmbeddingSpace(pdf_path:str , num_sentence_chunk_size : int = 10 , min_token_length: int = 30):
    pages_and_texts,pages_and_chunks = open_and_read_pdf(pdf_path=pdf_path,num_sentence_chunk_size=num_sentence_chunk_size)

    print(len(pages_and_chunks))
    df = pd.DataFrame(pages_and_texts)
    print(df.describe().round(2))

    df = pd.DataFrame(pages_and_chunks)
    print(df.describe().round(2))

    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    createEmbedding(pages_and_chunks_over_min_token_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--pdf_path', type=str, default='C:/Users/STL-LAPTOP/Downloads/Human-Nutrition-2020-Edition-1598491699.pdf',
                        help='path to the pdf file')

    parser.add_argument('--make_db_embedding', type=bool,
                        default=True,
                        help='making Embedding space for the local data base')

    args = parser.parse_args()

    # 1. making Embedding space for the local data base
    if args.make_db_embedding:
        makeDataBaseEmbeddingSpace(args.pdf_path)


    # load the embedding database
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    embeddings,pages_and_chunks = loadingEmeddingDataBase(device)


    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                          device=device)


    # 1.Define a query string.
    query = input("Enter your question:") #e.g What are the macronutrients, and what roles do they play in the human body?"
    # scores,indexs = retrieveRelevantResources(query,embeddings,pages_and_chunks,
    #                          embedding_model ,n_resources_to_return=5,print_time=True)




    login(access_token)
    # 1. Create quantization config
    use_quantization_config=True
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.float16)

    # 2. pick a model
    model_id = 'google/gemma-7b-it'
    #3. instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id,token=access_token)

    #4. instantiate the model
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                     torch_dtype=torch.float16,
                                                     quantization_config=quantization_config if use_quantization_config else None,
                                                     low_cpu_mem_usage=False,
                                                     token=access_token
                                                     )

    # Answer query with context and return context
    answer, context_items = ask(query=query,
                                tokenizer=tokenizer,
                                llm_model=llm_model,
                                embeddings=embeddings,
                                embedding_model=embedding_model,
                                pages_and_chunks=pages_and_chunks,
                                temperature=0.7,
                                max_new_tokens=512,
                                return_answer_only=False)


    print('Question:\n')
    printWrapped(query)
    print(f"Answer:\n")
    printWrapped(answer)
    #print(f"Context items:\n")
    #print(context_items)
