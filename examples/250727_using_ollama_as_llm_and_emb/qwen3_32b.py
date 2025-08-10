import os
import sys
import csv
import io
import json
from tqdm import tqdm
sys.path.append("..")
import logging
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

def main():

    os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    # Assumed llm model settings
    MODEL = "qwen3:32b"

    # Assumed embedding model settings
    EMBEDDING_MODEL = "nomic-embed-text"
    EMBEDDING_MODEL_DIM = 768
    EMBEDDING_MODEL_MAX_TOKENS = 8192


    async def ollama_model_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        # remove kwargs that are not supported by ollama
        kwargs.pop("max_tokens", None)
        kwargs.pop("response_format", None)
        import httpx
        ollama_client = ollama.AsyncClient(
        host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        timeout=httpx.Timeout(259200.0) 	
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        if hashing_kv is not None:
            args_hash = compute_args_hash(MODEL, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]
        # -----------------------------------------------------
        response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)

        result = response["message"]["content"]
        # Cache the response if having-------------------
        if hashing_kv is not None:
            await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
        # -----------------------------------------------------
        return result


    def remove_if_exist(file):
        if os.path.exists(file):
            os.remove(file)


    WORKING_DIR = "./250727_qwen3_32b"
    os.makedirs(WORKING_DIR, exist_ok=True)


    def query(question=None):
        rag = GraphRAG(
            working_dir=WORKING_DIR,
            best_model_func=ollama_model_if_cache,
            cheap_model_func=ollama_model_if_cache,
            embedding_func=ollama_embedding,
        )
        print(
            rag.query(
                question, param=QueryParam(mode="global")
            )
        )


    def insert():
        from time import time

        #with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        # FAKE_TEXT = f.read()
        texts = []
        doc_dir = "/data/hallucination/250726_muyu/250727_nano_graphrag/nano-graphrag/documents"
        files = list(os.scandir(doc_dir))
        for entry in tqdm(files, desc="Reading documents"):
            if entry.is_file() and entry.name.endswith('.txt'):  # Add file type check
                with open(entry.path, encoding="utf-8-sig") as f:
                    texts.append(f.read())
    
        FAKE_TEXT = "\n\n".join(texts)  # Join with double newlines

        remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
        remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

        rag = GraphRAG(
            working_dir=WORKING_DIR,
            enable_llm_cache=True,
            best_model_func=ollama_model_if_cache,
            cheap_model_func=ollama_model_if_cache,
            embedding_func=ollama_embedding,
        )
        start = time()
        rag.insert(FAKE_TEXT)
        print("indexing time:", time() - start)
        # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
        # rag.insert(FAKE_TEXT[half_len:])


    # We're using Ollama to generate embeddings for the BGE model
    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBEDDING_MODEL_DIM,
        max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
    )
    # async def ollama_embedding(texts: list[str]) -> np.ndarray:
    #     embed_text = []
    #     for text in texts:
    #         data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text, host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),)
    #         embed_text.append(data["embedding"])

    #     return embed_text

    async def ollama_embedding(texts: list[str]) -> np.ndarray:
        import httpx
        ollama_client = ollama.AsyncClient(
            host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout=httpx.Timeout(259200.0)
        )
        embed_text = []
        for text in texts:
            data = await ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)
            embed_text.append(data["embedding"])
        return embed_text

    insert()

    #main query
    qa_pairs = []
    with open ("/data/hallucination/250726_muyu/250727_nano_graphrag/nano-graphrag/truth_set_v1_new.csv", "r", encoding='utf-8-sig') as f:
        read = csv.DictReader(f)
        for row in read:
            qa_pairs.append({
                'question': row['Question'].strip(),
                'answer': row['Answer'].strip().strip('"'),
            })

    for qa in qa_pairs:
        response = query(qa['question'] + " Answer concisely.")
        qa['respond'] = str(response).strip()
   
    with io.open (f"/data/hallucination/250726_muyu/250727_nano_graphrag/nano-graphrag/examples/250727_using_ollama_as_llm_and_emb/qwen3_32b.json", "w") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":

    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", current_time)

    main()

    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", current_time)

