import numpy as np
import os
import logging
import ollama
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from pathlib import Path
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from datetime import datetime
import csv
import sys
import io
import json
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# !!! qwen2-7B maybe produce unparsable results and cause the extraction of graph to fail.
small_model = "granite3.3:8b"
big_model = "llama3.3:70b"
WORKING_DIR = "/data/hallucination/250726_muyu/250727_nano_graphrag/250808_granite_llama"

EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)

def _translate_openai_kwargs_for_ollama(kwargs: dict) -> dict:
    rf = kwargs.pop("response_format", None)
    if rf:
        # If JSON requested, tell Ollama to return JSON
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            kwargs["format"] = "json"
        elif isinstance(rf, str) and rf.lower() == "json":
            kwargs["format"] = "json"
        # If it's something else, just ignore or log

    # Map max_tokens to Ollama
    max_tokens = kwargs.pop("max_tokens", None)
    if max_tokens is not None:
        options = kwargs.get("options", {})
        options["num_predict"] = max_tokens
        kwargs["options"] = options

    return kwargs


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


async def small_ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(small_model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=small_model, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": small_model}})
    # -----------------------------------------------------
    return result

async def big_ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    # kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None)

    if history_messages is None:
        history_messages = []

    # DEBUG: check incoming kwargs from nano-graphrag
    print("\n[DEBUG] big_ollama_model_if_cache kwargs:", kwargs, "\n")

    # Convert only if JSON requested
    kwargs = _translate_openai_kwargs_for_ollama(kwargs)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(big_model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=big_model, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": big_model}})
    # -----------------------------------------------------
    return result


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def query(question=None):
    rag = GraphRAG(
        working_dir=WORKING_DIR,
		enable_llm_cache=False,
        best_model_func=big_ollama_model_if_cache,
        cheap_model_func=small_ollama_model_if_cache,
        embedding_func=local_embedding,
    )
    return rag.query(
        question, param=QueryParam(mode="global")
        
    )


def insert():
    from time import time

    folder = Path('/data/hallucination/250726_muyu/250727_nano_graphrag/nano-graphrag/documents')
    all_contents = [p.read_text(encoding='utf-8') for p in folder.glob('*.txt')]
   

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=False,
        best_model_func=big_ollama_model_if_cache,
        cheap_model_func=small_ollama_model_if_cache,
        embedding_func=local_embedding,
    )
    start = time()
    rag.insert(all_contents)
    print("indexing time:", time() - start)


if __name__ == "__main__":

    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", current_time)

    insert()

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
        qa['respond'] = str(response['response']).strip()
        qa['community_summary'] = str(response['points_context']).strip()

    with io.open (f"/data/hallucination/250726_muyu/250727_nano_graphrag/nano-graphrag/examples/250727_using_ollama_as_llm_and_emb/granite_llama.json", "w") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", current_time)
