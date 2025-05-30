from dotenv import load_dotenv
import tiktoken
import pandas as pd
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

load_dotenv()


# Context window length for OpenAI
MAX_TOKENS = 8191

# Doc folder and files
OUT_DIR = Path("./extracted_docs")
JSONS = sorted(OUT_DIR.glob("*.json"))

print("Found", len(JSONS), "Docling JSON files")


# Setup OpenAI Tokenizer and Chunker
tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model("gpt-4o"), max_tokens=MAX_TOKENS
)

chunker = HybridChunker(tokenizer=tokenizer)


def process_doc(doc_path):
    doc = DoclingDocument.load_from_json(doc_path)

    chunks = chunker.chunk(dl_doc=doc)

    types = []

    for i, chunk in enumerate(chunks):
        print(f"=== {i} ===")
        txt_tokens = tokenizer.count_tokens(chunk.text)
        print(f"chunk.text ({txt_tokens} tokens):\n{chunk.text!r}")

        ser_txt = chunker.contextualize(chunk=chunk)
        ser_tokens = tokenizer.count_tokens(ser_txt)
        print(f"chunker.contextualize(chunk) ({ser_tokens} tokens):\n{ser_txt!r}")

        meta_json = chunk.meta.export_json_dict()

        print()
        print(meta_json)

        print()

    # rows = []

    # for chunk in chunker.chunk(doc):
    #     row = {
    #         "text": chunk.text,
    #         "context": chunker.contextualize(chunk=chunk),
    #         "meta": chunk.meta.export_json_dict(),
    #     }
    #     rows.append(row)

    # df = pd.DataFrame(rows, columns=["text", "context", "meta"])

    # return df


print(process_doc(JSONS[1]))
