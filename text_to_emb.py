import gc

import numpy as np

from models.llm import SentenceEncoder
import ujson as json
from tqdm import tqdm

device = "cuda:0"

if __name__ == "__main__":
    model = SentenceEncoder("PubmedBERT", cache_dir="cache_data/model", max_lengths=512)

    with open("data/text/gene_text_description.json") as f:
        gene_desc_texts = json.load(f)

    with open("data/text/geneset_text_description.json") as f:
        geneset_desc_texts = json.load(f)


    gene_texts = []
    for gene_id, gene_desc_text in tqdm(gene_desc_texts.items()):
        gene_text = str(gene_id) + ": " + str(gene_desc_text)
        gene_texts.append(gene_text)

    geneset_texts = []
    for geneset, geneset_desc_text in tqdm(geneset_desc_texts.items()):
        geneset_text = str(geneset) + ": " + str(geneset_desc_text)
        geneset_texts.append(geneset_text)

    gene_text_embs = model.encode(gene_texts, to_tensor=False)
    print(gene_text_embs[0])
    # encode gene text
    np.save("data/text/gene_text_embs.npy", gene_text_embs)
    del gene_text_embs
    gc.collect()

    geneset_text_embs = model.encode(geneset_texts, to_tensor=False)
    print(geneset_text_embs[0])
    # encode geneset text
    np.save("data/text/geneset_text_embs.npy", geneset_text_embs)
