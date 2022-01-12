"""
using compare_mt https://github.com/neulab/compare-mt for ROUGE
"""

from summ_func import ext_oracle, lead_k
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize
from typing import List
import numpy as np

scorer = RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
_scorer = RougeScorer(['rouge1'], use_stemmer=True)

def compute_rouge(cand, ref):
    ref = sent_tokenize(ref)
    cand = sent_tokenize(cand)
    score = scorer.score("\n".join(ref), "\n".join(cand))
    rouge1 = score["rouge1"].fmeasure
    rouge2 = score["rouge2"].fmeasure
    return 2 * rouge1 * rouge2 / (rouge1 + rouge2 + 1e-20)

def _compute_rouge(cand, ref):
    ref = sent_tokenize(ref)
    cand = sent_tokenize(cand)
    score = scorer.score("\n".join(ref), "\n".join(cand))
    return score["rouge1"].fmeasure



def _ext_oracle(src: List[str], ref: str, sim_fn, max_sent: int = 3, max_len: int = -1, threshold: int = -1):
    """
    A functionality of generating the extractive oracle for a sample in the summarization dataset
    src: source documents
    ref: reference summaries
    sim_fn: sim_fn: similarity function between two strings
    max_sent: maximum number of oracle sentences
    max_len: maximum length of the oracle summaries
    threshold: a predefined threshold for stoping creteria
    """
    src = [x.strip() for x in src]
    ref = ref.strip()
    if len(src) == 0:
        src = ["#"]
    if len(ref) == 0:
        ref = "#"
    labels = [0] * len(src)
    # add the first sentence
    scores = [sim_fn(x, ref) for x in src]
    max_id = np.argmax(scores)
    # updating
    max_score = scores[max_id]
    oracle = [src[max_id]]
    labels[max_id] = 1
    # iterative search
    max_sent = len(src) if max_sent < 0 else min(max_sent, len(src))
    threshold = 0 if threshold < 0 else threshold
    cands = [(x, i) for (i, x) in enumerate(src) if i != max_id]
    while len(oracle) < max_sent:
        cur_oracle = " ".join(oracle)
        if max_len > 0 and len(word_tokenize(cur_oracle)) > max_len:
            break
        scores = [sim_fn(" ".join([cur_oracle, x[0]]), ref) for x in cands]
        max_id = np.argmax(scores)
        if scores[max_id] - max_score < threshold:
            break
        max_score = scores[max_id]
        oracle.append(cands[max_id][0])
        labels[cands[max_id][1]] = 1
        del cands[max_id]
    return {"source":src,
            "reference":ref,
            "oracle_summary":oracle,
            "oracle_labels":labels,
            "oracle_score":max_score}


def _lead_k(src: List[str], ref: str, sim_fn, k: int = 3):
    """
    A functionality of generating summaries using lead-k sentences
    src: source documents
    ref: reference summaries
    k: the number of leading sentences to use as summaries
    sim_fn: sim_fn: similarity function between two strings
    """
    src = [x.strip() for x in src]
    ref = ref.strip()
    if len(src) == 0:
        src = ["#"]
    if len(ref) == 0:
        ref = "#"
    src = src[:k]
    score = sim_fn(" ".join(src), ref)

    return {"source":src,
            "reference":ref,
            "lead_k_summary":src,
            "lead_k_score":score}

def demo():


    """
    from datalab import load_dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    documents = dataset["test"]['article'][0:3]
    summaries = dataset["test"]['highlights'][0:3]

    documents = [sent_tokenize(x.strip()) for x in documents]
    print(len(documents))
    res = ext_oracle(documents, summaries, _compute_rouge, max_sent=3, num_workers=8)
    print(res)
    """

    document = "I love this movie. He is a man. this is a good man. tomorrow is a nice day."
    document = sent_tokenize(document)
    # >>> ["I love this movie", "He is a man", "this is a good man", "tomorrow is a nice day"]
    summary = "He likes this movie"
    res = _ext_oracle(document, summary, _compute_rouge, max_sent=3)
    print(res)

    res = _lead_k(document, summary, _compute_rouge, k = 3)
    print(res)


    # lead_k(documents, summaries, 3, _compute_rouge, num_workers=8)
    # ext_oracle(src, tgt, _compute_rouge, max_sent=3, num_workers=8)
    # lead_k(src, tgt, 3, _compute_rouge, num_workers=8)


if __name__ == "__main__":
    demo()