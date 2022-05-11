"""
dependency: nltk
main functions: ext_oracle(), lead_k()
"""

from functools import partial
import json
from multiprocessing import Pool
from typing import List

from nltk import word_tokenize
import numpy as np


def _ext_oracle(
    src: List[str],
    ref: str,
    sim_fn,
    max_sent: int = -1,
    max_len: int = -1,
    threshold: int = -1,
):
    """
    A functionality of generating the extractive oracle for a
    sample in the summarization dataset
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
    return " ".join(oracle), labels, max_score


def thread_wrapper(x, fn):
    return fn(*x)


def ext_oracle(
    src: List[List[str]],
    ref: List[str],
    sim_fn,
    max_sent: int = -1,
    max_len: int = -1,
    threshold: int = -1,
    num_workers: int = 4,
    out_dir: str = None,
):
    """
    A functionality of generating the extractive oracle for a summarization dataset
    src: source documents, each sample should be a list of sentences (strings)
    ref: reference summaries
    sim_fn: similarity function between two strings
    max_sent: maximum number of sentences in the oracle summaries, optional
    max_len: maximum length of the oracle summaries, optional
    threshold: a predefined threshold for stoping creteria, optional
    num_workers: number of threads for computing
    out_dir: write the results to a file instead of returning
    them if the path is provided

    returns: oracle summaries (list of str)
    labels: one hot labels
    scores: scores of each oracle summary
    score: the average score
    """

    labels = []
    oracles = []
    scores = []
    score = 0
    cnt = 0
    if out_dir is not None:
        f = open(out_dir, "w")
    if num_workers > 1:
        fn = partial(
            _ext_oracle,
            sim_fn=sim_fn,
            max_sent=max_sent,
            max_len=max_len,
            threshold=threshold,
        )
        fn = partial(thread_wrapper, fn=fn)
        with Pool(processes=num_workers) as pool:
            result = pool.imap(fn, zip(src, ref), chunksize=64)
            for (o, l, s) in result:
                if out_dir is not None:
                    print(json.dumps({"label": l, "oracle": o, "score": s}), file=f)
                else:
                    oracles.append(o)
                    labels.append(l)
                    scores.append(s)
                score += s
                cnt += 1
    else:
        for (x, y) in zip(src, ref):
            o, l, s = _ext_oracle(x, y, sim_fn, max_sent, max_len, threshold)
            if out_dir is not None:
                print(json.dumps({"label": l, "oracle": o, "score": s}), file=f)
            else:
                oracles.append(o)
                labels.append(l)
                scores.append(s)
            score += s
            cnt += 1
    score = score / cnt
    if out_dir is not None:
        f.close()
    print(f"extractive oracle score: {score:.7f}")
    return oracles, labels, scores, score


def lead_k(
    src: List[List[str]],
    ref: List[str],
    k: int,
    sim_fn,
    num_workers: int = 4,
    out_dir: str = None,
):
    """
    A functionality of generating summaries using lead-k sentences
    src: source documents, each sample should be a list of sentences (strings)
    ref: reference summaries
    k: the number of leading sentences to use as summaries
    sim_fn: similarity function between two strings
    num_workers: number of threads for computing
    out_dir: write the results to a file instead of returning them
     if the path is provided

    returns: lead-k summaries (list of str)
    scores: scores of each lead-k summary
    score: the average score
    """
    summaries = []
    scores = []
    score = 0
    cnt = 0
    if out_dir is not None:
        f = open(out_dir, "w")
    if num_workers > 1:
        fn = partial(_lead_k, sim_fn=sim_fn, k=k)
        fn = partial(thread_wrapper, fn=fn)
        with Pool(processes=num_workers) as pool:
            result = pool.imap(fn, zip(src, ref), chunksize=64)
            for (x, s) in result:
                if out_dir is not None:
                    print(json.dumps({"k-lead": x, "score": s}), file=f)
                else:
                    summaries.append(x)
                    scores.append(s)
                score += s
                cnt += 1
    else:
        for (x, y) in zip(src, ref):
            x, s = _lead_k(x, y, k, sim_fn)
            if out_dir is not None:
                print(json.dumps({"k-lead": x, "score": s}), file=f)
            else:
                summaries.append(x)
                scores.append(s)
            score += s
            cnt += 1
    score = score / cnt
    if out_dir is not None:
        f.close()
    print(f"lead-k score: {score:.7f}")
    return summaries, scores


def _lead_k(src: List[str], ref: str, k: int, sim_fn):
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
    return " ".join(src), score
