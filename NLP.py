"""
ctd_relation_nli.py

Input:  input_ctd.csv  (columns: chemical,disease,pmid)
Output: ctd_relation_results.csv

Uses an NLI model (tasksource/deberta-small-long-nli) to score whether
a sentence/abstract entails the hypothesis "Exposure to {chemical} causes {disease}."
Selects the sentence with the highest entailment probability as supporting evidence.

Install required packages:
pip install biopython pandas tqdm nltk transformers torch rapidfuzz
"""

import os
import time
from pathlib import Path
import logging
import pandas as pd
from tqdm.auto import tqdm
import nltk
nltk.download("punkt", quiet=True)
from Bio import Entrez
from rapidfuzz import fuzz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional

# ---------- USER CONFIG ----------
INPUT_CSV = "input_ctd.csv"
OUTPUT_CSV = "ctd_relation_results.csv"

Entrez.email = "graham.gould@mail.mcgill.ca"  # your email
Entrez.api_key = None  # optional: set your NCBI API key here

NLI_MODEL = "tasksource/deberta-small-long-nli"

HYPOTHESIS_TEMPLATES = [
    "Exposure to {chemical} causes {disease}.",
    "{chemical} exposure increases the risk of {disease}.",
    "{chemical} is associated with {disease}.",
    "{chemical} is linked to {disease}.",
    "{chemical} exposure is related to {disease}."
]

FUZZY_THRESHOLD_SENTENCE = 75
FUZZY_THRESHOLD_WINDOW = 70
MAX_WINDOW_SENTENCES = 3

CACHE_DIR = Path("pmid_cache")
CACHE_DIR.mkdir(exist_ok=True)
SLEEP_BETWEEN_REQUESTS = 0.34

DEVICE = 0 if torch.cuda.is_available() else -1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------- Utilities ----------
import re

def fetch_abstract(pmid: str, use_cache=True, sleep_between=SLEEP_BETWEEN_REQUESTS) -> str:
    pmid = str(pmid).strip()
    if not pmid:
        return ""
    cache_file = CACHE_DIR / f"{pmid}.txt"
    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        text = handle.read()
        handle.close()
        text = re.sub(r"\s+", " ", text).strip()
        if use_cache:
            cache_file.write_text(text, encoding="utf-8")
        time.sleep(sleep_between)
        return text
    except Exception as e:
        logging.warning(f"Failed to fetch PMID {pmid}: {e}")
        return ""

def simple_entity_match(sentence: str, target: str, threshold: int = FUZZY_THRESHOLD_SENTENCE) -> bool:
    if not sentence or not target:
        return False
    s = sentence.lower()
    t = target.lower()
    if t in s:
        return True
    score = fuzz.partial_ratio(t, s)
    return score >= threshold

def find_candidate_sentences(abstract_text: str, chemical: str, disease: str) -> List[str]:
    sentences = nltk.sent_tokenize(abstract_text)
    candidates = []

    for sent in sentences:
        # include any sentence mentioning chemical or disease
        if simple_entity_match(sent, chemical) or simple_entity_match(sent, disease):
            candidates.append(sent)

    # fallback: if no sentences mention either, include all sentences
    if not candidates:
        candidates = sentences

    return candidates


# ---------- NLI Scorer ----------
class NLIScorer:
    def __init__(self, model_name: str, device: int = DEVICE):
        logging.info(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device != -1 and torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        id2label = {int(k): v.lower() for k, v in getattr(self.model.config, "id2label", {}).items()}
        self.entail_label_idx = next((idx for idx, lab in id2label.items() if "entail" in lab), 2)
        logging.info(f"Entailment label index set to {self.entail_label_idx}")

    def score(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=4096)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        if self.entail_label_idx < len(probs):
            return float(probs[self.entail_label_idx])
        return float(probs.max())

# ---------- PMIDs helper ----------
def re_split_pmids(pmid_field: str) -> List[str]:
    if pd.isna(pmid_field):
        return []
    s = str(pmid_field)
    parts = []
    for sep in [";", ",", "|"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            break
    if not parts:
        parts = s.split()
    normalized = []
    for p in parts:
        token = "".join(ch for ch in p.strip() if ch.isdigit())
        if token:
            normalized.append(token)
    return normalized

# ---------- Main pipeline ----------
def process_row_nli(scorer: NLIScorer, chemical: str, disease: str, pmid_field: str):
    """
    Process a single row: select best_sentence and compute global abstract confidence.
    Returns dict with:
        - pmid
        - chemical
        - disease
        - best_sentence
        - best_sentence_confidence
        - global_confidence
        - note
    """
    pmids = [p.strip() for p in re_split_pmids(pmid_field) if p.strip()]
    if not pmids:
        return {
            "pmid": None,
            "chemical": chemical,
            "disease": disease,
            "best_sentence": None,
            "best_sentence_confidence": 0.0,
            "global_confidence": 0.0,
            "note": "no_pmid",
        }

    results = []

    for pmid in pmids:
        abstract_text = fetch_abstract(pmid)
        if not abstract_text:
            continue

        # Candidate sentences
        candidates = find_candidate_sentences(abstract_text, chemical, disease)
        if not candidates:
            # fallback: split abstract into sentences if none matched
            candidates = nltk.sent_tokenize(abstract_text)

        # Prepare hypotheses
        hypotheses = [tmpl.format(chemical=chemical, disease=disease) for tmpl in HYPOTHESIS_TEMPLATES]

        # --- Best sentence scoring ---
        best_sentence = None
        best_sentence_conf = -1.0
        for sent in candidates:
            sent_score = max([scorer.score(sent, hyp) for hyp in hypotheses])
            if sent_score > best_sentence_conf:
                best_sentence_conf = sent_score
                best_sentence = sent

        # --- Global confidence for full abstract ---
        global_conf = max([scorer.score(abstract_text, hyp) for hyp in hypotheses])

        results.append({
            "pmid": pmid,
            "chemical": chemical,
            "disease": disease,
            "best_sentence": best_sentence,
            "best_sentence_confidence": float(best_sentence_conf),
            "global_confidence": float(global_conf),
            "note": "ok",
        })

    # Return the row with highest global confidence
    if results:
        best_result = max(results, key=lambda x: x["global_confidence"])
        return best_result
    else:
        return {
            "pmid": None,
            "chemical": chemical,
            "disease": disease,
            "best_sentence": None,
            "best_sentence_confidence": 0.0,
            "global_confidence": 0.0,
            "note": "no_evidence",
        }



def run_pipeline(input_csv: str = INPUT_CSV, output_csv: str = OUTPUT_CSV, max_rows: Optional[int] = None):
    df = pd.read_csv(input_csv, dtype=str)
    # standardize column names
    cols_lower = {c.lower(): c for c in df.columns}
    required = {"chemical", "disease", "pmid"}
    if not required.issubset(set(cols_lower.keys())):
        raise ValueError(f"Input CSV must contain columns: {required}")
    df = df.rename(columns={cols_lower[c]: c for c in cols_lower})
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if max_rows:
        df = df.iloc[:max_rows].copy()

    scorer = NLIScorer(NLI_MODEL)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        chem = row["chemical"]
        dis = row["disease"]
        pmid_field = row["pmid"]
        try:
            res = process_row_nli(scorer, chem, dis, pmid_field)
        except Exception as e:
            logging.exception("Error processing row with pmid(s) %s: %s", pmid_field, e)
            res = {
                "pmid": None,
                "chemical": chem,
                "disease": dis,
                "best_sentence": None,
                "best_sentence_confidence": 0.0,
                "global_confidence": 0.0,
                "note": f"error: {e}",
            }
        results.append(res)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    logging.info("Wrote results to %s", output_csv)
    return out_df



# ---------- Entrypoint ----------
if __name__ == "__main__":
    run_pipeline()