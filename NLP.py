import os, time
from pathlib import Path
import json
import pandas as pd
from tqdm.auto import tqdm
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from Bio import Entrez
from rapidfuzz import fuzz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ----------------------
# === USER CONFIG ===
# ----------------------
INPUT_CSV = 'input_ctd.csv'  # upload your CSV with columns: chemical,disease,pmid
OUTPUT_CSV = 'ctd_relation_results.csv'

Entrez.email = 'graham.gould@mail.mcgill.ca'
Entrez.api_key = None  # Optional: set your NCBI API key if you have one

BIOBERT_MODEL = 'tasksource/deberta-small-long-nli'
POSITIVE_LABEL_INDEX = 1

# ----------------------
# simple cache for PMIDs
cache_dir = Path('pmid_cache')
cache_dir.mkdir(exist_ok=True)

def fetch_abstract(pmid, use_cache=True, sleep_between=0.34):
    pmid = str(pmid)
    cache_file = cache_dir / f"{pmid}.txt"
    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding='utf-8')
    try:
        handle = Entrez.efetch(db='pubmed', id=pmid, rettype='abstract', retmode='text')
        text = handle.read()
        handle.close()
        if use_cache:
            cache_file.write_text(text, encoding='utf-8')
        time.sleep(sleep_between)
        return text
    except Exception as e:
        print(f"Error fetching PMID {pmid}: {e}")
        return ''

# fuzzy-based span finder and entity marker inserter
def find_best_span(sentence, entity, min_ratio=70):
    """Find the best substring span in 'sentence' matching 'entity' using fuzzy partial ratio.
    Returns (start_idx, end_idx, matched_text) or None if not found."""
    s = sentence
    t = entity.lower()
    # direct containment
    low = s.lower()
    idx = low.find(t)
    if idx != -1:
        return idx, idx+len(t), s[idx:idx+len(t)]
    # sliding window fuzzy: check substrings up to length of sentence
    best = (None, None, None, 0)
    words = s.split()
    n = len(words)
    # try windows of up to 6 words for speed
    max_window = min(6, n)
    for w in range(1, max_window+1):
        for i in range(0, n-w+1):
            span = ' '.join(words[i:i+w])
            score = fuzz.partial_ratio(t, span.lower())
            if score > best[3]:
                # compute character indices
                # find span occurrence in sentence (first occurrence of that sequence)
                start_char = s.find(span)
                if start_char >= 0:
                    best = (start_char, start_char+len(span), span, score)
    if best[3] >= min_ratio:
        return best[0], best[1], best[2]
    return None


def insert_entity_markers(sentence, chemical, disease):
    """Insert <e1>...</e1> around chemical and <e2>...</e2> around disease.
    Returns marked_sentence. If spans overlap or can't be found, returns None.
    """
    s = sentence
    c_span = find_best_span(s, chemical, min_ratio=65)
    d_span = find_best_span(s, disease, min_ratio=65)
    if not c_span or not d_span:
        return None
    c_start, c_end, c_text = c_span
    d_start, d_end, d_text = d_span
    # if spans overlap, choose the one with higher fuzz score or abort
    if not (c_end <= d_start or d_end <= c_start):
        # overlapping: try to break ties by checking exact containment
        if c_text.lower() == s[c_start:c_end].lower() and d_text.lower() == s[d_start:d_end].lower():
            # proceed but ensure order: e1 must be chemical, we'll place whichever comes first
            pass
        else:
            return None
    # ensure we insert tags in reverse order of indices to preserve indices
    parts = []
    if c_start < d_start:
        # chemical first
        before = s[:c_start]
        chem = s[c_start:c_end]
        middle = s[c_end:d_start]
        dise = s[d_start:d_end]
        after = s[d_end:]
        marked = f"{before}<e1>{chem}</e1>{middle}<e2>{dise}</e2>{after}"
    else:
        before = s[:d_start]
        dise = s[d_start:d_end]
        middle = s[d_end:c_start]
        chem = s[c_start:c_end]
        after = s[c_end:]
        marked = f"{before}<e2>{dise}</e2>{middle}<e1>{chem}</e1>{after}"
    return marked

# loading the BIOBERT model, else fallback to zero-shot NLI
use_re_model = True
try:
    print('Loading RE model:', BIOBERT_MODEL)
    re_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    re_model = AutoModelForSequenceClassification.from_pretrained(BIOBERT_MODEL)
    re_model.eval()
    print('Loaded RE model successfully.')
except Exception as e:
    print('Failed to load RE model:', e)
    print('Falling back to zero-shot NLI (facebook/bart-large-mnli)')
    use_re_model = False
    nli_model_name = 'facebook/bart-large-mnli'
    classifier = pipeline('zero-shot-classification', model=nli_model_name, device=0 if torch.cuda.is_available() else -1)

import nltk
from rapidfuzz import fuzz

def simple_entity_match(sentence, target, threshold=80):
    s = sentence.lower()
    t = target.lower()
    if t in s:
        return True
    score = fuzz.partial_ratio(t, s)
    return score >= threshold

def score_sentence_with_re_model(marked_sentence):
    """Run the RE model on a sentence with inserted markers. Return probability of relation (index POSITIVE_LABEL_INDEX)."""
    inputs = re_tokenizer(marked_sentence, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = re_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return float(probs[POSITIVE_LABEL_INDEX])

def score_sentence_with_nli(sentence, chemical, disease):
    labels = [f"{chemical} associated with {disease}", f"{chemical} causes {disease}", "no relation"]
    out = classifier(sentence, labels)
    scores = dict(zip(out['labels'], out['scores']))
    assoc = scores.get(labels[0], 0.0)
    cause = scores.get(labels[1], 0.0)
    return float(max(assoc, cause))

def process_row(chemical, disease, pmid):
    abstract_text = fetch_abstract(pmid)
    if not abstract_text:
        return {'pmid': pmid, 'chemical': chemical, 'disease': disease, 'best_sentence': None, 'confidence': 0.0, 'note': 'no_abstract'}
    sentences = nltk.tokenize.sent_tokenize(abstract_text)
    candidates = []
    for sent in sentences:
        if simple_entity_match(sent, chemical, threshold=75) and simple_entity_match(sent, disease, threshold=75):
            candidates.append(sent)
    # if no direct sentence found, try windowed join
    if not candidates:
        for i, sent in enumerate(sentences):
            if simple_entity_match(sent, chemical, threshold=85):
                window = sentences[max(0, i-2):min(len(sentences), i+3)]
                joined = ' '.join(window)
                if simple_entity_match(joined, disease, threshold=75):
                    candidates.append(joined)
    if not candidates:
        return {'pmid': pmid, 'chemical': chemical, 'disease': disease, 'best_sentence': None, 'confidence': 0.0, 'note': 'no_candidate_sentence'}

    best_sent = None
    best_score = -1.0
    used_method = 're_model' if use_re_model else 'nli'
    for sent in candidates:
        if use_re_model:
            marked = insert_entity_markers(sent, chemical, disease)
            if not marked:
                # try swapping roles if insertion failed
                marked = insert_entity_markers(sent, disease, chemical)
                # if swapped, the model expects e1=chemical, so it's wrong — prefer fallback
            if not marked:
                # fallback to NLI scoring for this sentence
                score = score_sentence_with_nli(sent, chemical, disease)
                method = 'nli'
            else:
                try:
                    score = score_sentence_with_re_model(marked)
                    method = 're_model'
                except Exception as e:
                    print('RE model scoring error:', e)
                    score = score_sentence_with_nli(sent, chemical, disease)
                    method = 'nli'
        else:
            score = score_sentence_with_nli(sent, chemical, disease)
            method = 'nli'
        if score > best_score:
            best_score = score
            best_sent = sent
            used_method = method
    return {'pmid': pmid, 'chemical': chemical, 'disease': disease, 'best_sentence': best_sent, 'confidence': float(best_score), 'method': used_method, 'note': 'ok'}

def run_pipeline(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV, max_rows=None):
    df = pd.read_csv(input_csv, dtype=str)
    required = {'chemical', 'disease', 'pmid'}
    cols_lower = {c.lower(): c for c in df.columns}
    if not required.issubset(set(cols_lower.keys())):
        raise ValueError(f"Input CSV must contain columns: {required}")
    # rename
    df = df.rename(columns={cols_lower[c]: c for c in cols_lower})
    df = df.rename(columns={c: c.lower() for c in df.columns})
    results = []
    rows = df.to_dict('records')
    if max_rows:
        rows = rows[:max_rows]
    for row in tqdm(rows, desc='Processing rows'):
        chem = row.get('chemical')
        dis = row.get('disease')
        pmid = row.get('pmid')
        try:
            res = process_row(chem, dis, pmid)
        except Exception as e:
            print('Error processing', pmid, e)
            res = {'pmid': pmid, 'chemical': chem, 'disease': dis, 'best_sentence': None, 'confidence': 0.0, 'note': f'error: {e}'}
        results.append(res)
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    return out_df

run_pipeline()

