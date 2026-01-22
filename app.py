from flask import Flask, render_template, url_for, request, jsonify
from keras_hub.tokenizers import SentencePieceTokenizer
import os, requests, math, random
import numpy as np

app = Flask(__name__)

TFSERVING_URL = os.getenv("TFSERVING_URL", "http://tfserving:8501/v1/models/sam9M:predict")

SEQ_LEN = 296
VOCAB_SIZE = 15000

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

try:
    with open("spm.model", "rb") as f:
        SPM_PROTO = f.read()
    SPM = SentencePieceTokenizer(
        proto=SPM_PROTO,
        add_bos=False,
        add_eos=False,
    )
except FileNotFoundError:
    print("Could not load tokenizer.") 


# -----------------------------
# Helpers: sampling + TF call
# -----------------------------
def callTFServing(prompt):
    # input_vec must be length 296 (your args_0)
    payload = {
        "signature_name": "serve",
        "instances": [{"args_0": prompt}]
    }
    r = requests.post(TFSERVING_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["predictions"][0]  # shape [296, 15000]


def tokenizePrompt(prompt: str):
    # Encode to token ids
    ids = SPM(prompt).numpy().tolist()

    # Truncate to last SEQ_LEN tokens (keep most recent context)
    if len(ids) > SEQ_LEN:
        ids = ids[-SEQ_LEN:]
    prompt_len = len(ids)

    # Pad to SEQ_LEN
    if len(ids) < SEQ_LEN:
        ids = ids + [PAD_ID] * (SEQ_LEN - len(ids))

    # Model expects floats (DT_FLOAT). We send token ids as floats.
    input_vec = [float(t) for t in ids]
    return input_vec, prompt_len, ids[:prompt_len]  # keep unpadded prompt ids for reconstruction


def detokenizeIds(token_ids):
    out = SPM.detokenize(token_ids)
    return str(out)


def softmaxNP(x):   # Numpy softmax function with significant speedup vs Math softmax
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def filterLogitsTopKTopP(logits, top_k=50, top_p=0.9):
    #logits = np.asarray(logits, dtype=np.float32)
    logits = np.asarray(logits)

    # --- top-k (matches: min_values = top_k_values[:, -1] then mask < min_values) ---
    k = int(top_k) if top_k is not None else 0
    if k > 0 and k < logits.size:
        kth = np.partition(logits, -k)[-k]
        logits = np.where(logits < kth, -1e10, logits)

    # --- top-p / nucleus (matches sort->softmax->cumsum->mask->threshold->mask) ---
    #p = float(top_p) if top_p is not None else 1.0
    #if p < 1.0:
        #sorted_idx = np.argsort(logits)[::-1]
        #sorted_logits = logits[sorted_idx]
        #sorted_probs = softmaxNP(sorted_logits)
        #cumprobs = np.cumsum(sorted_probs)

        #cutoff = cumprobs > p
        # shift cutoff right by 1 (keep at least the top token), matching tf.concat([0, cutoff[1:]])
        #if cutoff.size:
            #cutoff = np.concatenate([np.array([False]), cutoff[1:]])

        #sorted_logits_masked = np.where(cutoff, -1e10, sorted_logits)
        #thresh = np.min(sorted_logits_masked)
        #logits = np.where(logits < thresh, -1e10, logits)

    return logits


def sampleFromLogits(logits):
    """
    NumPy equivalent of tf.random.categorical(logits, 1) over a single logit vector.
    """
    #probs = softmaxNP(np.asarray(logits, dtype=np.float32))
    probs = softmaxNP(np.asarray(logits))
    index = int(np.random.choice(probs.size, p=probs))
    return logits[index]


def generateTopK(prompt: str, max_new_tokens=20, temperature=1.0, top_k=50, top_p=0.9):
    """
    Returns generated text (prompt + continuation).
    """
    # Prepare initial window + track generated ids
    window_vec, prompt_len, prompt_ids = tokenizePrompt(prompt)
    generated_ids = list(prompt_ids)

    # We'll track a "current length" inside the rolling window.
    # Once we start generating beyond SEQ_LEN, we keep using the last SEQ_LEN tokens.
    current_len = min(prompt_len, SEQ_LEN)

    for _ in range(max_new_tokens):
        logits_2d = callTFServing(window_vec)  # [SEQ_LEN][VOCAB_SIZE]

        # For teacher-forcing style LM: logits at position t predict token t+1
        # Next token after last real token often comes from pos = current_len - 1
        pos = max(current_len - 1, 0)
        logits = np.asarray(logits_2d[pos])

        # temperature
        if temperature and temperature != 1.0:
            logits = logits / float(temperature)

        # top-k filter then sample
        logits_f = filterLogitsTopKTopP(logits, top_k=top_k, top_p=top_p)
        next_id = sampleFromLogits(logits_f)

        # stop on EOS
        if EOS_ID is not None and next_id == EOS_ID:
            break

        generated_ids.append(next_id)

        # update rolling window: append next_id then keep last SEQ_LEN
        # window_vec is floats; keep sending floats
        window_vec = [x for x in window_vec if int(x) != PAD_ID]
        if len(window_vec) < SEQ_LEN:
            window_vec.append(float(next_id))
            # Pad to SEQ_LEN
            if len(window_vec) < SEQ_LEN:
                window_vec = window_vec + [PAD_ID] * (SEQ_LEN - len(window_vec))
        else:
            window_vec[:-1] = window_vec[1:]
            window_vec[-1] = float(next_id)

        # current_len increases until SEQ_LEN, then stays capped
        current_len = min(current_len + 1, SEQ_LEN)

    return detokenizeIds(generated_ids)


@app.post("/api/generate")
def apiGenerate():
    prompt = (request.json or {}).get("prompt", "")

    text = generateTopK(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.7,
        top_k=500,
        top_p=0.9,
    )
    return jsonify({"text": text})


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)