from flask import Flask, render_template, url_for, request, jsonify
from keras_hub.tokenizers import SentencePieceTokenizer
import os, requests, math, random

app = Flask(__name__)

TFSERVING_URL = os.getenv("TFSERVING_URL", "http://tfserving:8501/v1/models/sam9M:predict")
SEQ_LEN = 296
PAD_ID = 0
VOCAB_SIZE = 15000
EOS_ID = 2
BOS_ID = 1
UNK_ID = 3
try:
    with open("spm.model", "rb") as f:
        SPM_PROTO = f.read()
except FileNotFoundError:
    print("Could not load tokenizer.") 
SPM = SentencePieceTokenizer(
    proto=SPM_PROTO,
    add_bos=False,
    add_eos=False,
    )


# -----------------------------
# Helpers: sampling + TF call
# -----------------------------
def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def topKFilter(logits, k):
    """Keep only top-k logits; set the rest to a large negative number."""
    if k is None or k <= 0 or k >= len(logits):
        return logits
    idx = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    keep = set(idx[:k])
    neg_inf = -1e5
    return [logits[i] if i in keep else neg_inf for i in range(len(logits))]

def sampleFromProbs(probs):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


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
    # SentencePiece can decode ids directly
    #return SPM.detokenize([int(x) for x in token_ids])
    out = SPM.detokenize(token_ids)
    try:
        return out.numpy().decode("utf-8")
    except Exception:
        #return str(out[0])
        return str(out)


def generateTopK(prompt: str, max_new_tokens=20, temperature=1.0, top_k=50):
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
        logits = logits_2d[pos]

        # temperature
        if temperature and temperature != 1.0:
            logits = [x / temperature for x in logits]

        # top-k filter then sample
        logits = topKFilter(logits, top_k)
        probs = softmax(logits)
        next_id = sampleFromProbs(probs)

        # stop on EOS
        if EOS_ID is not None and next_id == EOS_ID:
            break

        generated_ids.append(next_id)

        # update rolling window: append next_id then keep last SEQ_LEN
        # window_vec is floats; keep sending floats
        window_vec.append(float(next_id))
        window_vec = window_vec[-SEQ_LEN:]

        # current_len increases until SEQ_LEN, then stays capped
        current_len = min(current_len + 1, SEQ_LEN)

    return detokenizeIds(generated_ids)


@app.post("/api/generate")
def apiGenerate():
    prompt = (request.json or {}).get("prompt", "")

    text = generateTopK(
        prompt=prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
    )
    return jsonify({"text": text})


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)