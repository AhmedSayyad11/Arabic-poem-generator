#!pip install transformers torch sentence-transformers -q

import os, pickle, re, time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, pipeline, GenerationConfig

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
DATA_PATH       = "/content/Arabic_poetry_dataset.csv"
CACHE_DIR       = "/content/poetry_cache"
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "verse_embeddings.npy")
VERSES_FILE     = os.path.join(CACHE_DIR, "verses.pkl")


# ─────────────────────────────
# 1. DATA FILTER
# ─────────────────────────────

def load_and_filter_verses(data):
    seen, verses, skipped = set(), [], 0
    for _, row in data.iterrows():
        poet = str(row.get('poet_name', 'مجهول')).strip()
        poem = str(row.get('poem_text', '')).strip()
        for line in poem.split('\n'):
            line = line.strip()
            wc   = len(line.split())
            if wc < 4 or wc > 15:                                skipped += 1; continue
            if not any('\u0600' <= c <= '\u06FF' for c in line): skipped += 1; continue
            if line in seen:                                      skipped += 1; continue
            seen.add(line)
            verses.append((poet, line))
    print(f"   Kept   : {len(verses):,} verses")
    print(f"   Skipped: {skipped:,} lines")
    return verses


# ─────────────────────────────
# 2. VERSE QUALITY RULES
# ─────────────────────────────

# Common Arabic filler/stop words that alone don't make a verse meaningful
FILLER_WORDS = {
    'في','من','إلى','على','عن','مع','هو','هي','أن','إن','كان',
    'قد','لا','ما','هذا','هذه','ذلك','تلك','التي','الذي','وقد',
    'أو','ثم','بل','لكن','حتى','إذا','كل','كما','لم','لن'
}

# Arabic verb patterns — a line should contain at least one real verb or rich noun
VERB_PATTERN = re.compile(
    r'(يَ|تَ|أَ|نَ|يُ|تُ|أُ|نُ|فَعَ|فَعِ|فَعُ|استَ|انْ|تَفَ)'
    r'|(\w{4,}(?:ون|ين|ات|ان|ين|وا|تم|نا|ها|كم|هم)$)'
)

def score_line(line, seed):
    """
    Score a candidate verse line 0–100.
    Returns (score, list_of_failures) so we can debug and pick the best.
    """
    words    = line.split()
    failures = []
    score    = 100

    # ── Rule 1: length 5–12 words ──────────────────────────────────────
    if len(words) < 5:
        failures.append("too short"); score -= 40
    elif len(words) > 12:
        failures.append("too long");  score -= 20

    # ── Rule 2: starts with seed (once) ───────────────────────────────
    seed_words = seed.split()
    if words[:len(seed_words)] != seed_words:
        failures.append("missing seed at start"); score -= 30

    # ── Rule 3: no consecutive duplicate words (أسد أسد) ──────────────
    for i in range(len(words) - 1):
        # Strip diacritics for comparison
        w1 = re.sub(r'[\u064B-\u065F]', '', words[i])
        w2 = re.sub(r'[\u064B-\u065F]', '', words[i + 1])
        if w1 == w2 and len(w1) > 1:
            failures.append(f"consecutive duplicate: {w1}"); score -= 50; break

    # ── Rule 4: no word repeated more than twice in the line ──────────
    from collections import Counter
    stripped = [re.sub(r'[\u064B-\u065F]', '', w) for w in words]
    counts   = Counter(stripped)
    for word, cnt in counts.items():
        if cnt > 2 and len(word) > 2 and word not in FILLER_WORDS:
            failures.append(f"word '{word}' repeated {cnt}x"); score -= 25; break

    # ── Rule 5: not all filler words ──────────────────────────────────
    non_filler = [w for w in stripped if w not in FILLER_WORDS and len(w) > 2]
    if len(non_filler) < 2:
        failures.append("no meaningful words"); score -= 40

    # ── Rule 6: line ends on a natural Arabic word boundary ───────────
    last = words[-1] if words else ''
    last = re.sub(r'[\u064B-\u065F،؛.!؟]', '', last)
    # Good endings: ة ا ي ن د ر م ل ب — common Arabic word-final letters
    if last and last[-1] not in 'ةاينمدربلوهتقكعسحضصجزغثذؤئءخشطظ':
        failures.append("truncated ending"); score -= 15

    # ── Rule 7: contains at least one content word (4+ letters) ───────
    long_words = [w for w in stripped if len(w) >= 4]
    if len(long_words) < 2:
        failures.append("no content words"); score -= 30

    return max(score, 0), failures


def pick_best_line(candidates, seed):
    """
    From a list of candidate lines, return the one with the highest quality score.
    candidates: list of strings
    """
    scored = []
    for line in candidates:
        s, fails = score_line(line, seed)
        scored.append((s, line, fails))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0]   # (score, line, failures)


# ─────────────────────────────
# 3. VERSE FINDER
# ─────────────────────────────

class InstantVerseFinder:
    def __init__(self, data):
        self.embedder = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        if self._cache_exists():
            self._load_index()
        else:
            print("📚 First-time setup — filtering verses...")
            self.all_verses = load_and_filter_verses(data)
            self._encode_and_save()

    def _cache_exists(self):
        return os.path.exists(EMBEDDINGS_FILE) and os.path.exists(VERSES_FILE)

    def _encode_and_save(self):
        print(f"\n⏳ Encoding {len(self.all_verses):,} verses (once only)...\n")
        texts = [v[1] for v in self.all_verses]
        self.verse_embeddings = self.embedder.encode(
            texts, batch_size=512, show_progress_bar=True,
            normalize_embeddings=True, convert_to_tensor=False
        )
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(EMBEDDINGS_FILE, self.verse_embeddings)
        with open(VERSES_FILE, 'wb') as f:
            pickle.dump(self.all_verses, f)
        print(f"\n✅ Saved to {CACHE_DIR}")

    def _load_index(self):
        print("⚡ Loading cached index...")
        t = time.time()
        self.verse_embeddings = np.load(EMBEDDINGS_FILE)
        with open(VERSES_FILE, 'rb') as f:
            self.all_verses = pickle.load(f)
        print(f"✅ {len(self.all_verses):,} verses loaded in {time.time()-t:.2f}s")

    def search(self, query, top_n=3):
        q       = self.embedder.encode(query, normalize_embeddings=True, convert_to_tensor=False)
        scores  = self.verse_embeddings @ q
        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.all_verses[i][0], self.all_verses[i][1], float(scores[i]))
                for i in top_idx]


# ─────────────────────────────
# 4. POEM GENERATOR
# ─────────────────────────────

class ArabicMicroPoet:
    def __init__(self):
        print("🔧 Loading generation model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "akhooli/gpt2-small-arabic-poetry"
        )
        self.pipe = pipeline(
            "text-generation",
            model="akhooli/gpt2-small-arabic-poetry",
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        self.pipe.model.generation_config = GenerationConfig(
            max_new_tokens=60,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        print("✅ Generator ready")

    def _extract_first_line(self, raw, seed):
        """
        Pull the single best verse line from raw model output.
        - Split on ' - ' (model separator) and newlines
        - Score every candidate
        - Return the winner
        """
        # Split on model's line separator and newlines
        text  = re.sub(r'\s*[-–—]\s*', '\n', raw)
        parts = [p.strip() for p in text.split('\n') if p.strip()]

        # Remove pure-prompt echoes (exact match of seed alone)
        parts = [p for p in parts if p != seed.strip()]

        # Must contain Arabic and 5–12 words
        candidates = []
        for p in parts:
            wc = len(p.split())
            if wc < 5 or wc > 12:
                continue
            if not any('\u0600' <= c <= '\u06FF' for c in p):
                continue
            candidates.append(p)

        if not candidates:
            return None, 0, ["no valid candidates"]

        score, best, fails = pick_best_line(candidates, seed)
        return best, score, fails

    def generate_line(self, seed):
        """
        Generate ONE high-quality verse line starting with seed.
        Runs up to 8 attempts, scores each, returns the best.
        """
        seed_clean   = seed.strip()
        all_attempts = []   # (score, line, failures)

        for attempt in range(8):
            temp = 0.68 + attempt * 0.04   # 0.68 → 0.96 across attempts

            raw = self.pipe(
                seed_clean,
                max_new_tokens=60,
                temperature=temp,
                top_k=50,
                top_p=0.93,
                repetition_penalty=1.4,    # stronger — punishes word repetition hard
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]['generated_text']

            line, score, fails = self._extract_first_line(raw, seed_clean)

            if line:
                # Enforce seed at start — if missing, prepend it
                words = line.split()
                seed_words = seed_clean.split()
                if words[:len(seed_words)] != seed_words:
                    line = seed_clean + ' ' + line
                    # Re-trim to max 12 words
                    line = ' '.join(line.split()[:12])

                # Re-score after potential prepend
                score, fails = score_line(line, seed_clean)
                all_attempts.append((score, line, fails))

                # Good enough — stop early
                if score >= 75:
                    break

        if not all_attempts:
            return f"{seed_clean} ...", 0

        # Return highest scoring attempt
        all_attempts.sort(key=lambda x: x[0], reverse=True)
        best_score, best_line, best_fails = all_attempts[0]
        return best_line, best_score


# ─────────────────────────────
# 5. MAIN
# ─────────────────────────────

def main():
    try:
        print(f"📂 Loading {DATA_PATH}...")
        data = pd.read_csv(DATA_PATH, encoding='utf-8-sig').dropna()
        print(f"✅ {len(data):,} poems loaded\n")
    except Exception as e:
        print(f"❌ {e}"); return

    poet_gen = ArabicMicroPoet()
    finder   = InstantVerseFinder(data)

    while True:
        print("\n" + "="*44)
        print("   ⚡ Arabic Poetry Assistant".center(44))
        print("="*44)
        print("  1.  Find matching verse")
        print("  2.  Generate a verse line")
        print("  3.  Exit")

        choice = input("\nChoose (1–3): ").strip()

        if choice == '1':
            query = input("\nEnter theme or word: ").strip()
            if not query:
                print("Please enter something."); continue

            t       = time.time()
            results = finder.search(query, top_n=3)
            ms      = (time.time() - t) * 1000

            print(f"\n🎯 Top matches ({ms:.0f}ms):\n")
            for i, (poet, verse, score) in enumerate(results, 1):
                print(f"  {i}. [{poet}]  relevance: {score:.2f}")
                print(f"     {verse}\n")

        elif choice == '2':
            seed = input("\nEnter starting word or phrase (Arabic): ").strip()
            if not seed:
                print("Please enter something."); continue

            print(f"\n⏳ Generating verse starting with '{seed}'...")
            t                = time.time()
            line, quality    = poet_gen.generate_line(seed)
            ms               = (time.time() - t) * 1000

            print(f"\n✨ Generated in {ms:.0f}ms  (quality score: {quality}/100):\n")
            print(f"   {line}\n")

        elif choice == '3':
            print("\nمع السلامة 👋"); break
        else:
            print("Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()