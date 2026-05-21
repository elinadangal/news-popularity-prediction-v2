"""
text_preprocessor.py
NLP preprocessing and numerical feature extraction from raw article text.
Implemented entirely from scratch - no NLTK, no regex library.

Used at PREDICTION TIME: the user types a title + content into the web form,
and we extract the same 8 numerical features the model was trained on.
"""

# ── Built-in stop-word list (replaces NLTK) ───────────────────────────────────
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "just", "than",
    "that", "this", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "each", "every", "some", "any", "few", "more", "most",
    "other", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "then",
    "once", "here", "there", "s", "t", "re", "ve", "ll", "d", "m",
})

# Positive sentiment words
_POSITIVE_WORDS = frozenset({
    "good", "great", "excellent", "positive", "best", "success", "win",
    "happy", "joy", "wonderful", "amazing", "fantastic", "top", "rise",
    "gain", "improve", "benefit", "love", "safe", "strong", "outstanding",
    "brilliant", "superb", "remarkable", "perfect", "awesome", "helpful",
    "effective", "innovative", "leading", "record", "boost", "achieve",
    "award", "victory", "growth", "advance", "progress", "save", "support",
})

# Negative sentiment words
_NEGATIVE_WORDS = frozenset({
    "bad", "worst", "negative", "fail", "loss", "sad", "terrible", "awful",
    "poor", "fall", "decline", "crisis", "danger", "risk", "attack", "death",
    "war", "crime", "hurt", "fear", "corrupt", "fraud", "scandal",
    "disaster", "collapse", "threat", "kill", "violence", "abuse", "suffer",
    "tragedy", "devastating", "horrific", "critical", "alarming", "dispute",
})

# Subjective / opinion words
_SUBJECTIVE_WORDS = frozenset({
    "big", "small", "new", "old", "high", "low", "long", "short", "young",
    "important", "major", "local", "national", "global", "political",
    "social", "economic", "public", "private", "significant", "key",
    "vital", "essential", "controversial", "popular", "rare", "shocking",
    "dramatic", "surprising", "unexpected", "historic",
})


# ── Low-level text functions ──────────────────────────────────────────────────

def _to_lower(text):
    return text.lower()


def _remove_special_chars(text):
    """Replace every non-alphanumeric, non-space character with a space."""
    result = []
    for ch in text:
        if ch.isalpha() or ch.isdigit() or ch == ' ':
            result.append(ch)
        else:
            result.append(' ')
    return ''.join(result)


def _tokenize(text):
    """Split on whitespace; drop empty strings."""
    return [t for t in text.split() if t]


def _remove_stop_words(tokens):
    return [t for t in tokens if t not in _STOP_WORDS]


def preprocess_text(text):
    """
    Full pipeline: lowercase -> remove special chars -> tokenize -> remove stop words.
    Returns a list of clean tokens.
    """
    text   = _to_lower(text)
    text   = _remove_special_chars(text)
    tokens = _tokenize(text)
    return _remove_stop_words(tokens)


# ── Feature extraction ────────────────────────────────────────────────────────

def _word_count(text):
    return len(_tokenize(text))


def _average_token_length(text):
    words = _tokenize(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def extract_text_features(title, content,
                            num_hrefs=0, num_imgs=0, num_videos=0):
    """
    Extract the 8 numerical features used by the trained model from raw text.

    Parameters
    ----------
    title      : str  - article headline
    content    : str  - article body
    num_hrefs  : int  - number of hyperlinks  (entered by user in form)
    num_imgs   : int  - number of images
    num_videos : int  - number of videos

    Returns
    -------
    list[float]  - 8 values in the order of SELECTED_FEATURE_COLS
        [n_tokens_title, n_tokens_content, average_token_length,
         num_hrefs, num_imgs, num_videos,
         global_subjectivity, global_sentiment_polarity]
    """
    tokens = preprocess_text(content)
    total  = max(len(tokens), 1)

    pos_count  = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg_count  = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    subj_count = sum(1 for t in tokens if t in _SUBJECTIVE_WORDS)

    global_subjectivity       = (pos_count + neg_count + subj_count) / total
    global_sentiment_polarity = (pos_count - neg_count) / total

    return [
        float(_word_count(title)),        # n_tokens_title
        float(_word_count(content)),      # n_tokens_content
        _average_token_length(content),   # average_token_length
        float(num_hrefs),                 # num_hrefs
        float(num_imgs),                  # num_imgs
        float(num_videos),                # num_videos
        global_subjectivity,              # global_subjectivity
        global_sentiment_polarity,        # global_sentiment_polarity
    ]