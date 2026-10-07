"""
text_preprocessor.py
────────────────────────────────────────────────────────────────────────────────
Extracts the 8 numerical features from raw article text to match the scale
of the UCI Online News Popularity dataset features as closely as possible.

Features extracted (in order):
    1. n_tokens_title           - word count of title
    2. n_tokens_content         - word count of content
    3. average_token_length     - average characters per word in content
    4. num_hrefs                - user supplied via form (default: 3)
    5. num_imgs                 - user supplied via form (default: 2)
    6. num_videos               - user supplied via form (default: 0)
    7. global_subjectivity      - proportion of subjective/opinion words
    8. global_sentiment_polarity - (positive - negative) / total words
────────────────────────────────────────────────────────────────────────────────
"""

# ── Very large positive word list ─────────────────────────────────────────────
_POSITIVE_WORDS = frozenset({
    "good", "great", "excellent", "best", "better", "positive", "success",
    "successful", "win", "winner", "winning", "happy", "happiness", "joy",
    "joyful", "wonderful", "amazing", "fantastic", "outstanding", "brilliant",
    "superb", "remarkable", "perfect", "awesome", "helpful", "effective",
    "innovative", "leading", "record", "boost", "boosted", "achieve",
    "achieved", "achievement", "award", "victory", "growth", "advance",
    "progress", "save", "support", "improve", "improved", "improvement",
    "benefit", "benefits", "love", "safe", "safety", "strong", "strength",
    "powerful", "power", "top", "rise", "rising", "gain", "gains", "hope",
    "hopeful", "exciting", "excited", "celebrate", "celebration", "proud",
    "pride", "champion", "champions", "thrive", "thriving", "flourish",
    "flourishing", "revolutionary", "breakthrough", "historic", "milestone",
    "incredible", "extraordinary", "exceptional", "valuable", "opportunity",
    "opportunities", "inspire", "inspiring", "inspired", "innovation",
    "creative", "creativity", "solution", "solutions", "advantage", "profit",
    "reward", "rewards", "upgrade", "launch", "launches", "launched",
    "discover", "discovery", "expand", "expanding", "expansion", "generate",
    "generating", "maximize", "maximum", "optimize", "optimized", "enhance",
    "enhanced", "efficient", "efficiency", "reliable", "reliability",
    "trusted", "trust", "loyalty", "loyal", "popular", "popularity",
    "impressive", "impress", "triumph", "triumphant", "prosperous",
    "prosperity", "recover", "recovery", "stable", "stability", "surge",
    "surging", "accelerate", "acceleration", "transform", "transformation",
    "empowering", "empower", "potential", "unlock", "unlimited", "free",
    "freedom", "clean", "clear", "bright", "brighter", "best", "healthy",
    "health", "cure", "cured", "healed", "healing", "protect", "protection",
    "secure", "security", "peace", "peaceful", "united", "unity", "together",
    "collaborate", "collaboration", "partnership", "partnerships", "deal",
    "deals", "agree", "agreement", "signed", "approved", "confirmed",
    "recommend", "recommended", "praise", "praised", "commend", "commended",
    "honor", "honored", "respect", "respected", "admire", "admired",
    "quality", "high", "higher", "highest", "true", "truth", "fair",
    "fairness", "justice", "legitimate", "open", "transparency", "transparent",
    "accessible", "access", "connected", "connect", "engage", "engagement",
    "committed", "commitment", "dedicated", "dedication", "passion",
    "passionate", "vision", "visionary", "smart", "intelligence", "intelligent",
    "advanced", "advance", "cutting", "edge", "state", "art"
})

# ── Very large negative word list ─────────────────────────────────────────────
_NEGATIVE_WORDS = frozenset({
    "bad", "worst", "terrible", "awful", "horrible", "poor", "negative",
    "fail", "failed", "failure", "loss", "lose", "losing", "sad", "sadness",
    "unhappy", "miserable", "misery", "fear", "fearful", "afraid", "danger",
    "dangerous", "risk", "risky", "threat", "threatening", "threaten",
    "attack", "attacked", "attacks", "violence", "violent", "crime", "criminal",
    "criminals", "illegal", "corrupt", "corruption", "fraud", "fraudulent",
    "scam", "scandal", "controversy", "controversial", "crisis", "crises",
    "disaster", "catastrophe", "catastrophic", "devastating", "devastation",
    "destroy", "destroyed", "destruction", "collapse", "collapsed", "crash",
    "crashed", "fall", "fallen", "decline", "declining", "decrease",
    "decreasing", "drop", "dropping", "plunge", "plunging", "shrink",
    "shrinking", "cut", "cuts", "cutting", "layoff", "layoffs", "fired",
    "dismiss", "dismissed", "bankruptcy", "bankrupt", "debt", "deficit",
    "recession", "depression", "inflation", "unemployment", "unemployed",
    "poverty", "poor", "homeless", "hunger", "starving", "starvation",
    "disease", "illness", "sick", "sickness", "death", "dead", "deadly",
    "kill", "killed", "killing", "murder", "murder", "suicide", "abuse",
    "abused", "abuse", "victim", "victims", "suffer", "suffering", "pain",
    "painful", "hurt", "injury", "injured", "war", "warfare", "conflict",
    "battle", "fight", "fighting", "bomb", "bombing", "explosion", "explode",
    "terror", "terrorist", "terrorism", "extremism", "extremist", "hate",
    "hatred", "racist", "racism", "discrimination", "prejudice", "injustice",
    "unfair", "inequality", "protest", "riot", "riots", "unrest", "chaos",
    "chaotic", "unstable", "instability", "uncertainty", "uncertain",
    "worry", "worried", "concern", "concerned", "alarm", "alarming", "alarmed",
    "shock", "shocking", "horrific", "horrifying", "horror", "nightmare",
    "tragedy", "tragic", "devastating", "heartbreaking", "devastating",
    "critical", "serious", "severe", "extreme", "dire", "grim", "bleak",
    "dark", "darkness", "broken", "break", "breaking", "damage", "damaged",
    "harm", "harmful", "hazard", "hazardous", "toxic", "pollution", "polluted",
    "contaminate", "contamination", "corrupt", "corrupted", "betray",
    "betrayal", "lie", "lies", "lying", "fake", "false", "mislead",
    "misleading", "deceive", "deception", "manipulation", "manipulate",
    "exploit", "exploitation", "oppress", "oppression", "suppress",
    "suppression", "censorship", "censor", "ban", "banned", "block",
    "blocked", "restrict", "restriction", "sanction", "sanctions", "penalty",
    "penalize", "punish", "punishment", "sentence", "convicted", "guilty",
    "arrest", "arrested", "detained", "detention", "prison", "jail",
    "charge", "charged", "accuse", "accused", "allegation", "allegations",
    "suspect", "suspected", "investigation", "investigated", "probe",
    "inquiry", "lawsuit", "sue", "sued", "fine", "fined", "evict", "evicted",
    "evacuation", "evacuate", "flee", "fled", "refugee", "displaced",
    "shortage", "lack", "missing", "lost", "damage", "destroy", "abandon"
})

# ── Subjectivity word list (opinion and judgment words) ───────────────────────
_SUBJECTIVE_WORDS = frozenset({
    "think", "believe", "feel", "seems", "appears", "likely", "probably",
    "possibly", "perhaps", "maybe", "certainly", "definitely", "clearly",
    "obviously", "apparently", "arguably", "arguably", "reportedly",
    "allegedly", "supposedly", "presumably", "essentially", "basically",
    "generally", "typically", "usually", "often", "sometimes", "rarely",
    "never", "always", "every", "most", "many", "few", "some", "several",
    "various", "numerous", "countless", "majority", "minority", "significant",
    "important", "critical", "crucial", "vital", "essential", "necessary",
    "key", "major", "minor", "primary", "secondary", "main", "central",
    "fundamental", "basic", "core", "leading", "top", "best", "worst",
    "better", "worse", "great", "small", "big", "large", "huge", "massive",
    "enormous", "tiny", "little", "new", "old", "young", "modern", "ancient",
    "traditional", "innovative", "creative", "unique", "special", "common",
    "rare", "popular", "controversial", "surprising", "unexpected",
    "shocking", "remarkable", "extraordinary", "ordinary", "normal",
    "unusual", "strange", "interesting", "boring", "exciting", "dull",
    "fascinating", "compelling", "powerful", "weak", "strong", "fragile",
    "complex", "simple", "complicated", "straightforward", "difficult",
    "easy", "challenging", "problematic", "concerning", "worrying",
    "promising", "disappointing", "impressive", "underwhelming", "beautiful",
    "ugly", "attractive", "appealing", "repulsive", "wonderful", "terrible",
    "amazing", "awful", "excellent", "horrible", "fantastic", "dreadful",
    "should", "must", "need", "require", "deserve", "ought", "could",
    "would", "might", "may", "suggest", "argue", "claim", "assert",
    "maintain", "contend", "insist", "emphasize", "highlight", "stress",
    "note", "observe", "recognize", "acknowledge", "admit", "deny",
    "reject", "accept", "support", "oppose", "agree", "disagree", "debate"
})


def _tokenize(text):
    """Split text on whitespace and punctuation into word tokens."""
    result = []
    word = []
    for ch in text.lower():
        if ch.isalpha():
            word.append(ch)
        else:
            if word:
                result.append(''.join(word))
                word = []
    if word:
        result.append(''.join(word))
    return result


def _word_count(text):
    """Count whitespace-separated words."""
    return len([w for w in text.split() if w])


def _average_word_length(text):
    """Average character count per word (excluding spaces and punctuation)."""
    words = [w for w in text.split() if w]
    if not words:
        return 4.5   # UCI dataset average
    # Strip punctuation from each word before measuring
    cleaned = []
    for w in words:
        letters = ''.join(ch for ch in w if ch.isalpha())
        if letters:
            cleaned.append(letters)
    if not cleaned:
        return 4.5
    return sum(len(w) for w in cleaned) / len(cleaned)


def extract_text_features(title, content,
                           num_hrefs=3, num_imgs=2, num_videos=0):
    """
    Extract the 8 numerical features used by the trained model.

    Parameters
    ----------
    title      : str  - article headline
    content    : str  - article body text
    num_hrefs  : int  - number of hyperlinks (default: 3)
    num_imgs   : int  - number of images (default: 2)
    num_videos : int  - number of videos (default: 0)

    Returns
    -------
    list[float] - 8 values matching SELECTED_FEATURE_COLS order
    """
    # Feature 1: word count of title
    n_tokens_title = float(_word_count(title))

    # Feature 2: word count of content
    n_tokens_content = float(_word_count(content))

    # Feature 3: average word length in content
    avg_token_length = _average_word_length(content)

    # Features 4,5,6: metadata from form
    f_hrefs  = float(num_hrefs)
    f_imgs   = float(num_imgs)
    f_videos = float(num_videos)

    # Features 7 & 8: subjectivity and sentiment from content tokens
    tokens = _tokenize(content)
    total  = max(len(tokens), 1)

    pos_count  = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg_count  = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    subj_count = sum(1 for t in tokens if t in _SUBJECTIVE_WORDS)

    # Subjectivity: proportion of opinion/sentiment words
    # Direct proportion - NO scaling multiplier (fixes the "only Viral" bug)
    raw_subjectivity = (pos_count + neg_count + subj_count) / total
    global_subjectivity = min(raw_subjectivity, 1.0)

    # Sentiment polarity: (positive - negative) ratio
    if (pos_count + neg_count) > 0:
        global_sentiment_polarity = (pos_count - neg_count) / (pos_count + neg_count)
    else:
        global_sentiment_polarity = 0.0

    return [
        n_tokens_title,           # 1. n_tokens_title
        n_tokens_content,         # 2. n_tokens_content
        avg_token_length,         # 3. average_token_length
        f_hrefs,                  # 4. num_hrefs
        f_imgs,                   # 5. num_imgs
        f_videos,                 # 6. num_videos
        global_subjectivity,      # 7. global_subjectivity
        global_sentiment_polarity, # 8. global_sentiment_polarity
    ]