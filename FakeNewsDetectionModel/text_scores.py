import logging

import textstat
from lexicalrichness import LexicalRichness

METHODS = {
    "flesch_reading_ease": textstat.flesch_reading_ease,
    "smog_index": textstat.smog_index,
    "flesch_kincaid_grade": textstat.flesch_kincaid_grade,
    "coleman_liau_index": textstat.coleman_liau_index,
    "automated_readability_index": textstat.automated_readability_index,
    "dale_chall_readability_score": textstat.dale_chall_readability_score,
    "difficult_words": textstat.difficult_words,
    "linsear_write_formula": textstat.linsear_write_formula,
    "gunning_fog": textstat.gunning_fog,
    "text_standard": textstat.text_standard,
}

NEGATIONS = "no|not|never|none|nothing|nobody|neither|nowhere|hardly|scarcely|barely|doesn't|isn't|wasn't|shouldn't|wouldn't|couldn't|won't|can't|don't"
INTERROGATIVES = "what|who|when|where|which|why|how"
POWER_WORDS = "improve|trust|immediately|discover|profit|learn|know|understand|powerful|best|win|more|bonus|exclusive|extra|you|free|health|guarantee|new|proven|safety|money|now|today|results|protect|help|easy|amazing|latest|extraordinary|how to|worst|ultimate|hot|first|big|anniversary|premiere|basic|complete|save|plus|create"
CASUAL_WORDS = "make|because|how|why|change|use|since|reason|therefore|result"
TENTATIVE_WORDS = "may|might|can|could|possibly|probably|it is likely|it is unlikely|it is possible|it is probable|tends to|appears to|suggests that|seems to"
EMOTION_WORDS = "ordeal|outrageous|provoke|repulsive|scandal|severe|shameful|shocking|terrible|tragic|unreliable|unstable|wicked|aggravate|agony|appalled|atrocious|corruption|damage|disastrous|disgusted|dreadful|eliminate|harmful|harsh|inconsiderate|enraged|offensive|aggressive|frustrated|controlling|resentful|anger|sad|fear|malicious|infuriated|critical|violent|vindictive|furious|contrary|condemning|sarcastic|poisonous|jealous|retaliating|desperate|alienated|unjustified|violated"


def calculate_score(data, method):
    assert method in METHODS.keys()
    logging.info(f"Processing score {method}")
    method = METHODS[method]
    return list(map(method, data['text']))


def lexical_richness(data):
    logging.info(f"Processing score lexical richness")
    ttr = []
    for doc in data['text']:
        lex = LexicalRichness(doc)
        ttr.append(lex.ttr)
    data['lexical_richness'] = ttr
    return data
