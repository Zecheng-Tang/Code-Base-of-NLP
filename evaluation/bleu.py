import nltk
from nltk.tokenize import WordPunctTokenizer


def bleu(refs, cands):
    result = {}
    for i in range(1, 5):
        result["bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu([[r] for r in refs], cands, weights=tuple([1./i for j in range(i)])))
    return result

def main():
    re