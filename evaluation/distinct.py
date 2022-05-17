from nltk import ngrams
from nltk.tokenize import WordPunctTokenizer

def distinct(cands):
    result = {}
    for i in range(1, 5):
        num, all_ngram, all_ngram_num = 0, {}, 0.
        for k, cand in enumerate(cands):
            ngs = ["_".join(c) for c in ngrams(cand, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = "%.4f"%(len(all_ngram) / float(all_ngram_num))
    return result

tokenizer=WordPunctTokenizer()
cand = None
cand_token=  [tokenizer.tokenize(c) for c in cand]
print(distinct(cand_token))