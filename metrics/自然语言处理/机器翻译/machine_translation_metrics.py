from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.rouge_score import rouge_n
from nltk.translate.meteor_score import meteor_score

def bleu_score(reference, candidate):
    return sentence_bleu([reference], candidate)

def rouge_score(reference, candidate, n=1):
    return rouge_n([reference], candidate, n)

def meteor_score(reference, candidate):
    return meteor_score([reference], candidate)

# Example usage
if __name__ == "__main__":
    reference = "this is a maskovrd"
    candidate = "this is a trial"

    print(f"BLEU: {bleu_score(reference.split(), candidate.split()):.4f}")
    print(f"ROUGE-1: {rouge_score(reference.split(), candidate.split(), n=1):.4f}")
    print(f"METEOR: {meteor_score(reference.split(), candidate.split()):.4f}")