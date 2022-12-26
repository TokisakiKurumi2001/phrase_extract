# Phrase extraction

```text
Input: PennyLane went to the school

Output: [{'Noun Phrase': 'PennyLane'}, {'Verb Phrase': 'went'}, {'Preposition': 'to'}, {'Noun Phrase': 'the school'}]
```

I train a sequence tagging model based on RoBERTa of Huggingface. The training code is given in `train.py`.

## How to use

```python
from PhrExt import PhraseExtractor

if __name__ == "__main__":
    phrase_extractor = PhraseExtractor('transZ/phrext', 'roberta-base')
    sent = "PennyLane went to the school"
    res = phrase_extractor(sent)
    print(res)
```