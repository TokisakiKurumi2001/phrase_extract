from PhrExt import PhraseExtractor

if __name__ == "__main__":
    phrase_extractor = PhraseExtractor('phrext_model/v1', 'roberta-base')
    sent = "PennyLane went to the school"
    res = phrase_extractor(sent)
    print(res)