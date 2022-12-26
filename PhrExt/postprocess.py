from PhrExt import PhrExtModel, LabelConverter
from transformers import RobertaTokenizerFast
import torch

class PhraseExtractor:
    def __init__(self, ckpt: str, tokenizer_ckpt: str) :
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_ckpt)
        self.model = PhrExtModel.from_pretrained(ckpt)
        self.label_converter = LabelConverter()

    def __normalize(self, sent):
        # convert Ġ to space
        return sent.replace("Ġ", " ").strip()

    def __call__(self, sent: str):
        tokenized_inputs = self.tokenizer(sent, return_tensors="pt")
        logits = self.model(**tokenized_inputs).squeeze(0)
        predicts = torch.argmax(logits, dim=-1)
        tokens = [self.tokenizer.convert_ids_to_tokens(id) for id in tokenized_inputs.input_ids[0].tolist()[1:-1]]
        predict_labels = [self.label_converter.id2label[id] for id in predicts[1:-1]]

        which_chunk = ""
        dict = {}
        res = []
        for i, l in enumerate(predict_labels):
            if l[0] == "B":
                if i != 0:
                    dict = {}
                    dict[self.label_converter.abbr2full[which_chunk]] = self.__normalize(acc)
                    res.append(dict)
                acc = ""
                which_chunk = l[2:] 
                acc += tokens[i]
            elif l[0] == "I":
                if l[2:] == which_chunk:
                    acc += tokens[i]
                else:
                    if i != 0:
                        dict = {}
                        dict[self.label_converter.abbr2full[which_chunk]] = self.__normalize(acc)
                        res.append(dict)
                    acc = ""
                    which_chunk = l[2:]
                    acc += tokens[i]
        dict = {}
        dict[self.label_converter.abbr2full[which_chunk]] = self.__normalize(acc)
        res.append(dict)
        return res