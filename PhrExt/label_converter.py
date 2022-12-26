class LabelConverter:
    def __init__(self):
        self.label2id = {
            'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
            'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
            'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22
        }
        self.id2label = [k for k in self.label2id.keys()]
        self.abbr2full = {
            "ADJP": "Adjective Phrase",
            "ADVP": "Adverbial Phrase",
            "CONJP": "Conjunction Phrase",
            "INTJ": "Interjection",
            "LST": "List marker",
            "NP": "Noun Phrase",
            "PP": "Preposition",
            "PRT": "Particle",
            "SBAR": "Clause introduce by subordinating conjunction",
            "UCP": "Unlike Coordinated Phrase",
            "VP": "Verb Phrase"
        }