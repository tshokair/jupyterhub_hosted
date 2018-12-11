class continuous_label_encoder():
    def __init__(self, labels):
        self.ordered_labels = labels
        self.num_labels = len(labels)
        self.fit_labels = self.fit()
        self.inverted_map = self.inverted()

    def fit(self):
        encoded_labels = [i for i in range(0,self.num_labels)]
        encoder_dict = dict(zip(self.ordered_labels, encoded_labels))
        return encoder_dict

    def inverted(self):
        return  {v: k for k, v in self.fit_labels.items()}

    def classes(self):
        return list(self.fit_labels)

    def transform(self, non_encoded):
        return [self.fit_labels[label] for label in non_encoded]

    def invert(self, encoded):

        return [self.inverted_map[label] for label in encoded]
