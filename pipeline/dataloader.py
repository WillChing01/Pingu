class DataLoader:
    def __init__(self, kind):
        self.kind = kind

    def __len__(self):
        raise NotImplementedError

    def iterator(self):
        raise NotImplementedError
