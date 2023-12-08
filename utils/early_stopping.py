class EarlyStopping:
    def __init__(self, patience=10, delta=0.0001, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait before stopping if metric has not improved.
            delta (float): Minimum change in the monitored metric to be considered as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the metric has stopped decreasing;
                        in 'max' mode it will stop when the metric has stopped increasing.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf')
        self.delta = delta
        self.early_stop = False

        if mode == 'min':
            self.val_score = float('inf')
            self.compare_fn = lambda a, b: a < b
        elif mode == 'max':
            self.val_score = float('-inf')
            self.compare_fn = lambda a, b: a > b
        else:
            raise ValueError("mode should be either 'min' or 'max'")

    def __call__(self, score):
        if self.compare_fn(score, self.best_score + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop
