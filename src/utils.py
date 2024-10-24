class EarlyStopping:
    def __init__(self, improvement, period) -> None:
        self.best = None
        self.best_epoch = 0
        self.improvement = improvement
        self.period = period
    
    def should_stop(self, val, epoch):
        if self.best is None:
            self.best = val
            self.best_epoch = epoch
            return False
        
        check_val = 2 * (val - self.best) / abs(val + self.best)
        if check_val > self.improvement:
            self.best_epoch = epoch
            self.best = val
        
        if epoch - self.best_epoch > self.period:
            return True
        return False
