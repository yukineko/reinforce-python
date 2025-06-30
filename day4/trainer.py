class Trainer():
    def __init__(self,
                 buffer_size=1024,
                 batch_size=12,
                 gamma=0.9,
                 report_interval=10,
                 log_dir=""):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        
            