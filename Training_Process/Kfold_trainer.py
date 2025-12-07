from sklearn.model_selection import KFold
from .epoch_trainer import FoldRunner
from Results.Logs.logger import log_message

class KFoldRunner:
    def __init__(self, dataset, model_class, config):
        self.dataset = dataset
        self.model_class = model_class
        self.config = config
        self.fold_runner = FoldRunner(model_class, config)

    def run(self):
        kfold = KFold(n_splits=self.config.train.k_folds, shuffle=True, random_state=42)
        all_train_mae, all_val_mae, all_r2 = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            log_message(f"🔁 شروع Fold {fold_idx+1}/{self.config.train.k_folds}")
            train_data = [self.dataset[i] for i in train_idx]
            val_data = [self.dataset[i] for i in val_idx]
            train_mae, val_mae, r2, best_val_loss = self.fold_runner.run_fold(train_data, val_data, fold_idx)

            all_train_mae.append(train_mae)
            all_val_mae.append(val_mae)
            all_r2.append(r2)
            log_message(f"✅ پایان Fold {fold_idx+1} - Best Val MAE: {best_val_loss:.4f}")

        return all_train_mae, all_val_mae, all_r2