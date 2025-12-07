from torch_geometric.loader import DataLoader
from .trainer import Trainer
from Results.Logs.logger import log_message

class FoldRunner:
    def __init__(self, model_class, config):
        self.model_class = model_class
        self.config = config

    def run_fold(self, train_data, val_data, fold_index):
        
        train_loader = DataLoader(train_data, batch_size=self.config.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.train.batch_size)
        

        model_name = self.config.model.model_name
        if model_name in ["GT", "GAT"]:
            model = self.model_class(self.config.model.num_layer,self.config.model.num_heads, self.config.model.input_dim + self.config.model.pe_dim,
                                 self.config.model.hidden_dim, self.config.model.output_dim).to(self.config.system.device)
        else:
            model = self.model_class(self.config.model.num_layer, self.config.model.input_dim + self.config.model.pe_dim,
                                    self.config.model.hidden_dim, self.config.model.output_dim).to(self.config.system.device)
            
        trainer = Trainer(model, self.config)

        best_val_loss = float('inf')
        counter = 0
        fold_train_mae, fold_val_mae, fold_r2 = [], [], []

        for epoch in range(self.config.train.num_epochs):
            train_loss = trainer.train(train_loader)
            val_loss, r2_w, r2_h, _, _ = trainer.evaluate(val_loader)
            trainer.scheduler.step(val_loss)

            log_message(f"Fold {fold_index+1} - Epoch {epoch+1}: Train MAE={train_loss:.4f}, Val MAE={val_loss:.4f}, R2_w={r2_w:.2f}, R2_h={r2_h:.2f}")

            fold_train_mae.append(train_loss)
            fold_val_mae.append(val_loss)
            fold_r2.append((r2_w, r2_h))

            #earlystoing
            #if val_loss < best_val_loss:
                #best_val_loss = val_loss
                #counter = 0
            #else:
                #counter += 1
                #if counter >= self.config.train.early_stopping_patience:
                    #log_message(f"⏹ Early stopping در Fold {fold_index+1} در Epoch {epoch+1}")
                    #break

        return fold_train_mae, fold_val_mae, fold_r2, best_val_loss