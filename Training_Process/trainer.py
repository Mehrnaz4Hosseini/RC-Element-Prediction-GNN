import torch
import numpy as np
from sklearn.metrics import r2_score

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.system.device
        #تابع هزینه
        self.criterion = torch.nn.L1Loss()
        #تابع بهینه سازی
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
        #تابع تنظیم نرخ یادگیری به صورت دینامیکی
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=config.scheduler.lr_patience, factor=config.scheduler.lr_factor)

    # آموزش
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(self.config.system.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    # ارزیابی و تست
    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.config.system.device)
                out = self.model(batch)
                val_loss += self.criterion(out, batch.y).item()
                #ذخیره مقادیر واقعی و پیش بینی شده برای  عرض و ارتفاع
                y_true.append(batch.y.cpu().numpy())
                y_pred.append(out.cpu().numpy())

        # تبدیل لیست بردارهای پیش‌بینی و مقادیر واقعی به آرایه‌ی دوبعدی (برای محاسبات بعدی)
        # مثلاً اگر y_true به شکل زیر باشه:
        # [array([0.1, 0.2]), array([0.3, 0.4]), array([0.5, 0.6])]
        # پس از np.vstack به شکل زیر درمیاد:
        # array([[0.1, 0.2]
        #        [0.3, 0.4],
        #        [0.5, 0.6]])
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        # مقدار R²     | تفسیر
        # ---------------------------------------------
        # 1.0          | مدل کاملاً بی‌نقصه؛ ۱۰۰٪ تغییرات خروجی رو درست پیش‌بینی کرده.
        # 0.9          | مدل ۹۰٪ از تغییرات رو توضیح داده؛ عالیه.
        # 0.5          | مدل فقط نصف تغییرات رو تونسته درک کنه؛ متوسطه.
        # 0.0          | مدل اصلاً بهتر از حدس زدن میانگین نیست.
        # < 0          | مدل بدتر از حدس زدن میانگین عمل کرده! خیلی ضعیفه.
        # ---------------------------------------------

        r2_w = r2_score(y_true[:, 0], y_pred[:, 0])
        r2_h = r2_score(y_true[:, 1], y_pred[:, 1])
        
        return val_loss / len(val_loader), r2_w, r2_h, y_true, y_pred
    
    # پیش بینی
    def prediction(self,one_sample):
        self.model.eval()
        with torch.no_grad():
            out = self.model(one_sample)
            return out
