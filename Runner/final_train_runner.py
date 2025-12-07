'الحمدلله'

#------------------------------------
#------------------------------------
# افزودن مسیر ریشه پروژه به مسیر جستجوی ماژول‌های پایتون (sys.path)
# این کار باعث می‌شود بتوان فایل‌های پایتونی داخل پوشه‌های دیگر پروژه را با import فراخوانی کرد
# مخصوصاً زمانی که این اسکریپت از زیرپوشه‌ای مانند 'Runner' اجرا می‌شود
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import time
import torch
#------------------------------------
#------------------------------------

#فراخواندن کلاس هایی که برای ساخت مدل ساختیم
#import config
from Config.config import Config
from Data_Manager.graph_builder import GraphBuilder
from Training_Process.trainer import Trainer
from Models.graph_transformer import GraphTransformer
from Models.GAT import GATNet
from Models.GIN import GINNet
from Models.GSAGE import GraphSAGENet
from Models.GCN import GCNNet
from torch_geometric.loader import DataLoader
from Utils.utils import set_seed


st= time.time()
#_______________________________________________________________
if __name__ == '__main__':

    set_seed(42)  # 🔐 برای تکرارپذیری

    #فراخوانی داده ها و انجام عملیات پیش پردازش
    config = Config()

    #فراخوانی داده ها
    general_df = pd.read_excel(config.paths.test_general_data_path) #ادرس اکسل اطلاعات عمومی پروژه ها رو میگیره
    graph_names = general_df.iloc[:, 0].dropna().tolist() #اسم همه پروژه های رو از ستون اول اکسل لیست می کنه
    data_manager = GraphBuilder(graph_names, config.paths.test_data_Path, pe_dim=config.model.pe_dim) 
    all_train_data = data_manager.get_all_graphs()

    #بارگذاری و دسته بندی داده 
    all_train_loader = DataLoader(all_train_data, batch_size = config.train.batch_size)

    #تعریف مدل
    model_name = config.model.model_name
    if model_name == "GT":
        model = GraphTransformer(config.model.num_layer,config.model.num_heads, 
                             config.model.input_dim + config.model.pe_dim,
                             config.model.hidden_dim, config.model.output_dim).to(config.system.device)
    elif model_name == "GAT":
        model = GATNet(config.model.num_layer,config.model.num_heads, 
                             config.model.input_dim + config.model.pe_dim,
                             config.model.hidden_dim, config.model.output_dim).to(config.system.device)
    elif model_name == "GIN":
        model = GINNet(config.model.num_layer,  config.model.input_dim + config.model.pe_dim,
                       config.model.hidden_dim, config.model.output_dim).to(config.system.device) 
    elif model_name == "GSAGE":
        model = GraphSAGENet(config.model.num_layer,  config.model.input_dim + config.model.pe_dim,
                       config.model.hidden_dim, config.model.output_dim).to(config.system.device)
    elif model_name == "GCN":
        model = GCNNet(config.model.num_layer,  config.model.input_dim + config.model.pe_dim,
                       config.model.hidden_dim, config.model.output_dim).to(config.system.device)
        
    final_train_model = Trainer(model, config)

    #پردازش مدل و  محاسبه خطاها
    # لیستی برای ذخیره خطاها
    epoch_losses = []
    best_loss = float("inf")   # شروع با بی‌نهایت
    best_weights = None        # برای ذخیره وزن بهترین مدل
    best_epoch = -1
    for epoch in range(config.train.num_epochs):
            train_loss = final_train_model.train(all_train_loader)
            print(f"Epoch {epoch+1}/{config.train.num_epochs} - Train MAE: {train_loss:.4f}")
            # ذخیره شماره ایپوک و خطا
            epoch_losses.append({"Epoch": epoch+1, "Train_MAE": train_loss})
            # اگر خطای فعلی کمتر از بهترین خطا تا الان بود → ذخیره وزن‌ها
            if train_loss < best_loss:
                best_loss = train_loss
                best_weights = model.state_dict()
                best_epoch = epoch+1
    # ساخت دیتافریم از لیست
    df = pd.DataFrame(epoch_losses)
    # ذخیره به فایل اکسل
    df.to_excel("epoch_losses.xlsx", index=False)

# ذخیره وزن‌های بهترین ایپوک
os.makedirs(os.path.dirname(config.paths.model_save_path), exist_ok=True)
torch.save(best_weights, config.paths.model_save_path)
print(f"✅ بهترین وزن‌ها مربوط به ایپوک {best_epoch} با MAE={best_loss:.4f} ذخیره شد.")
#_______________________________________________________________
et = time.time()

print("زمان اجرا:", et - st, "ثانیه")

print('یاعلی')