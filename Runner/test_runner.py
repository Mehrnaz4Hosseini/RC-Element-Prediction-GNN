'الهی رضک به رضائک'

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


st= time.time()
#_______________________________________________________________
if __name__ == '__main__':
    #فراخوانی داده ها و انجام عملیات پیش پردازش
    config = Config()

    #فراخوانی داده ها
    general_df = pd.read_excel(config.paths.test_general_data_path) #ادرس اکسل اطلاعات عمومی پروژه ها رو میگیره
    graph_names = general_df.iloc[:, 0].dropna().tolist() #اسم همه پروژه های رو از ستون اول اکسل لیست می کنه
    data_manager = GraphBuilder(graph_names, config.paths.test_data_Path, pe_dim=config.model.pe_dim) 
    test_data = data_manager.get_all_graphs()

    #بارگذاری و دسته بندی داده 
    test_loader = DataLoader(test_data, batch_size = config.train.batch_size)

    #تعریف مدل
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
    
    # 🔄 بارگذاری وزن‌های ذخیره‌شده
    #اگر نخواهید از وزن های ذخیره شده استفاده کنید باید این بخش را حذف کنید
    model.load_state_dict(torch.load(config.paths.model_save_path, map_location=config.system.device))
    model.eval()  # تغییر حالت به evaluation

    test_model = Trainer(model, config)

    #پردازش مدل و  محاسبه خطاها
    val_loss, r2_w, r2_h, _, _ = test_model.evaluate(test_loader)
#_______________________________________________________________
et = time.time()

print("زمان اجرا:", et - st, "ثانیه")

print('یاعلی')

