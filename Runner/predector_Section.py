'ایاک النعبد و ایاک النستعین'

#------------------------------------
#------------------------------------
# افزودن مسیر ریشه پروژه به مسیر جستجوی ماژول‌های پایتون (sys.path)
# این کار باعث می‌شود بتوان فایل‌های پایتونی داخل پوشه‌های دیگر پروژه را با import فراخوانی کرد
# مخصوصاً زمانی که این اسکریپت از زیرپوشه‌ای مانند 'Runner' اجرا می‌شود
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
#------------------------------------
import pandas as pd
import time
import torch


st= time.time()
#_______________________________________________________________
if __name__ == '__main__':
    #فراخوانی داده ها و انجام عملیات پیش پردازش
    config = Config()

    #فراخوانی داده ها
    graph_name = config.paths.graph_name
    data_manager = GraphBuilder(graph_name, config.paths.test_data_Path, pe_dim=config.model.pe_dim) 
    graph , feature_df  = data_manager.get_one_graph()
    graph = graph.to(config.system.device)

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
    
    pred_model = Trainer(model, config)

    # 🔄 بارگذاری وزن‌های ذخیره‌شده
    #اگر نخواهید از وزن های ذخیره شده استفاده کنید باید این بخش را حذف کنید
    model.load_state_dict(torch.load(config.paths.model_save_path, map_location=config.system.device))
    model.eval()  # تغییر حالت به evaluation

    #پیش بینی مقاطع
    y_pred = pred_model.prediction(graph).cpu().numpy() # ← خروجی مدل
      
    y_true = graph.y.cpu().numpy()                      # ← مقادیر واقعی

    #نام نودها از اکسل ویژگی‌ها
    node_names = feature_df.iloc[:, 0].values

    # 3. ساخت دیتافریم خروجی
    df = pd.DataFrame({
        "NodeName": node_names,
        "Predicted_Width": y_pred[:, 0],
        "Predicted_Height": y_pred[:, 1],
        "True_Width": y_true[:, 0],
        "True_Height": y_true[:, 1],
    })

    #ذخیره به فایل اکسل
    df.to_excel(config.paths.pred_path, index=False)
#_______________________________________________________________
et = time.time()

print("زمان اجرا:", et - st, "ثانیه")

print('یاعلی')

