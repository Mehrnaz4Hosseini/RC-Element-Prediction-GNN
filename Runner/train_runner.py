'الهی من لی غیرک'

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
#------------------------------------
#------------------------------------

#فراخواندن کلاس هایی که برای ساخت مدل ساختیم
#import config
from Config.config import Config
from Data_Manager.graph_builder import GraphBuilder
from Training_Process.Kfold_trainer import KFoldRunner
from Models.graph_transformer import GraphTransformer
from Models.GAT import GATNet
from Models.GIN import GINNet
from Models.GSAGE import GraphSAGENet
from Models.GCN import GCNNet
from Results.Plots.visualizer import Visualizer
from Results.Result.ResultCalculator import ResultCalculator
from Results.Excel.Saver import ResultSaver
from Utils.utils import set_seed
from Results.Logs.logger import log_message


st= time.time()
print('بسم الله')
#_______________________________________________________________
if __name__ == '__main__':

    set_seed(42)  # 🔐 برای تکرارپذیری
    
    #فراخوانی داده ها و انجام عملیات پیش پردازش
    config = Config()

    log_message("✅ برنامه با موفقیت اجرا شد.")

    #فراخوانی داده ها
    general_df = pd.read_excel(config.paths.general_data_path) #ادرس اکسل اطلاعات عمومی پروژه ها رو میگیره
    graph_names = general_df.iloc[:, 0].tolist() #اسم همه پروژه های رو از ستون اول اکسل لیست می کنه
    data_manager = GraphBuilder(graph_names, config.paths.data_root, pe_dim=config.model.pe_dim) 
    dataset = data_manager.get_all_graphs()

    #پردازش مدل و  محاسبه خطاها
    model_name = config.model.model_name
    if model_name == "GT":
       runner = KFoldRunner(dataset, GraphTransformer, config)
    elif model_name == "GAT":
        runner = KFoldRunner(dataset, GATNet, config)
    elif model_name == "GIN":
        runner = KFoldRunner(dataset, GINNet, config)
    elif model_name == "GSAGE":
        runner = KFoldRunner(dataset, GraphSAGENet , config)
    elif model_name == "GCN":
        runner = KFoldRunner(dataset, GCNNet, config)

    train_mae_list, val_mae_list, r2_list = runner.run()

    #رسم نتایج روی نمودار
    #visualizer = Visualizer()
    #visualizer.plot_results(train_mae_list, val_mae_list, r2_list)

    #ذخیر خطا آموزش و ارزیابی در آخرین ایپوک و بهترین ایپوک
    calculator = ResultCalculator(train_mae_list, val_mae_list, r2_list, st)
    result_dict = calculator.get_result_dict()
    logger = ResultSaver(config.paths.resultExcel_dir)
    logger.add_result(result_dict)
#_______________________________________________________________
et = time.time()

print("زمان اجرا:", et - st, "ثانیه")

print('یاعلی')