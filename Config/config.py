'وحده لاشریک له'

from dataclasses import dataclass
import torch
import os

@dataclass
#آدرس ها
class PathConfig:
    #آموزش
    #آدرس ویژگی  داده ها
    data_root: str = "E:/DataSet/AV_SStruct313/TrainData/traindata"
    #آدرس اکسل ویژگی های کلی
    general_data_path: str = "E:/DataSet/AV_SStruct313/GeneralData/GeneralData 3(train)313.xlsx"
    #آدرس ذخیره نتایج
    result_root: str = os.path.join(os.path.dirname(__file__), "results")
    plots_dir: str = os.path.join(os.path.dirname(__file__), "results", "plots")
    logs_dir: str = os.path.join(os.path.dirname(__file__), "results", "logs")
    resultExcel_dir: str = "results.xlsx"
    #آدرس ذخیره مدل نهایی
    name = "GT250"
    model_save_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),"output","models", f"final_model_{name}.pth")

    #تست
    #آدرس ویژگی  داده ها
    test_data_Path: str = "E:/DataSet/AV_SStruct313/TestData/testdata"
    #آدرس اکسل ویژگی های کلی
    test_general_data_path: str ="E:/DataSet/AV_SStruct313/GeneralData/GeneralData 4(test)313.xlsx"

    #پیش بینی
    graph_name = 'B_KazemiMonfared'
    pred_path: str = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),"output", "prediction", f"predsec_{graph_name}.xlsx")   


@dataclass
class ModelConfig:
    #نام مدل
    #GT, GAT, GIN, GSAGE, GCN
    model_name = "GT"
    #تعداد لایه ها
    num_layer: int = 3
    #تعداد نورون  لایه ها
    #num-Feature=42
    #num-Label=2
    #توی مدل در بخش ورودی به صورت  خودکار تعداد نورون های ورودی مجموع ویژگی ها و تعداد بردارویژه ها نوشته میشه
    input_dim: int = 42
    hidden_dim: int = 64
    output_dim: int = 2
    #تعداد سرهای توجه
    num_heads: int = 2
    #تعداد بردارویژه ها برای پوزیشنال اینکودینگ
    pe_dim: int = 15


@dataclass
class TrainConfig:
    num_epochs: int = 250
    early_stopping_patience: int = 10
    batch_size: int = 8
    k_folds: int = 5
    learning_rate: float = 0.01
    weight_decay: float = 0.0001


@dataclass
class SchedulerConfig:
    #چند دوربره جلو و اگر بهتر نشد متوقف بشه
    lr_patience: int = 15
    #چقدر پایین بیاد هر مرحله
    lr_factor: float = 0.1


@dataclass
class SystemConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ✅ شیء کلی پیکربندی
class Config:
    paths = PathConfig()
    model = ModelConfig()
    train = TrainConfig()
    scheduler = SchedulerConfig()
    system = SystemConfig()
