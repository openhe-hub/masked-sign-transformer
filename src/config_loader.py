import toml
import torch
import os

def load_config():
    """
    加载项目根目录的 config.toml 文件。
    假设此脚本位于 src 目录中。
    """
    # 获取当前文件所在目录的父目录，即项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config/config.toml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    config = toml.load(config_path)
    
    # --- 动态处理配置 ---
    # 处理设备自动检测
    if config['training']['device'] == 'auto':
        config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 将相对于根目录的数据路径转换为绝对路径，更稳健
    config['data']['data_dir'] = os.path.join(project_root, config['data']['data_dir'])

    return config

# 加载配置，以便项目中其他模块可以直接导入使用
# from config_loader import config
config = load_config()