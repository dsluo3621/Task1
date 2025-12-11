import logging
import os
import pandas as pd


def init_logger(log_file: str = "../logs/mcv2_analysis.log"):
    """初始化日志系统"""
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 配置日志格式
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),  # 写入文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    logging.info("日志系统初始化完成")


def export_to_csv(df: pd.DataFrame, save_path: str) -> bool:
    """导出DataFrame到CSV文件"""
    try:
        # 创建导出目录（父目录不存在则报错）
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            raise PermissionError(f"目录不存在且无法创建：{parent_dir}")

        # 导出CSV（保留索引，编码为utf-8）
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        logging.info(f"数据导出成功：{save_path}")
        return True
    except PermissionError as e:
        logging.error(f"数据导出失败（权限不足/目录不存在）：{str(e)}")
        return False
    except Exception as e:
        logging.error(f"数据导出失败：{str(e)}")
        return False