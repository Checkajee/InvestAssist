"""
config module for trade agent
"""
from pathlib import Path
import yaml
import os

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class ProjectConfig:

    def __init__(self) -> None:
        # 加载配置文件
        config_filename = "config.yaml"
        yaml_path = Path(__file__).parent / config_filename
        print(f"Loading config from: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as fr:
            config = yaml.load(fr, Loader=yaml.FullLoader)
        for k in config:
            setattr(self, k, config[k])

cfg = ProjectConfig()

if __name__ == "__main__":
    print(f"System Language: {cfg.system_language}")
    print(f"LLM Config: {cfg.llm}")
    print(f"LLM Thinking Config: {cfg.llm_thinking}")
    print(f"Available attributes: {[attr for attr in dir(cfg) if not attr.startswith('_')]}")