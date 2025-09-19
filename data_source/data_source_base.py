import pandas as pd
from pathlib import Path

class DataSourceBase:
    
    def __init__(self, name: str):
        self.name = name
        # 使用相对于当前文件的路径
        project_root = Path(__file__).parent.parent
        self.data_cache_dir = project_root / "data_cache" / self.name
        if not self.data_cache_dir.exists():
            self.data_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_data_cached(self, trigger_time: str) -> pd.DataFrame:
        """
        从本地缓存文件获取数据，返回格式应该是 pandas DataFrame
        包含列：['title', 'content', 'pub_time', 'url']
        """
        cache_file_name = trigger_time.replace(" ", "_").replace(":", "-")
        cache_file = self.data_cache_dir / f"{cache_file_name}.pkl"
        if cache_file.exists():
            df = pd.read_pickle(cache_file)
            if df['pub_time'].dtype == 'datetime64[ns]':
                df['pub_time'] = df['pub_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            return df
        else:
            return None

    def save_data_cached(self, trigger_time: str, data: pd.DataFrame): 
        cache_file_name = trigger_time.replace(" ", "_").replace(":", "-")
        cache_file = self.data_cache_dir / f"{cache_file_name}.pkl"
        data.to_pickle(cache_file)

    def get_data(self, trigger_time: str) -> pd.DataFrame:
        """
        从数据源获取数据，返回格式应该是 pandas DataFrame
        包含列：['title', 'content', 'pub_time', 'url']
        """
        pass

if __name__ == "__main__":
    pass