import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from loguru import logger
import json
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        pass


class CSVDataLoader(BaseDataLoader):
    def __init__(self, low_memory: bool = False, encoding: str = 'utf-8'):
        self.low_memory = low_memory
        self.encoding = encoding
        
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        logger.info(f"Loading CSV from {source}")
        try:
            df = pd.read_csv(source, low_memory=self.low_memory, encoding=self.encoding, parse_dates=True, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows from {source}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def validate(self, df: pd.DataFrame) -> bool:
        if df.empty:
            logger.warning("Loaded DataFrame is empty")
            return False
        return True


class ParquetDataLoader(BaseDataLoader):
    """Load data from Parquet files with column filtering"""
    
    def __init__(self, engine: str = 'pyarrow'):
        self.engine = engine
        
    def load(self, source: str, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Load Parquet file with optional column selection"""
        logger.info(f"Loading Parquet from {source}")
        
        try:
            if columns:
                df = pd.read_parquet(source, columns=columns, engine=self.engine, **kwargs)
            else:
                df = pd.read_parquet(source, engine=self.engine, **kwargs)
            
            logger.info(f"Successfully loaded {len(df)} rows from {source}")
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet: {e}")
            raise
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate Parquet data structure"""
        if df.empty:
            logger.warning("Loaded DataFrame is empty")
            return False
        return True


class SQLDataLoader(BaseDataLoader):
    """Load data from SQL databases with query optimization"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        
    def load(self, query: str, chunksize: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Execute SQL query and load results"""
        logger.info(f"Executing SQL query: {query[:100]}...")
        
        try:
            from sqlalchemy import create_engine
            engine = create_engine(self.connection_string)
            
            if chunksize:
                chunks = []
                for chunk in pd.read_sql_query(query, engine, chunksize=chunksize):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_sql_query(query, engine, **kwargs)
            
            logger.info(f"Successfully loaded {len(df)} rows from SQL")
            return df
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            raise
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate SQL query results"""
        if df.empty:
            logger.warning("SQL query returned no results")
            return False
        return True


class KafkaDataLoader(BaseDataLoader):
    """Stream data from Apache Kafka with windowing support"""
    
    def __init__(self, bootstrap_servers: List[str], group_id: str):
        from kafka import KafkaConsumer
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer = None
        
    def load(self, topic: str, timeout_ms: int = 10000, max_messages: Optional[int] = None) -> pd.DataFrame:
        """Consume messages from Kafka topic"""
        logger.info(f"Starting Kafka consumer for topic: {topic}")
        
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            messages = []
            count = 0
            
            for message in self.consumer:
                if max_messages and count >= max_messages:
                    break
                messages.append(message.value)
                count += 1
                
            df = pd.DataFrame(messages)
            logger.info(f"Successfully consumed {len(df)} messages from Kafka")
            return df
            
        except Exception as e:
            logger.error(f"Error consuming Kafka messages: {e}")
            raise
        finally:
            if self.consumer:
                self.consumer.close()
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate Kafka stream data"""
        if df.empty:
            logger.warning("No messages received from Kafka")
            return False
        return True


class APIDataLoader(BaseDataLoader):
    """Load data from REST APIs with pagination support"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        
    def load(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        """Fetch data from REST API endpoint"""
        import requests
        
        logger.info(f"Fetching data from API: {endpoint}")
        
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                headers=headers,
                params=params or {},
                **kwargs
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try common pagination patterns
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'results' in data:
                    df = pd.DataFrame(data['results'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected API response format: {type(data)}")
            
            logger.info(f"Successfully loaded {len(df)} records from API")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching API data: {e}")
            raise
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate API response data"""
        if df.empty:
            logger.warning("API returned empty dataset")
            return False
        return True


class DataLoaderFactory:
    """Factory for creating appropriate data loader based on source type"""
    
    _loaders = {
        '.csv': CSVDataLoader,
        '.parquet': ParquetDataLoader,
        'sql': SQLDataLoader,
        'kafka': KafkaDataLoader,
        'api': APIDataLoader,
    }
    
    @classmethod
    def get_loader(cls, source: str, **kwargs) -> BaseDataLoader:
        """Get appropriate loader based on source type"""
        
        if source.startswith('http') or source.startswith('https'):
            return APIDataLoader(base_url=source, **kwargs)
        elif source.startswith('kafka://'):
            servers = kwargs.pop('servers', ['localhost:9092'])
            group_id = kwargs.pop('group_id', 'fraud-detection')
            return KafkaDataLoader(bootstrap_servers=servers, group_id=group_id)
        elif '://' in source:  # Assume SQL connection string
            return SQLDataLoader(connection_string=source, **kwargs)
        else:
            # File-based
            path = Path(source)
            ext = path.suffix.lower()
            
            if ext not in cls._loaders:
                raise ValueError(f"Unsupported file format: {ext}")
            
            return cls._loaders[ext](**kwargs)


def load_data(
    source: str,
    loader_type: Optional[str] = None,
    validate: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Universal data loading function
    
    Args:
        source: Data source (file path, URL, connection string, topic)
        loader_type: Force specific loader type ('csv', 'parquet', 'sql', 'kafka', 'api')
        validate: Whether to validate loaded data
        **kwargs: Additional arguments passed to loader
        
    Returns:
        pd.DataFrame: Loaded and optionally validated data
    """
    logger.info(f"Loading data from {source}")
    
    try:
        if loader_type:
            if loader_type == 'csv':
                loader = CSVDataLoader(**kwargs)
            elif loader_type == 'parquet':
                loader = ParquetDataLoader(**kwargs)
            elif loader_type == 'sql':
                loader = SQLDataLoader(**kwargs)
            elif loader_type == 'kafka':
                loader = KafkaDataLoader(**kwargs)
            elif loader_type == 'api':
                loader = APIDataLoader(**kwargs)
            else:
                raise ValueError(f"Unknown loader type: {loader_type}")
        else:
            loader = DataLoaderFactory.get_loader(source, **kwargs)
        
        df = loader.load(source, **kwargs)
        
        if validate and not loader.validate(df):
            logger.warning("Data validation failed")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
