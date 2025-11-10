"""
Data I/O Module - Pluggable backends for Local/S3/Azure/Databricks/DENODO

Provides unified DataBackend interface for reading/writing data across
different storage systems. Supports Local filesystem, AWS S3, Azure Blob Storage,
Databricks Delta Lake, and DENODO SQL connections.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DataBackend(ABC):
    """Abstract base class for data backends."""

    @abstractmethod
    def read_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read CSV file."""
        pass

    @abstractmethod
    def read_parquet(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Parquet file."""
        pass

    @abstractmethod
    def read_excel(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Excel file."""
        pass

    @abstractmethod
    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Parquet file."""
        pass

    @abstractmethod
    def write_excel(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Excel file."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass

    @abstractmethod
    def list_files(self, path: str, pattern: str = "*") -> list[str]:
        """List files matching pattern."""
        pass


class LocalBackend(DataBackend):
    """Local filesystem backend."""

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        logger.info("Initialized LocalBackend", base_path=str(self.base_path))

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base_path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

    def read_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read CSV file."""
        resolved = self._resolve_path(path)
        logger.debug("Reading CSV", path=str(resolved))
        return pd.read_csv(resolved, **kwargs)

    def read_parquet(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Parquet file."""
        resolved = self._resolve_path(path)
        logger.debug("Reading Parquet", path=str(resolved))
        return pd.read_parquet(resolved, **kwargs)

    def read_excel(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Excel file."""
        resolved = self._resolve_path(path)
        logger.debug("Reading Excel", path=str(resolved))
        return pd.read_excel(resolved, **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Parquet file."""
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing Parquet", path=str(resolved), rows=len(df))
        df.to_parquet(resolved, **kwargs)

    def write_excel(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Excel file."""
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing Excel", path=str(resolved), rows=len(df))
        df.to_excel(resolved, **kwargs)

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return self._resolve_path(path).exists()

    def list_files(self, path: str, pattern: str = "*") -> list[str]:
        """List files matching pattern."""
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return []
        return [str(p) for p in resolved.glob(pattern)]


class S3Backend(DataBackend):
    """AWS S3 backend."""

    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1", **credentials: Any):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3Backend. Install with: pip install boto3")

        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region

        # Initialize S3 client
        session_kwargs = {}
        if "access_key_id" in credentials:
            session_kwargs["aws_access_key_id"] = credentials["access_key_id"]
        if "secret_access_key" in credentials:
            session_kwargs["aws_secret_access_key"] = credentials["secret_access_key"]

        self.s3_client = boto3.client("s3", region_name=region, **session_kwargs)
        logger.info("Initialized S3Backend", bucket=bucket, prefix=prefix, region=region)

    def _get_s3_key(self, path: str) -> str:
        """Get full S3 key."""
        if self.prefix:
            return f"{self.prefix}/{path.lstrip('/')}"
        return path.lstrip("/")

    def read_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read CSV from S3."""
        key = self._get_s3_key(path)
        logger.debug("Reading CSV from S3", bucket=self.bucket, key=key)
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return pd.read_csv(obj["Body"], **kwargs)

    def read_parquet(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Parquet from S3."""
        s3_path = f"s3://{self.bucket}/{self._get_s3_key(path)}"
        logger.debug("Reading Parquet from S3", path=s3_path)
        return pd.read_parquet(s3_path, **kwargs)

    def read_excel(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Excel from S3."""
        key = self._get_s3_key(path)
        logger.debug("Reading Excel from S3", bucket=self.bucket, key=key)
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return pd.read_excel(obj["Body"], **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Parquet to S3."""
        s3_path = f"s3://{self.bucket}/{self._get_s3_key(path)}"
        logger.info("Writing Parquet to S3", path=s3_path, rows=len(df))
        df.to_parquet(s3_path, **kwargs)

    def write_excel(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Excel to S3."""
        import io

        key = self._get_s3_key(path)
        logger.info("Writing Excel to S3", bucket=self.bucket, key=key, rows=len(df))

        buffer = io.BytesIO()
        df.to_excel(buffer, **kwargs)
        buffer.seek(0)
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())

    def exists(self, path: str) -> bool:
        """Check if object exists in S3."""
        key = self._get_s3_key(path)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_files(self, path: str, pattern: str = "*") -> list[str]:
        """List files in S3 prefix."""
        prefix = self._get_s3_key(path)
        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        if "Contents" not in response:
            return []
        return [obj["Key"] for obj in response["Contents"]]


class AzureBlobBackend(DataBackend):
    """Azure Blob Storage backend."""

    def __init__(self, container: str, account_url: str, **credentials: Any):
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required for AzureBlobBackend. "
                "Install with: pip install azure-storage-blob"
            )

        self.container = container
        self.account_url = account_url

        # Initialize blob service client
        if "account_key" in credentials:
            credential = credentials["account_key"]
        elif "sas_token" in credentials:
            credential = credentials["sas_token"]
        else:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()

        self.blob_service = BlobServiceClient(account_url=account_url, credential=credential)
        self.container_client = self.blob_service.get_container_client(container)
        logger.info("Initialized AzureBlobBackend", container=container)

    def read_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read CSV from Azure Blob."""
        logger.debug("Reading CSV from Azure Blob", path=path)
        blob_client = self.container_client.get_blob_client(path)
        stream = blob_client.download_blob()
        return pd.read_csv(stream, **kwargs)

    def read_parquet(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Parquet from Azure Blob."""
        logger.debug("Reading Parquet from Azure Blob", path=path)
        blob_client = self.container_client.get_blob_client(path)
        stream = blob_client.download_blob()
        return pd.read_parquet(stream, **kwargs)

    def read_excel(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Excel from Azure Blob."""
        logger.debug("Reading Excel from Azure Blob", path=path)
        blob_client = self.container_client.get_blob_client(path)
        stream = blob_client.download_blob()
        return pd.read_excel(stream, **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Parquet to Azure Blob."""
        import io

        logger.info("Writing Parquet to Azure Blob", path=path, rows=len(df))
        buffer = io.BytesIO()
        df.to_parquet(buffer, **kwargs)
        buffer.seek(0)

        blob_client = self.container_client.get_blob_client(path)
        blob_client.upload_blob(buffer, overwrite=True)

    def write_excel(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Excel to Azure Blob."""
        import io

        logger.info("Writing Excel to Azure Blob", path=path, rows=len(df))
        buffer = io.BytesIO()
        df.to_excel(buffer, **kwargs)
        buffer.seek(0)

        blob_client = self.container_client.get_blob_client(path)
        blob_client.upload_blob(buffer, overwrite=True)

    def exists(self, path: str) -> bool:
        """Check if blob exists."""
        blob_client = self.container_client.get_blob_client(path)
        return blob_client.exists()

    def list_files(self, path: str, pattern: str = "*") -> list[str]:
        """List blobs with prefix."""
        return [blob.name for blob in self.container_client.list_blobs(name_starts_with=path)]


class DatabricksBackend(DataBackend):
    """Databricks Delta Lake backend."""

    def __init__(self, catalog: str, schema: str, host: str, token: str, **kwargs: Any):
        try:
            from databricks import sql
        except ImportError:
            raise ImportError(
                "databricks-sql-connector is required for DatabricksBackend. "
                "Install with: pip install databricks-sql-connector"
            )

        self.catalog = catalog
        self.schema = schema
        self.host = host
        self.token = token
        logger.info("Initialized DatabricksBackend", catalog=catalog, schema=schema)

    def _get_connection(self) -> Any:
        """Get Databricks SQL connection."""
        from databricks import sql

        return sql.connect(
            server_hostname=self.host, http_path="/sql/1.0/warehouses/xxx", access_token=self.token
        )

    def read_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read CSV (not typical for Databricks)."""
        raise NotImplementedError("CSV reading not typically used with Databricks Delta Lake")

    def read_parquet(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read from Delta table."""
        table_name = f"{self.catalog}.{self.schema}.{path}"
        logger.debug("Reading Delta table", table=table_name)

        query = f"SELECT * FROM {table_name}"
        with self._get_connection() as conn:
            return pd.read_sql(query, conn)

    def read_excel(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read Excel (not typical for Databricks)."""
        raise NotImplementedError("Excel reading not typically used with Databricks Delta Lake")

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write to Delta table."""
        table_name = f"{self.catalog}.{self.schema}.{path}"
        logger.info("Writing to Delta table", table=table_name, rows=len(df))

        # This would typically use Databricks' DataFrame API or spark
        # For now, raise as not implemented - would need Spark context
        raise NotImplementedError("Writing to Delta tables requires Spark context")

    def write_excel(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write Excel (not typical for Databricks)."""
        raise NotImplementedError("Excel writing not typically used with Databricks Delta Lake")

    def exists(self, path: str) -> bool:
        """Check if Delta table exists."""
        table_name = f"{self.catalog}.{self.schema}.{path}"
        query = f"SHOW TABLES IN {self.catalog}.{self.schema} LIKE '{path}'"

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return len(cursor.fetchall()) > 0
        except Exception:
            return False

    def list_files(self, path: str, pattern: str = "*") -> list[str]:
        """List Delta tables in schema."""
        query = f"SHOW TABLES IN {self.catalog}.{self.schema}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [row[1] for row in cursor.fetchall()]


class DenodoBackend(DataBackend):
    """DENODO SQL backend."""

    def __init__(self, dsn: str, user: str, password: str, database: str, **kwargs: Any):
        try:
            import pyodbc
        except ImportError:
            raise ImportError(
                "pyodbc is required for DenodoBackend. Install with: pip install pyodbc"
            )

        self.dsn = dsn
        self.user = user
        self.password = password
        self.database = database
        logger.info("Initialized DenodoBackend", dsn=dsn, database=database)

    def _get_connection(self) -> Any:
        """Get DENODO connection."""
        import pyodbc

        return pyodbc.connect(f"DSN={self.dsn};UID={self.user};PWD={self.password};DATABASE={self.database}")

    def read_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read from view/table (path is table name)."""
        logger.debug("Reading from DENODO view", view=path)
        query = f"SELECT * FROM {path}"

        with self._get_connection() as conn:
            return pd.read_sql(query, conn)

    def read_parquet(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read from view/table."""
        return self.read_csv(path, **kwargs)

    def read_excel(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Read from view/table."""
        return self.read_csv(path, **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write to DENODO table."""
        logger.info("Writing to DENODO table", table=path, rows=len(df))

        with self._get_connection() as conn:
            df.to_sql(path, conn, if_exists="replace", index=False)

    def write_excel(self, df: pd.DataFrame, path: str, **kwargs: Any) -> None:
        """Write to DENODO table."""
        self.write_parquet(df, path, **kwargs)

    def exists(self, path: str) -> bool:
        """Check if table/view exists."""
        query = f"SELECT 1 FROM {path} WHERE 1=0"

        try:
            with self._get_connection() as conn:
                pd.read_sql(query, conn)
            return True
        except Exception:
            return False

    def list_files(self, path: str, pattern: str = "*") -> list[str]:
        """List tables in database."""
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = ?"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (path,))
            return [row[0] for row in cursor.fetchall()]


def get_backend(profile: str, config: Dict[str, Any], credentials: Optional[Dict[str, Any]] = None) -> DataBackend:
    """
    Factory function to get appropriate data backend.

    Args:
        profile: Backend profile name (local, s3, azure_blob, databricks_delta, denodo)
        config: Configuration dictionary with backend options
        credentials: Optional credentials dictionary

    Returns:
        DataBackend instance

    Raises:
        ValueError: If profile is not supported
    """
    credentials = credentials or {}

    if profile == "local":
        base_path = config.get("base_path", None)
        return LocalBackend(base_path=base_path)

    elif profile == "s3":
        return S3Backend(
            bucket=config["s3_bucket"],
            prefix=config.get("s3_prefix", ""),
            region=config.get("s3_region", "us-east-1"),
            **credentials.get("aws", {}),
        )

    elif profile == "azure_blob":
        return AzureBlobBackend(
            container=config["azure_container"],
            account_url=config["azure_account_url"],
            **credentials.get("azure", {}),
        )

    elif profile == "databricks_delta":
        return DatabricksBackend(
            catalog=config["databricks_catalog"],
            schema=config.get("databricks_schema", "default"),
            host=credentials.get("databricks", {}).get("host", ""),
            token=credentials.get("databricks", {}).get("token", ""),
        )

    elif profile == "denodo":
        return DenodoBackend(
            dsn=config["denodo_dsn"],
            user=credentials.get("denodo", {}).get("user", ""),
            password=credentials.get("denodo", {}).get("password", ""),
            database=credentials.get("denodo", {}).get("database", "treasury_data"),
        )

    else:
        raise ValueError(
            f"Unsupported backend profile: {profile}. "
            f"Supported: local, s3, azure_blob, databricks_delta, denodo"
        )
