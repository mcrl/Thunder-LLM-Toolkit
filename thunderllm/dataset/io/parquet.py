import os
import json
from datasets.packaged_modules.json.json import *
from datasets.io.parquet import *


class SnuParquetDatasetWriter(ParquetDatasetWriter):

    def __init__(
        self,
        dataset: Dataset,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_rows: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **parquet_writer_kwargs,
    ):
        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = batch_size or get_writer_batch_size(dataset.features)
        self.num_rows = num_rows
        self.storage_options = storage_options or {}
        self.parquet_writer_kwargs = parquet_writer_kwargs

    def write(self, meta_info: dict = {}) -> int:

        if self.num_rows is None:
            return super().write()
        else:
            batch_size = self.batch_size if self.batch_size else config.DEFAULT_MAX_BATCH_SIZE
            if self.num_rows < 0 or self.num_rows > len(self.dataset):
                self.num_rows = len(self.dataset)
            batch_size = min(batch_size, self.num_rows)
            written = 0
            for idx in range(0, len(self.dataset) - len(self.dataset) % self.num_rows, self.num_rows):
                if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
                    path_or_buf = os.path.join(self.path_or_buf, str(
                        idx // self.num_rows).zfill(8) + ".parquet")
                    with fsspec.open(path_or_buf, "wb", **(self.storage_options or {})) as buffer:
                        written += self._write_num_rows(
                            file_obj=buffer, start_offset=idx, batch_size=batch_size, **self.parquet_writer_kwargs)
                else:
                    raise NotImplementedError()

            meta_info = {
                "total_samples": (idx // self.num_rows + 1) * self.num_rows,
                "num_files": (idx // self.num_rows) + 1,
                "samples_per_file": self.num_rows,
                "file_format": "parquet",
                "files": [],
                **self.dataset.meta_info,
                **meta_info,
            }

            with open(os.path.join(self.path_or_buf, 'meta_info.json'), 'w') as f:
                json.dump(meta_info, f, ensure_ascii=False)

            return written

    def _write(self, file_obj: BinaryIO, batch_size: int, **parquet_writer_kwargs) -> int:
        """Writes the pyarrow table as Parquet to a binary file handle.

        Caller is responsible for opening and closing the handle.
        """
        written = 0
        _ = parquet_writer_kwargs.pop("path_or_buf", None)
        schema = self.dataset.features.arrow_schema

        writer = pq.ParquetWriter(
            file_obj, schema=schema, **parquet_writer_kwargs)

        for offset in hf_tqdm(
            range(0, len(self.dataset), batch_size),
            unit="ba",
            desc="Creating parquet from Arrow format",
        ):
            batch = query_table(
                table=self.dataset._data,
                key=slice(offset, offset + batch_size),
                indices=self.dataset._indices,
            )
            writer.write_table(batch)
            written += batch.nbytes
        writer.close()
        return written

    def _write_num_rows(self, file_obj: BinaryIO, start_offset: int, batch_size: int, **parquet_writer_kwargs) -> int:
        """Writes the pyarrow table as Parquet to a binary file handle.

        Caller is responsible for opening and closing the handle.
        """
        written = 0
        _ = parquet_writer_kwargs.pop("path_or_buf", None)
        schema = self.dataset.features.arrow_schema

        writer = pq.ParquetWriter(
            file_obj, schema=schema, **parquet_writer_kwargs)

        for offset in hf_tqdm(
            range(start_offset, start_offset + self.num_rows, batch_size),
            unit="ba",
            desc="Creating parquet from Arrow format",
        ):
            batch = query_table(
                table=self.dataset._data,
                key=slice(offset, min(offset + batch_size,
                          start_offset + self.num_rows)),
                indices=self.dataset._indices,
            )
            writer.write_table(batch)
            written += batch.nbytes
        writer.close()
        return written
