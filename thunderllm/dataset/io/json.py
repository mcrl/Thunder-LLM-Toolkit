import os
import json
from typing import Union, List

from datasets.packaged_modules.json.json import *
from datasets.io.json import *

from json import JSONDecodeError

@dataclass
class SnuJsonConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON."""

    features: Optional[datasets.Features] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None
    field: Optional[str] = None
    use_threads: bool = True  # deprecated
    block_size: Optional[int] = None  # deprecated
    chunksize: int = 10 << 20  # 10MB
    newlines_in_values: Optional[bool] = None
    columns: Optional[Union[str, List[str]]] = None,

    def __post_init__(self):
        super().__post_init__()


class SnuJson(Json):
    BUILDER_CONFIG_CLASS = SnuJsonConfig

    def _generate_tables(self, files):
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            # If the file is one json object and if we need to look at the items in one specific field
            if self.config.field is not None:
                with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                    dataset = ujson_loads(f.read())
                # We keep only the field we are interested in
                dataset = dataset[self.config.field]
                df = pandas_read_json(io.StringIO(ujson_dumps(dataset)))
                if df.columns.tolist() == [0]:
                    df.columns = list(
                        self.config.features) if self.config.features else ["text"]
                pa_table = pa.Table.from_pandas(df, preserve_index=False)
                yield file_idx, self._cast_table(pa_table)

            # If the file has one json object per line
            else:
                with open(file, "rb") as f:
                    batch_idx = 0
                    # Use block_size equal to the chunk size divided by 32 to leverage multithreading
                    # Set a default minimum value of 16kB if the chunk size is really small
                    block_size = max(self.config.chunksize // 32, 16 << 10)
                    encoding_errors = (
                        self.config.encoding_errors if self.config.encoding_errors is not None else "strict"
                    )
                    while True:
                        batch = f.read(self.config.chunksize)
                        if not batch:
                            break
                        # Finish current line
                        try:
                            batch += f.readline()
                        except (AttributeError, io.UnsupportedOperation):
                            batch += readline(f)
                        # PyArrow only accepts utf-8 encoded bytes
                        if self.config.encoding != "utf-8":
                            batch = batch.decode(
                                self.config.encoding, errors=encoding_errors).encode("utf-8")
                        try:
                            while True:
                                try:
                                    pa_table = paj.read_json(
                                        io.BytesIO(batch), read_options=paj.ReadOptions(block_size=block_size),
                                    )
                                    if self.config.columns:
                                        pa_table = pa_table.select(
                                            self.config.columns)
                                    break
                                except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                                    if (
                                        isinstance(e, pa.ArrowInvalid)
                                        and "straddling" not in str(e)
                                        or block_size > len(batch)
                                    ):
                                        raise
                                    else:
                                        # Increase the block size in case it was too small.
                                        # The block size will be reset for the next file.
                                        logger.debug(
                                            f"Batch of {len(batch)} bytes couldn't be parsed with block_size={block_size}. Retrying with block_size={block_size * 2}.")
                                        block_size *= 2
                        except pa.ArrowInvalid as e:
                            try:
                                with open(
                                    file, encoding=self.config.encoding, errors=self.config.encoding_errors
                                ) as f:
                                    df = pandas_read_json(
                                        f)[self.config.columns]
                            except JSONDecodeError:
                                logger.error(
                                    f"Failed to load JSON from file '{file}' with error {type(e)}: {e}. Pass this file")
                            except ValueError:
                                logger.error(
                                    f"Failed to load JSON from file '{file}' with error {type(e)}: {e}")
                                raise e
                            if df.columns.tolist() == [0]:
                                df.columns = list(
                                    self.config.features) if self.config.features else ["text"]
                            try:
                                pa_table = pa.Table.from_pandas(
                                    df, preserve_index=False)
                            except pa.ArrowInvalid as e:
                                logger.error(
                                    f"Failed to convert pandas DataFrame to Arrow Table from file '{file}' with error {type(e)}: {e}"
                                )
                                raise ValueError(
                                    f"Failed to convert pandas DataFrame to Arrow Table from file {file}."
                                ) from None
                            yield file_idx, self._cast_table(pa_table)
                            break
                        yield (file_idx, batch_idx), self._cast_table(pa_table)
                        batch_idx += 1


class SnuJsonDatasetReader(JsonDatasetReader):
    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        columns: Optional[Union[str, List[str]]] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):

        super().__init__(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )

        if columns and isinstance(columns, str):
            columns = [columns]

        self.builder = SnuJson(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            features=features,
            columns=columns,
            **kwargs,
        )


class SnuJsonDatasetWriter(JsonDatasetWriter):
    def __init__(
        self,
        dataset: Dataset,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        num_rows: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **to_json_kwargs,
    ):
        if num_proc is not None and num_proc <= 0:
            raise ValueError(f"num_proc {num_proc} must be an integer > 0.")

        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
        self.num_proc = num_proc
        self.num_rows = num_rows
        if self.num_rows:
            self.batch_size = min(self.batch_size, self.num_rows)
        self.encoding = "utf-8"
        self.storage_options = storage_options or {}
        self.to_json_kwargs = to_json_kwargs

    def write(self, meta_info: dict = {}) -> int:

        if self.num_rows is None:
            return super().write()
        else:
            _ = self.to_json_kwargs.pop("path_or_buf", None)
            orient = self.to_json_kwargs.pop("orient", "records")
            lines = self.to_json_kwargs.pop("lines", True)
            if "index" not in self.to_json_kwargs and orient in ["split", "table"]:
                self.to_json_kwargs["index"] = False

            # Determine the default compression value based on self.path_or_buf type
            default_compression = "infer" if isinstance(
                self.path_or_buf, (str, bytes, os.PathLike)) else None
            compression = self.to_json_kwargs.pop(
                "compression", default_compression)

            if compression not in [None, "infer", "gzip", "bz2", "xz"]:
                raise NotImplementedError(
                    f"`datasets` currently does not support {compression} compression")

        if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
            written = 0
            if self.num_rows < 0 or self.num_rows > len(self.dataset):
                self.num_rows = len(self.dataset)
            for idx in range(0, len(self.dataset) - len(self.dataset) % self.num_rows, self.num_rows):
                path_or_buf = os.path.join(self.path_or_buf, str(
                    idx // self.num_rows).zfill(8) + '.jsonl')
                with fsspec.open(
                    path_or_buf, "wb", compression=compression, **(self.storage_options or {})
                ) as buffer:
                    written += self._write_num_rows(
                        file_obj=buffer, orient=orient, start_offset=idx, lines=lines, **self.to_json_kwargs)

            meta_info = {
                "total_samples": (idx // self.num_rows + 1) * self.num_rows,
                "num_files": (idx // self.num_rows) + 1,
                "samples_per_file": self.num_rows,
                "file_format": "jsonl",
                **self.dataset.meta_info,
                **meta_info,
            }

            with open(os.path.join(self.path_or_buf, 'meta_info.json'), 'w') as f:
                json.dump(meta_info, f, ensure_ascii=False)
            return written
        else:
            raise NotImplementedError()

    def _write_num_rows(
        self,
        file_obj: BinaryIO,
        orient,
        start_offset: int,
        lines: bool,
        **to_json_kwargs,
    ) -> int:
        written = 0
        for offset in hf_tqdm(
            range(start_offset, start_offset + self.num_rows, self.batch_size),
            unit="ba",
            desc="Creating json from Arrow format",
        ):
            json_str = self._batch_json_num_rows(
                (offset, orient, lines, to_json_kwargs, start_offset + self.num_rows))
            written += file_obj.write(json_str)
        return written

    def _batch_json_num_rows(self, args):
        offset, orient, lines, to_json_kwargs, last_offset = args

        batch = query_table(
            table=self.dataset.data,
            key=slice(offset, min(offset + self.batch_size, last_offset)),
            indices=self.dataset._indices,
        )
        json_str = batch.to_pandas().to_json(
            path_or_buf=None, orient=orient, lines=lines, **to_json_kwargs)
        if not json_str.endswith("\n"):
            json_str += "\n"
        return json_str.encode(self.encoding)
