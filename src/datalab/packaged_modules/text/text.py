from dataclasses import dataclass
from typing import Optional

import pyarrow as pa

import datalab


logger = datalab.utils.logging.get_logger(__name__)


@dataclass
class TextConfig(datalab.BuilderConfig):
    """BuilderConfig for text files."""

    features: Optional[datalab.Features] = None
    encoding: str = "utf-8"
    chunksize: int = 10 << 20  # 10MB
    keep_linebreaks: bool = False


class Text(datalab.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = TextConfig

    def _info(self):
        return datalab.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """The `data_files` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].

        If str or List[str], then the dataset returns only the 'train' split.
        If dict, then keys should be from the `datalab.Split` enum.
        """
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [datalab.SplitGenerator(name=datalab.Split.TRAIN, gen_kwargs={"files": files})]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datalab.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_tables(self, files):
        schema = pa.schema(self.config.features.type if self.config.features is not None else {"text": pa.string()})
        for file_idx, file in enumerate(files):
            batch_idx = 0
            with open(file, "r", encoding=self.config.encoding) as f:
                while True:
                    batch = f.read(self.config.chunksize)
                    if not batch:
                        break
                    batch += f.readline()  # finish current line
                    batch = batch.splitlines(keepends=self.config.keep_linebreaks)
                    pa_table = pa.Table.from_arrays([pa.array(batch)], schema=schema)
                    # Uncomment for debugging (will print the Arrow table size and elements)
                    # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                    # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                    yield (file_idx, batch_idx), pa_table
                    batch_idx += 1
