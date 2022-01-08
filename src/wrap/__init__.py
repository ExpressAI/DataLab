import os
import json
import enum
import datasets
import pyarrow


class Save(enum.Enum):
    RUNTIME = 1
    MEMORY = 2
    LOCAL = 3


class WrapDataset(datasets.Dataset):
    def __init__(self, dataset=None, handles=[], **kwargs):
        if dataset != None:
            self.__dict__.update(dataset.__dict__)
            self.handles = handles
        else:
            super(WrapDataset, self).__init__(**kwargs)

    def __getitem__(self, key):
        item = super(WrapDataset, self).__getitem__(key)
        attrs = {}
        if hasattr(self, "handles"):
            for row in self.handles:
                attrs[row["field"]] = row["handler"](item)
        return {**item, **attrs}


class WrapDatasetDict(datasets.DatasetDict):
    def __getitem__(self, key):
        return WrapDataset(
            dataset=super(WrapDatasetDict, self).__getitem__(key),
            handles=[*self.handles]
        )

    def __load_disk(self, split):
        filename = self[split].cache_files[0]["filename"]
        mapfile = pyarrow.memory_map(filename, "r")
        return pyarrow.ipc.open_stream(mapfile).read_all()

    def __write_disk(self, split, pa_table):
        filename = self[split].cache_files[0]["filename"]
        schema = self[split]._data.table.schema.serialize()
        batches = list(map(lambda batch: batch.serialize(), pa_table.to_batches()))
        eof = b"\xFF\xFF\xFF\xFF\x00\x00\x00\x00"

        total = schema.size + len(eof)
        for batch in batches:
            total += batch.size
        mapfile = pyarrow.memory_map(filename, "r+")
        mapfile.resize(total)
        mapfile.seek(0)

        mapfile.write(schema)
        for batch in batches:
            mapfile.write(batch)
        mapfile.write(eof)

    def __delete_field(self, split, field):
        def __delete_each(obj):
            del obj[field]
            return obj
        self[split] = self[split].map(__delete_each)

    def __schema_backup(self, split, field, dtype=None):
        filename = self[split].cache_files[0]["filename"]
        (filepath, filename) = os.path.split(filename)
        (filename, extent) = os.path.splitext(filename)
        path = os.path.join(filepath, filename + ".json")
        obj = {}
        if not os.path.exists(path):
            open(path, "w")
        with open(path, "r") as obj_file:
            try:
                obj = json.load(obj_file)
            except:
                pass
        if (dtype != None):
            obj[field] = dtype
        else:
            del obj[field]
        with open(path, "w") as obj_file:
            json.dump(obj, obj_file)

    def __register(self, field, handler, type):
        if handler != None:
            if type == Save.RUNTIME:
                miss = True
                for index, item in enumerate(self.handles):
                    if item["field"] == field:
                        item["handler"] = handler
                        miss = False
                        break
                if miss:
                    self.handles.append({"field": field, "handler": handler})
            else:
                for split in self:
                    new_column = []
                    for item in self[split].__iter__():
                        new_column.append(handler(item))
                    if field in self[split].column_names:
                        self.__delete_field(split, field)
                    self[split] = self[split].add_column(field, new_column)

                    if (type == Save.LOCAL):
                        pa_table = self.__load_disk(split)
                        if field in pa_table.column_names:
                            pa_table = pa_table.drop(field)
                        pa_table = pa_table.append_column(field, pyarrow.array(new_column))
                        self.__write_disk(split, pa_table)

                        column_table = datasets.table.InMemoryTable.from_pydict({field: new_column})
                        inferred_feature = datasets.Features.from_arrow_schema(column_table.schema)
                        self.__schema_backup(split, field, inferred_feature[field].dtype)

        else:
            if type == Save.RUNTIME:
                for index, item in enumerate(self.handles):
                    if item["field"] == field:
                        self.handles.pop(index)
                        break
            else:
                for split in self:
                    self.__delete_field(split, field)
                    if (type == Save.LOCAL):
                        pa_table = self.__load_disk(split)
                        pa_table = pa_table.drop(field)
                        self.__write_disk(split, pa_table)
                        self.__schema_backup(split, field)

        if (type == Save.LOCAL):
            new_dataset = load_dataset(*self.args, **self.kwargs)
            for split in self:
                self[split] = new_dataset[split]

    def register(self, field, handler=None, **kvargs):
        runtime = True
        if (kvargs.__contains__("runtime")):
            runtime = kvargs["runtime"]
        self.__register(field, handler, Save.RUNTIME if runtime else Save.MEMORY)

    def register_reload(self, field, handler=None):
        self.__register(field, handler, Save.LOCAL)


def load_dataset(*args, **kwargs):
    dataset = WrapDatasetDict(datasets.load_dataset(*args, **kwargs))
    dataset.args = args
    dataset.kwargs = kwargs
    dataset.handles = []
    return dataset
