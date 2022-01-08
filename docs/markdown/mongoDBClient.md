
## `MongoDBClientCore(cluster: str)`
临时存在的内建类，请不要在包外调用。该类用于与特定的 MongoDB 账户的指定集群连接，是真正的 Client 类。
- 参数
    - `cluster (str)`：集群名称，只能是 `"cluster0"` 或 `"cluster1"` 中的一个。

## `MongoDBClient`
临时存在的内建类，请不要在包外调用。该类是 `MongoDBClientCore` 的包装类，用于提供与 MongoDB 连接的包装类，从而减少不必要的与服务器建立连接的时间代价。
- 参数
    - `cluster (str)`：集群名称，只能是 `"cluster0"` 或 `"cluster1"` 中的一个。

### `__query(database: str, collection: str, query: dict, one=True)`
内建方法，请不要在 `MongoDBClient` 类外调用。此方法向指定的数据库和集合查询对象，这是 `find` 方法的封装。
- 参数
    - `database (str)`：数据库的名称；
    - `collection (str)`：集合的名称；
    - `query (dict)`：查询条件，用 `find` 方法的查询方式；
    - `one (bool)`：是否使用 `find_one` 方法。

### `__insert(self, database: str, collection: str, data: dict)`
内建方法，请不要在 `MongoDBClient` 类外调用。此方法向指定的数据库和集合查询对象插入一条数据。
- 参数
    - `database (str)`：数据库的名称；
    - `collection (str)`：集合的名称；
    - `data (dict)`：要插入的数据。

### `drop(self, database: str, collection: str = None, confirm: bool = False)`
不是内建方法，但请尽量不要调用此函数。此方法删除云端数据库的某个数据库或集合。
- 参数
    - `database (str)`：数据库的名称；
    - `collection (str)`：可选，集合的名称，当指定此参数时只删除这个集合，否则删除整个数据库。
    - `confirm (dict)`：确认，默认为 `False`。只有传入 `True` 时才实际执行删除指令，防止错误的调用此函数导致难以挽回的损失。

### `query_metadata(self, dataset_name: str)`
从 MongoDB 获取指定数据集的元数据。
- 参数
    - `database (str)`：数据库的名称。
- 返回值 `(dict)`：含有元数据的字典。

### `insert_metadata(self, metadata: dict)`
向 MongoDB 的元数据集合（`dataset_metadata`）中插入新的元数据，此方法没有检查元数据的格式。
- 参数
    - `metadata (dict)`：数据集的元数据字典。

### `insert_sample(self, collection: str, sample: dict)`
向 MongoDB 的 `samples_of_dataset` 数据库插入样本，此方法没有检查样本数据的格式。
- 参数
    - `collection (str)`：集合的名称，此方法一般会导致新集合的建立；
    - `sample (dict)`：样本的字典。

