# 配置实验任务队列

`python -m repo` 使用 `config.py` 的 `TASK_QUEUE` 构造本时段的任务。默认配置保留原来的 I0、D0、P0：

```python
TASK_QUEUE = [
    ("iterative", 0),
    ("debate", 0),
    ("parallel", 0),
]
```

每个条目是 `(任务类型, 样本编号)`。支持 `iterative`、`debate`、`parallel`；编号从 0 开始，表示该类型数据加载器输出的第几条样本。数据加载器使用既有的固定种子和顺序，编号不是原始数据文件的行号。同一数据文件和加载设置下，编号对应相同样本。

例如：

```python
TASK_QUEUE = [
    ("parallel", 0),
    ("debate", 2),
    ("iterative", 1),
    ("parallel", 4),
    ("parallel", 0),
]
```

这会按列表顺序构造 P 的第一条、D 的第三条、I 的第二条、P 的第五条，再重复一次 P 的第一条。列表顺序是交给调度器的初始队列顺序；后续执行仍服从所选调度策略。

每类只读取到所需最大编号加一的位置。重复条目会独立构造任务图，第一次使用 `parallel-0` 这样的任务编号，同一样本再次出现时使用 `parallel-0-repeat1`、`parallel-0-repeat2`，避免调度器覆盖重复任务。`TaskSpec.sample_index` 和结果报告的 `sample_index` 单独记录来源样本编号。

空列表表示没有待调度任务。未知任务类型、负编号、非整数编号、格式错误的条目或超出可用数据范围的编号都会报错，不会自动换成别的样本。

也可以通过 `build_fixed_dataset(task_queue=...)` 显式构造队列，再传给已有的 `run_experiment(tasks=...)`。不传 `task_queue` 时读取当前 `config.py`。

此次仅改变样本选择和任务队列组成。I/P 每样本的评论数、分块方式，P 的专家数，D 的问题数、代理数、辩论轮数，以及模型、生成参数、缓存和调度策略均保持原值。

`test_dataset.py` 使用替身替代数据读取及 Helium 构图，可在服务器验证编号选择、顺序、重复任务、配置默认值和参数检查，不启动引擎。本地未运行项目或测试。
