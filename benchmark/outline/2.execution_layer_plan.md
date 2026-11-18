# 执行层改造方案：纯净 Helium / Helium_new

> 目标：调度器负责任务与占卡；Helium / Helium_new 只负责「拿齐输入 → 在已分配的引擎上跑图 → 返回结果/耗时」。  
> 两边改法对称。优先 **少改核心**，用已有能力接线。

---

## 1. 你提的目标 vs 建议补全

你希望的入口大致是：

> 接受 **图 + 提示词 + 被分配的卡** → 就能跑

建议补全为一次任务运行的完整入参：

| 入参 | 是否必须 | 说明 |
|------|----------|------|
| **编译图**（或可 `compile` 的图结构） | 必须 | 调度数据集里的图 |
| **运行输入 / 提示词相关绑定** | 必须* | 若图已是 `CompiledGraph` 且输入已编进去，可不再单独传；若只有结构，需要 `inputs`/提示词侧数据才能 `compile` |
| **分配到的 replica 列表** | 必须 | 不要只传 `cuda:i`，应是 **已常驻引擎的连接信息**（见下） |
| **用 Helium 还是 Helium_new** | 必须 | 对照 vs 实验；或等价地 `placement=off/spread` |
| **GenerationConfig** | 强烈建议 | model / temperature / max_tokens 等；与提示词不是同一层 |
| **HeliumRequestConfig** | 建议有默认 | CAS / RA / profiling 等；可先写死与现 ref 脚本一致 |
| **场景标签** | 调度器用，执行层可不看 | 纯净执行层可不接收标签 |

\*「提示词」若已打进编译图，执行层接口可以是 `(compiled_graph, replicas, ...)`；若图与输入分离，则是 `(graph, inputs, replicas, ...)`。数据集格式定稿时二选一即可。

**「被分配的卡」建议定义为 replica，而非裸 device：**

```text
Replica = { device, host, port, model_id }  # 或等价句柄
```

原因：大纲要求 vLLM **常驻在调度侧**；执行进程只 **连接** 这些端口。只传 `cuda:0` 却仍在 Helium 内 `vllm-local.start()`，启停还是绑死的。

---

## 2. 推荐架构（改动面最小）

```text
┌─────────────────────────────────────────────┐
│ 调度进程                                      │
│  - 启动时拉起 N 个常驻 vLLM（OpenAI 兼容 HTTP） │
│  - 队列 / 策略 / 占卡表                        │
│  - 每任务：spawn 执行进程或调用 runner API      │
└─────────────────────────────────────────────┘
                    │ 传入：图(+输入)、replicas[k]、helium|helium_new
                    ▼
┌─────────────────────────────────────────────┐
│ 纯净执行层（一任务一进程，推荐）                 │
│  HeliumServerConfig.llm_service_configs =     │
│    对每个 replica 一条 `vllm-openai` 配置       │
│    （host/port/model，不起本地引擎）            │
│  server.start()  → 只起 client + util workers │
│  server.execute(graph, request_config)        │
│  server.close()  → 只拆本次 Helium，不动池     │
└─────────────────────────────────────────────┘
```

### 为什么用 `vllm-openai` 而不是继续 `vllm-local`

代码现状：

- `vllm-local`：`LLMWorker._start` → `_llm.start()` → **起引擎进程**；`_join` → **停引擎**。
- `vllm-openai` / `OpenAILLM`：`start/stop` 只管 **HTTP client loop**，引擎在外部。

因此：执行层改配成 `vllm-openai` 指向池里的 port 后，

- **不必大改** `LLMWorker` / Processor / 图调度 / helium_new 的 spread；
- Helium `start/close` 自然不再带走 GPU 引擎；
- helium / helium_new **同一套接法**。

池侧单独负责：按 device 起 N 个 OpenAI 兼容 server（可复用仓库里已有 vLLM HTTP 启动路径），生命期覆盖整次调度实验。

### 多进程

- **推荐**：一任务一执行进程 → HeliumServer 单例互不干扰。  
- 调度进程只做队列与占卡；不在同一进程里塞多个 Server。

---

## 3. 纯净执行层对外 API（逻辑）

伪代码级（先不定文件名）：

```text
run_helium_task(
  flavor: "helium" | "helium_new",
  graph: CompiledGraph,          # 或 graph+inputs 再 compile
  replicas: list[Replica],       # 长度 = 本任务 k
  generation_config: GenerationConfig,
  request_config: HeliumRequestConfig | None = default,
) -> { execute_seconds, outputs?, error? }
```

内部步骤：

1. 按 `replicas` 组装 `llm_service_configs`（`name=vllm-openai`，每卡一个 port）。  
2. `flavor=helium_new` 时设 `HELIUM_INTRA_GRAPH_PLACEMENT=spread`（或以后改为显式参数）；对照为 `off`。  
3. `HeliumServer(cfg).start()` → `execute({name: graph}, request_config)` → 计时 → `close()`。  
4. **禁止**在执行层再 `vllm-local` 起引擎。

对照：`replicas` 长度恒为 1。  
实验：长度为策略给出的 k。

---

## 4. 预计改动范围（为何「并不多」）

| 位置 | 改动 |
|------|------|
| 调度侧 **VLLM 池**（新模块） | 启停 N 个常驻 server；查询空闲；按任务租借 replica |
| 执行侧 **薄封装**（新模块，benchmark 或 runtime 旁路） | 由 replicas 生成 ServerConfig + 跑 execute |
| `helium` / `helium_new` 核心 | **尽量不改**；最多加「从 replicas 构造 config」的小 helper |
| 现有 `vllm-local` 路径 | 保留给旧 ref 脚本；调度路径不走它 |
| 数据集 / 策略 | 本阶段可后做；执行层用手工构造的 graph+replicas 冒烟即可 |

**不必先改**：图算子、Optimizer、CAS、helium_new spread 算法本身。

**可选后续**：若坚持执行层仍用 `vllm-local` 嵌入引擎，则要改 `LLMWorker`「外部已 start 则跳过 start/stop」——改动面更大，不推荐作第一步。

---

## 5. 风险与注意点

1. **模型与 replica 一致**：GenerationConfig.model / 服务端加载的模型要匹配。  
2. **prefix cache**：仍在各常驻引擎内；任务换绑不同卡即换不同 cache，与现实验一致。  
3. **benchmarking / SystemProfiling**：可先关掉或保持与现脚本一致；注意 profiling 不要误伤计时。  
4. **旧 ref 脚本**：继续用 `vllm-local` 一键起停即可，与纯净路径并存。  
5. **helium_new 与 helium 的 Server 类**：封装里按 flavor import 不同 `HeliumServer`；配置组装逻辑共用。

---

## 6. 建议落地顺序（仍可一点一点来）

1. **池冒烟**：手动起 2 个常驻 vLLM HTTP，用极简 `vllm-openai` Helium 配置 `execute` 一张小图，验证 close 后引擎仍在。  
2. **纯净 runner API**：固定接口 `(graph, replicas, flavor) → execute_seconds`。  
3. **对照/实验各跑通一条** iterative 或 debate（replicas 手写）。  
4. 再接调度队列与策略接口。  
5. 最后再定「调度数据集」如何从原始数据导出图+输入+标签。

---

## 7. 结论

- 目标正确：**执行层纯净化**；卡由调度分配。  
- 目标需补：**replica 端点（不只 device）**、**GenerationConfig / RequestConfig**、**helium vs helium_new**、图与提示词是否已编译合一。  
- 实现上你的直觉对：**核心逻辑改得不多**；关键一刀是 **执行层改连外部 `vllm-openai`，引擎启停外置到池**，而不是在 Helium 里继续 `vllm-local` 参数化起卡。
