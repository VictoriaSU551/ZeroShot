# ZeroShot：Zero-Shot Cost Models 论文复现

复现论文：Benjamin Hilprecht、Carsten Binnig，**Zero-Shot Cost Models for Out-of-the-box Learned Cost Prediction**，PVLDB 15(11), 2361–2374, 2022。

- [论文](https://doi.org/10.14778/3551793.3551799)
- [作者源代码](https://github.com/DataManagementLab/zero-shot-cost-estimation)，基于提交 `9ceabbdd753b6ce0c21a23cd6e64214e60e1197c`
- [作者数据](https://osf.io/ga2xj/)
- [上游原始说明](docs/UPSTREAM_README.md)

本项目与另一个 **Deep DB 论文复现项目完全分离**。所有环境、下载、实验输出均在本 `ZeroShot` 目录内；使用独立 Git 仓库。这里的 `deepdb_augmented` 是 **ZeroShot 官方发布的带基数估计查询计划**，不是另一项目的训练产物。原作者代码中也包含 `deepdb.py` 和 `data_driven_cardinalities`，保留是为了维护 ZeroShot 官方实现的完整性。

## 当前范围

本阶段目标是跑通真实数据上的跨数据库训练、验证、测试、指标复算及模型重载，**不是宣称整篇论文的数值已经复现**。

2026-10-08 已完成三种输入设置（PostgreSQL 估计基数、作者提供的 DeepDB 估计基数、真实基数）各 5 epochs 的短训练。每种设置使用 8,268 条训练查询、1,460 条验证查询，并在 IMDB 的 5 个工作负载上评估共 1,968 条查询。三个模型均通过 checkpoint 重载和逐查询 Q-error 复算；5 项回归测试通过。

- 完整下载官方 `deepdb_augmented` 和 `parsed_plans` 两个预处理压缩包，验证官方 SHA256 和 ZIP CRC；128 个文件约 3.28 GB（十进制）。
- 使用作者模型、三种官方超参数配置与 QLoss；CPU 环境采用 Python 3.11、PyTorch 2.2.2、DGL 1.1.2。
- 短训练固定种子 0，每个源工作负载抽取至多 256 条查询；19 个训练数据库，IMDB 完全留作测试数据库。
- 特征统计仅由源数据库建立；源查询按上游实现分成 85% 训练、15% 验证，标签归一化也沿用上游实现。IMDB 运行时标签不进入训练或验证。
- IMDB 的 standard、index、synthetic、scale、JOB-light 分别测试，每个工作负载最多 512 条；JOB-light 在发布数据中有 50 条，全部评估。
- 查看 [运行汇总](output/reproduction_summary.json)、[独立核验](output/verification.json) 和 [后续实验与分工](docs/remaining-work.md)。结果汇总不存在或没有成功状态时，不应当声称实验完成。

短训练中得到的 Q-error 不能与论文完整训练结果直接比较。`exact` 输入包含真实基数，是理想化实验设置，不代表无需执行查询即可获得的生产输入。

## 目录

```text
ZeroShot/
  data/                     官方数据与 SHA256 清单（不上传数据）
  output/                   抽样输入、模型、预测、日志与核验结果
  .venv/                    本项目独立环境（不上传）
  models/                   作者的模型实现，附最小兼容修复
  cross_db_benchmark/        作者的 DBGen 基准实现
  data_driven_cardinalities/ 作者随 ZeroShot 提供的基数注释代码
  setup/                    作者的超参数与数据库设置
  tests/                    长期保留的兼容与评估回归测试
  docs/                     上游说明、执行范围、后续分工
  download_data.py           下载、校验、解压和跳过已验证数据
  reproduce.py               短训练、测试、checkpoint 重载与一致性校验
  verify_results.py          独立复算已完成运行的指标
```

模型、原始数据、压缩包、虚拟环境及本地论文 PDF 不提交 Git。成功解压后删除压缩包；重载验证后删除重复模型副本。保留可重用模型、抽样索引、输入哈希和最终日志。

## Windows PowerShell 快速运行

在仓库根目录执行。无需连接 PostgreSQL，因为使用作者已经采集的查询计划与运行时。

```powershell
py -3.11 -m venv .venv
$env:PIP_CACHE_DIR = "$PWD/output/pip-cache"
./.venv/Scripts/python.exe -m pip install --upgrade pip --index-url https://pypi.org/simple
./.venv/Scripts/python.exe -m pip install -r requirements-repro.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe download_data.py
./.venv/Scripts/python.exe reproduce.py --mode postgres
./.venv/Scripts/python.exe reproduce.py --mode deepdb
./.venv/Scripts/python.exe reproduce.py --mode exact
./.venv/Scripts/python.exe verify_results.py
./.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

`requirements-repro.txt` 是训练流程的最小直接依赖；`requirements-lock.txt` 固定本次已验证环境的全部版本。上游 `requirements.txt` 还包含重新采集数据库、Jupyter 及其他基数模型的依赖，不作为本流程的安装入口。
Linux 可使用 Python 3.11 创建 `.venv` 并替换解释器路径为 `.venv/bin/python`；本次实际验证平台为 Windows CPU，GPU/Linux 尚未验证。

`reproduce.py` 自动固定 Python hash seed、NumPy/PyTorch 种子，使用 4 个 CPU 线程和 `num_workers=0`。再次运行相同配置会核验已有输入与模型绑定，并重载 checkpoint；不会重新训练已完成的模型。更改模型代码或配置后，不能复用绑定不同的运行目录。

增大训练预算的示例（仍只是 IMDB 留一实验，计算量明显增加）：

```powershell
./.venv/Scripts/python.exe reproduce.py --mode postgres --samples-per-workload 0 --test-samples 0 --epochs 200
```

正式实验应先按论文冻结方案、确认内存与训练时间，再执行。[后续清单](docs/remaining-work.md) 区分完整训练、复杂查询、基线、few-shot、消融和原始数据采集。

## 对上游实现的改动

- `collections.MutableMapping` 改为 Python 3.11 支持的 `collections.abc.MutableMapping`。
- DGL 消息复制函数 `copy_src` 改为语义等价的 `copy_u`。
- 测试集改为固定顺序，消除训练随机状态导致的重载时批次变化；训练与验证集仍保留上游打乱方式。旧版本随机测试批次曾使相同模型的个别预测出现批次相关差异，固定批次后要求重载预测通过严格一致性检查。此检查证明固定批次下可复现，尚未证明所有批次组合下预测完全不变。
- 新增独立下载、抽样、实验绑定、指标核验和协作说明；不更改模型架构、损失或官方超参数。
- 评估按实际逐查询记录统计数量，不使用上游按整批累计的 `val_num_tuples` 作为真实样本量。

## 合作和许可

建议采用分支 + Pull Request；具体分工见 [remaining-work.md](docs/remaining-work.md)，规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。
上游 Apache 2.0 许可保留于 [LICENSE.md](LICENSE.md)。使用论文、代码或数据请按作者要求引用；数据与论文的权利不因代码许可而改变。

```bibtex
@article{Hilprecht2022ZeroShot,
  author = {Benjamin Hilprecht and Carsten Binnig},
  title = {Zero-Shot Cost Models for Out-of-the-box Learned Cost Prediction},
  journal = {Proceedings of the VLDB Endowment},
  volume = {15}, number = {11}, pages = {2361--2374}, year = {2022},
  doi = {10.14778/3551793.3551799}
}
```
