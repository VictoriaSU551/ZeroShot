# 数据

作者提供的 OSF 数据：<https://osf.io/ga2xj/>。

运行 `python download_data.py` 下载、校验 SHA256、检查 ZIP CRC 并解压：

- `runs/deepdb_augmented/`：带 DeepDB 基数估计的执行计划。
- `runs/parsed_plans/`：解析后的执行计划，含复杂查询。
- `manifest.json`：官方压缩包 SHA256、下载地址和所有解压文件 SHA256。

这两套是完整的官方预处理压缩包；原始数据库表和原始查询日志不属于当前直接训练流程。
脚本成功解压后删除压缩包；重复执行先核验已解压文件，不重复下载。数据不提交 Git。
原始数据保持不变；短训练所用抽样数据和特征统计写入 `output/inputs_*`。
