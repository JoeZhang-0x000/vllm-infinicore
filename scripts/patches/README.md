# 平台补丁索引

每个平台使用独立目录，目录中保存 `manifest.json`、补丁和说明。根目录不承载某个平台的默认清单。

| 平台 | 补丁目标 | 清单 | 获取方式 |
| --- | --- | --- | --- |
| [MetaX](metax/README.md) | 模块化 InfiniCore 的 InfiniOps 子模块 | [manifest.json](metax/manifest.json) | 固定上游提交或本地离线副本 |
| [Ascend](ascend/README.md) | legacy InfiniCore | [manifest.json](ascend/manifest.json) | 本地补丁 |

CUDA 构建目前没有专用补丁，不复用 MetaX 补丁。各清单的 `base_revision` 必须与 [版本锁](../../vllm_infinicore/infinicore.lock.json) 对应组件一致；`patches` 数组定义应用顺序，`files` 记录应用后源码的 SHA256。

移动或重命名补丁时保持文件字节不变，并同步清单的 `file` 字段和引用。项目 `.gitattributes` 对全部平台的 `.patch` 禁用文本换行转换与补丁自身的空白检查，避免 Git 改写字节而破坏 SHA256 校验。
