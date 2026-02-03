# 中国网络环境说明（IR-base）

面向国内网络环境的依赖/模型下载优化建议。IR-base 本身是纯 Python 项目，不依赖 Docker；如需 Docker 配置，请参考上层仓库的相关文档。

---

## 1. pip 加速（推荐）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

可选镜像：
- 阿里云：`https://mirrors.aliyun.com/pypi/simple/`
- 豆瓣：`https://pypi.douban.com/simple/`

---

## 2. HuggingFace 下载加速

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

示例：
```bash
huggingface-cli download <repo_id> \
  --local-dir /path/to/models \
  --resume-download \
  --local-dir-use-symlinks False
```

若 HF-Mirror 不稳定，可使用 ModelScope 作为替代：
```bash
pip install modelscope

python - <<'PY'
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/RLRetriever', cache_dir='models/')
PY
```

---

## 3. GitHub 克隆加速（可选）

```bash
git clone https://ghproxy.com/https://github.com/username/repo.git
```

---

## 4. 常见问题

- **pip 安装失败**：检查镜像配置是否生效
- **模型下载慢**：确保设置 `HF_ENDPOINT`，或换用 ModelScope
- **网盘/代理限制**：建议在服务器侧配置可用代理或镜像

---

如需完整的构建/检索流程，请查看 `README.md` 与 `QUICKSTART.md`。
