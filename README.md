# AI_Resources

收集 AI（机器学习、深度学习等）相关学习资料，涵盖数据结构与算法、AI 产品经理、数据分析、机器学习、自然语言处理等方向。

本项目采用 **GitHub + HuggingFace 双端存储** 方案：
- **大文件 (>50MB)**: 自动分流并托管至 [HuggingFace Datasets](https://huggingface.co/datasets/hfhfn/AI_Resources)
- **小文件与代码**: 保存在 GitHub 仓库
- **动态导航**: 通过 [GitHub Pages](https://hfhfn.github.io/AI_Resources) 提供统一的搜索与下载界面

---

## 目录

### 01 - 数据结构和算法

Python 实现常见数据结构与算法，含动图演示和课件。

| 模块 | 内容 |
|------|------|
| [day01](01-数据结构和算法/day01/) | 二分查找、冒泡排序、插入排序、归并排序、选择排序、链表反转等 |
| [day02](01-数据结构和算法/day02/) | 快速排序、桶排序、基数排序、第K大元素等 |
| [day03](01-数据结构和算法/day03/) | 链表、栈、队列、环检测、回文链表等 |
| [day04](01-数据结构和算法/day04/) | 二叉树、二叉搜索树、路径和等 |
| [动图](01-数据结构和算法/动图/) | 排序算法动图演示 |
| [课件](01-数据结构和算法/课件/) | 算法课程 PDF 课件（第一课 ~ 第四课） |

---

### 02 - 成为AI产品经理

系统化的 AI 产品经理课程，从行业认知到算法技术到模型评估。

| 模块 | 内容 |
|------|------|
| [01-开篇词](02-成为AI产品经理/01-开篇词%20(1讲)/) | 产品经理的未来价值壁垒 |
| [02-知己知彼](02-成为AI产品经理/02-知己知彼，AI和AI产品经理%20(4讲)/) | 行业视角、个人视角、技术视角、成长路径 |
| [03-项目管控能力篇](02-成为AI产品经理/03-项目管控能力篇%20(3讲)/) | AI 产品落地全流程、AI 模型构建过程 |
| [04-算法技术能力篇](02-成为AI产品经理/04-算法技术能力篇%20(11讲)/) | KNN、线性回归、逻辑回归、朴素贝叶斯、决策树、SVM、K-means、深度学习 |
| [05-模型评估能力篇](02-成为AI产品经理/05-模型评估能力篇%20(7讲)/) | 混淆矩阵、KS/AUC、回归评估、PSI稳定性、模型监控 |
| [06-春节策划周](02-成为AI产品经理/06-春节策划周%20(2讲)/) | 用户增长模型、模型评估概念回顾 |

---

### 03 - 数据分析

从基础到实战的数据分析学习资料。

| 模块 | 内容 |
|------|------|
| [数据分析基础](03-数据分析/01-数据分析基础/) | Linux、MySQL、Excel、Kettle、Tableau、HDFS、Hive 等 (PPTX 课件) |
| [Excel](03-数据分析/Excel/) | Excel 数据分析案例与课件 |
| [Pandas](03-数据分析/Pandas/) | 完整 Pandas 教程（20章），含 Matplotlib/Seaborn/Pyecharts 可视化及综合案例 |

**Pandas 教程目录：**

1. Python 数据分析简介 & 环境搭建
2. DataFrame、数据结构、数据分析入门
3. 数据组合、缺失数据处理、数据整理、数据类型
4. Apply 自定义函数、数据分组、数据透视表、datetime
5. Matplotlib / Pandas / Seaborn / Pyecharts 绘图
6. 综合案例：Appstore 分析、优衣库销售分析、RFM 用户分群

---

### 04 - 机器学习

基于 GitBook 构建的机器学习教程，分为算法篇和计算库篇。

**算法篇：**

| 模块 | 内容 |
|------|------|
| [K-近邻算法](04-机器学习/算法/K-近邻算法/) | KNN 原理与实践 |
| [线性回归](04-机器学习/算法/线性回归/) | 线性回归原理与实践 |
| [逻辑回归](04-机器学习/算法/逻辑回归/) | 逻辑回归原理与实践 |
| [决策树算法](04-机器学习/算法/决策树算法/) | 决策树原理与实践 |
| [集成学习](04-机器学习/算法/集成学习/) | 集成学习方法 |
| [聚类算法](04-机器学习/算法/聚类算法/) | 聚类算法原理与实践 |

**计算库篇：**

| 模块 | 内容 |
|------|------|
| [Numpy](04-机器学习/计算库/Numpy/) | Numpy 科学计算 |
| [Pandas](04-机器学习/计算库/Pandas/) | Pandas 数据处理 |
| [Matplotlib](04-机器学习/计算库/Matplotlib/) | Matplotlib 数据可视化 |
| [环境搭建](04-机器学习/计算库/env/) | 开发环境配置 |
| [机器学习预处理](04-机器学习/计算库/ml_pre/) | 数据预处理方法 |

---

### 05 - NLP（自然语言处理）

从深度学习基础到聊天机器人项目实战的 NLP 完整教程。

**Part 1 - 神经网络和 PyTorch：**

| 模块 | 内容 |
|------|------|
| [1.1 深度学习和神经网络](05-NLP(自然语言处理)/1.1%20深度学习和神经网络/) | 深度学习介绍、神经网络介绍 |
| [1.2 PyTorch](05-NLP(自然语言处理)/1.2%20pytorch/) | PyTorch 安装使用、梯度下降、线性回归、数据加载、手写数字识别 |
| [1.3 循环神经网络](05-NLP(自然语言处理)/1.3%20循环神经网络/) | RNN 基础、情感分类、序列化容器、神经网络分词 |

**Part 2 - 项目实现：**

| 模块 | 内容 |
|------|------|
| [2.1 项目准备](05-NLP(自然语言处理)/2.1%20项目准备/) | 聊天机器人介绍、需求分析、环境/语料准备、文本分词 |
| [2.2 FastText 文本分类](05-NLP(自然语言处理)/2.2%20fasttext文本分类/) | 文本分类方法、FastText 实现与原理 |
| [2.3 Seq2Seq 与闲聊机器人](05-NLP(自然语言处理)/2.3%20Seq2Seq模型和闲聊机器人/) | Seq2Seq 原理、Attention、BeamSearch、闲聊机器人优化 |
| [2.4 QA 机器人](05-NLP(自然语言处理)/2.4%20QA机器人/) | QA 召回与排序、代码封装与接口 |

---

## 快速开始

### 1. 环境准备

确保本地已安装 Python 3.8+ 及 Git。然后安装核心依赖：

```bash
pip install "huggingface_hub>=0.17.0"
```

### 2. 克隆与初始化

```bash
git clone https://github.com/hfhfn/AI_Resources.git
cd AI_Resources
```

**一键配置：**
- **Windows**: 双击或运行 `setup.bat`
- **Linux/macOS**: 运行 `bash setup.sh`

> 该脚本将引导您完成 HuggingFace 认证、大文件首次分发及 Git 自动关联。

### 3. 配置 GitHub Actions (实现自动化)

在 GitHub 仓库设置中添加以下 Secrets：
路径：`Settings -> Secrets and variables -> Actions -> New repository secret`

| Secret 名称 | 描述 | 获取方式 |
| :--- | :--- | :--- |
| `HF_TOKEN` | HuggingFace 写入权限令牌 | [HF Tokens Settings](https://huggingface.co/settings/tokens) (创建 "Write" 权限的 Token) |

### 4. 开启 GitHub Pages

进入 `Settings -> Pages -> Build and deployment`，将 **Source** 设置为 **"GitHub Actions"**（不要选 "Deploy from a branch"）。

> 仓库已包含 `deploy-pages.yml` 静态部署 workflow 和 `.nojekyll` 文件，会自动绕过 Jekyll 构建，确保中文文件名正常访问。

部署完成后，您可以通过以下地址访问资源导航页：
`https://hfhfn.github.io/AI_Resources`

---

## 项目结构

```
AI_Resources/
├── scripts/distribute_files.py   # 核心分发引擎 (重试、日志、智能时间戳)
├── setup.bat / setup.sh          # 一键环境搭建 (同步、分发、提交、推送)
├── data/file_manifest.json       # 自动生成的文件元数据清单
├── index.html                    # 毛玻璃风格资源导航前端
├── .nojekyll                     # 绕过 Jekyll 构建
├── .github/workflows/
│   ├── distribute-files.yml      # HF 大文件同步 (只读模式)
│   └── deploy-pages.yml          # GitHub Pages 静态部署
├── 01-数据结构和算法/
├── 02-成为AI产品经理/
├── 03-数据分析/
├── 04-机器学习/
└── 05-NLP(自然语言处理)/
```

---

## 文件更新说明

### 添加大文件

1. 将文件放入仓库目录（>50MB 文件将自动转移至 HuggingFace）
2. 运行 `setup.bat`（Windows）或 `bash setup.sh`（Linux/macOS）
3. 脚本自动完成：上传至 HuggingFace、更新 `.gitignore`、生成 manifest、提交推送

### 删除大文件

1. 本地删除文件：`rm your_large_file.bin`
2. 运行 `setup.bat` 或 `bash setup.sh`
3. 脚本自动处理：清理 `.gitignore` 规则、删除 HF 远程文件、更新 manifest

> 无需手动编辑 `.gitignore`，脚本全自动处理。

---

## 常见问题

**Q: 文件没有出现在 GitHub Pages 中？**
- 检查文件是否已推送到 GitHub / HuggingFace
- 确认 GitHub Pages 已启用，且 `data/file_manifest.json` 已更新

**Q: HuggingFace 上传失败？**
- 运行 `huggingface-cli whoami` 验证登录
- 确保 `HF_TOKEN` 具有 "Write" 权限

**Q: 大文件下载很慢？**
- 国内用户可使用镜像：`huggingface-cli download hfhfn/AI_Resources --repo-type dataset --endpoint https://hf-mirror.com`

---

## 许可证
本项目收集的资料仅供学习交流使用。
