
# **Whisper**

[[博客]](https://openai.com/blog/whisper)

[[论文]](https://arxiv.org/abs/2212.04356)

[[模型卡]](https://github.com/openai/whisper/blob/main/model-card.md)

[[Colab示例]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

Whisper 是一个通用语音识别模型。它基于大规模多样化音频数据集进行训练，同时也是一个多任务模型，能够执行多语言语音识别、语音翻译和语言识别任务。

**方法**

使用 Transformer 序列到序列模型来训练多种语音处理任务，包括多语言语音识别、语音翻译、语言识别和语音活动检测。这些任务以预测解码器的标记序列形式表示，从而使单一模型能够取代传统语音处理流水线的多个阶段。这种多任务训练格式使用一组特殊标记作为任务指定符或分类目标。

**安装设置**

我们使用 Python 3.9.9 和 [PyTorch](https://pytorch.org/) 1.10.1 进行模型训练和测试，但代码库兼容 Python 3.8-3.11 和最新的 PyTorch 版本。代码库还依赖一些 Python 包，特别是 [OpenAI 的 tiktoken](https://github.com/openai/tiktoken)，它提供了快速的分词器实现。您可以通过以下命令下载并安装（或更新到）最新版本的 Whisper：

pip install -U openai-whisper

或者，您可以通过以下命令拉取并安装此代码库的最新提交及其 Python 依赖项：

pip install git+https://github.com/openai/whisper.git

要更新到最新版本，请运行：

pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

此外，您需要在系统上安装命令行工具 [ffmpeg](https://ffmpeg.org/)，可以通过大多数包管理器获得：

# 在 Ubuntu 或 Debian 上

sudo apt update && sudo apt install ffmpeg

# 在 Arch Linux 上

sudo pacman -S ffmpeg

# 在 macOS 上使用 Homebrew (https://brew.sh/)

brew install ffmpeg

# 在 Windows 上使用 Chocolatey (https://chocolatey.org/)

choco install ffmpeg

# 在 Windows 上使用 Scoop (https://scoop.sh/)

scoop install ffmpeg

您可能还需要安装 [rust](http://rust-lang.org)，以防 [tiktoken](https://github.com/openai/tiktoken) 没有为您的平台提供预编译的 wheel 文件。如果在运行上述 **pip install** 命令时出现安装错误，请按照 [入门页面](https://www.rust-lang.org/learn/get-started) 安装 Rust 开发环境。此外，可能需要配置 **PATH** 环境变量，例如 **export PATH="$HOME/.cargo/bin:$PATH"**。如果安装失败并提示 **No module named 'setuptools_rust'**，需要安装 **setuptools_rust**，可以运行以下命令：

pip install setuptools-rust

**可用模型和语言**

有六种模型尺寸，其中四种提供仅支持英语的版本，分别在速度和准确性之间做出权衡。下表列出了可用模型的名称、所需内存及相对于大模型的推理速度：

**大小**	**参数量**	**英语模型**	**多语言模型**	**所需显存**	**相对速度**

tiny**	**39 M**	**tiny.en**	**tiny**	**~1 GB**	**~10x

base**	**74 M**	**base.en**	**base**	**~1 GB**	**~7x

small**	**244 M**	**small.en**	**small**	**~2 GB**	**~4x

medium**	**769 M**	**medium.en**	**medium**	**~5 GB**	**~2x

large**	**1550 M**	**N/A**	**large**	**~10 GB**	**1x

turbo**	**809 M**	**N/A**	**turbo**	**~6 GB**	**~8x

**.en** 模型适用于仅处理英语的应用，通常表现更好，尤其是在 **tiny.en** 和 **base.en** 模型上。对于 **small.en** 和 **medium.en** 模型，这种差异会减少。此外，**turbo** 模型是 **large-v3** 的优化版本，具有更快的转录速度，同时准确性略有降低。

Whisper 的性能因语言而异。下图展示了 **large-v3** 和 **large-v2** 模型在 Common Voice 15 和 Fleurs 数据集上的 WER（词错误率）或 CER（字符错误率， *斜体* ）的表现。更多关于其他模型和数据集的 WER/CER 指标可在 [论文](https://arxiv.org/abs/2212.04356) 附录 D.1、D.2 和 D.4 中找到，附录 D.3 还提供了翻译任务的 BLEU（双语评估得分）。

**命令行使用**

以下命令将使用 **turbo** 模型转录音频文件中的语音：

whisper audio.flac audio.mp3 audio.wav --model turbo

默认设置（选择 **turbo** 模型）适用于转录英语。如果需要转录非英语的音频文件，可以使用 **--language** 参数指定语言：

whisper japanese.wav --language Japanese

添加 **--task translate** 参数可以将语音翻译为英语：

whisper japanese.wav --language Japanese --task translate

运行以下命令查看所有可用选项：

whisper --help

可用语言的完整列表请参阅 [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)。

**Python 使用**

可以在 Python 中使用 Whisper 进行转录：

**import** whisper

model = whisper.load_model(**"turbo"**)

result = model.transcribe(**"audio.mp3"**)

print(result[**"text"**])

**transcribe()** 方法会读取整个文件，并使用滑动的 30 秒窗口处理音频，对每个窗口执行自回归序列到序列预测。

**以下是使用 **whisper.detect_language()** 和 **whisper.decode()** 提供对模型的底层访问的示例：**

**import** whisper

model = whisper.load_model(**"turbo"**)

# 加载音频并对其进行填充/修剪以适配 30 秒

audio = whisper.load_audio(**"audio.mp3"**)

audio = whisper.pad_or_trim(audio)

# 生成对数梅尔频谱图并移动到与模型相同的设备

mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# 检测所讲语言

_, probs = model.detect_language(mel)

print(**f"检测到的语言: **{max(probs, key=probs.get)}**"**)

# 解码音频

options = whisper.DecodingOptions()

result = whisper.decode(model, mel, options)

# 打印识别的文本

print(result.text)

**更多示例**

请使用 [🙌 Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell) 讨论区分享更多 Whisper 的使用示例和第三方扩展，如 Web 演示、与其他工具的集成、不同平台的移植等。

**许可**

Whisper 的代码和模型权重根据 MIT 许可证发布。详情请参阅 [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE)。
