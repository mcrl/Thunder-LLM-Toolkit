# Thunder-LLM-Toolkit

## Setup
### Requirements
- CUDA: 12.1+ with compatible NVIDIA drivers
- cuDNN: 9.3+
- Compiler: GCC 9+ or Clang 10+ with C++17 support
- Python: 3.11 recommended
- Anaconda
- MPI implementation (e.g. OpenMPI)
- flash-attention (https://github.com/Dao-AILab/flash-attention)
- lm-eval-harness (https://github.com/EleutherAI/lm-evaluation-harness)

### Optional
- vllm (https://github.com/vllm-project/vllm)

### Thunder-LLM-Toolkit Installation
```bash
conda create -n thunderllm python=3.11
conda activate thunderllm

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch
pip install ray transformers accelerate evaluate datasets deepspeed[autotuning]
pip install --no-build-isolation transformer_engine[pytorch]

# optional: for inference using vllm
pip install vllm==0.5.4

git clone https://github.com/mcrl/Thunder-LLM-Toolkit
cd Thunder-LLM-Toolkit
pip install .
```

### ThunderTok Installation

```bash
git submodule init
git submodule update

cd esaxx-rs
git apply ../esaxx-pub.patch

curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

cd ../tokenizers
git checkout tags/v0.21.0
git apply ../ThunderTok.patch

# To use Thunder-LLM Tokenizer
git apply ../Beta.patch

cd tokenizers/bindings/python
pip install .
```

## Training

### ThunderTok

```python
from tokenizers import models, trainers, Tokenizer, Regex
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from transformers import PreTrainedTokenizerFast

model = models.Unigram()
trainer = trainers.ThunderTokTrainer()
tokenizer = Tokenizer(model)

### KorTok stye pre-tokenizer ###
pre_tokenizer = Sequence(
				Split(pattern=Regex("'s|'t|'re|'ve|'m|'ll|'d|(?: ?\p{L}+)+|\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"), behavior='isolated'),
				ByteLevel(use_regex=False, add_prefix_space=False),
)
tokenizer.pre_tokenizer = pre_tokenizer

tokenizer.train([train_file.txt], trainer=trainer)
hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_object=tokenizer)
hf_tokenizer.save_pretrained(save_path)
```

### Pretrain
```bash
# Basic usage with defaults (16 processes, host1:8,host2:8 hosts)
bash examples/training/run.sh 16 host1:8,host2:8 examples/training/driver.sh
```

### SFT
```bash
# Basic usage with defaults (16 processes, host1:8,host2:8 hosts)
bash examples/training/run.sh 16 host1:8,host2:8 examples/training/sft_driver.sh
```
### DPO
```bash
# Basic usage with defaults (16 processes, host1:8,host2:8 hosts)
bash examples/training/run.sh 16 host1:8,host2:8 examples/training/dpo_driver.sh
```

## Evaluation

### For generic Huggingface Models

```bash
# For running other tasks
bash examples/inference/run_lmeval_reference.sh

# For humaneval / KR-Humaneval
# We currently do not support running this eval on this platform: use lm eval
bash examples/inference/run_humaneval.sh
```

### For our TE Models

```bash
# For running other tasks
bash examples/inference/run_te_reference.sh

# For humaneval / KR-Humaneval
# We currently do not support running this eval on this platform: use lm eval
bash examples/inference/run_humaneval_te.sh
```


## License


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] 

Unless otherwise stated, this repository is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


**Exception: `thunderllm/inference/` subdirectory**  
The code under `thunderllm/inference/` is derived from [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and is licensed under the **MIT License**.  
This subdirectory retains its original license and attribution.

---
| Path                              | License             |
|-----------------------------------|---------------------|
| `thunderllm/inference`            | MIT                 |
| `tests/inference`                 | MIT                 |
| everything else                   | CC BY-NC-SA 4.0     |

---

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.i


