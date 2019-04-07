# AI의 음악, Google Magenta
디미고 1학년 음악 수행평가 자유주제 프로젝트 발표

- [MAESTRO Dataset의 MIDI-only version](https://magenta.tensorflow.org/datasets/maestro#maestro-v100-midizip)을 사용
- Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
  Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. [Enabling
  Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://arxiv.org/abs/1810.12247).
  arXiv preprint arXiv:1810.12247, 2018.

## Setup
macOS 기준

```bash
brew install sox
pip3 install magenta --user
```

- `run.py`에서 `CHECKPOINT_DIR`을 다운로드한 데이터셋 경로로 설정
