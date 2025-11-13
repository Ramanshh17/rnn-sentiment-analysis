# RNN/LSTM for Sentiment Analysis

Implementation of "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)

## 📄 Paper
- **Title**: Long Short-Term Memory
- **Authors**: Sepp Hochreiter, Jürgen Schmidhuber
- **Year**: 1997
- **Link**: [Neural Computation](https://www.bioinf.jku.at/publications/older/2604.pdf)

## 🎯 Task
Sentiment Analysis on IMDB Movie Reviews Dataset
- **Dataset**: 50,000 movie reviews
- **Classes**: Positive (1) / Negative (0)
- **Task**: Binary classification

## 🏗️ Architecture

### LSTM Cell
The core innovation is the LSTM cell with:
- **Forget Gate**: Decides what to forget from cell state
- **Input Gate**: Decides what new information to add
- **Output Gate**: Decides what to output
- **Cell State**: Long-term memory

### Our Implementation


📚 Key Concepts
Why LSTM over vanilla RNN?
✅ Solves vanishing gradient problem
✅ Can capture long-term dependencies
✅ Better at remembering information
Bidirectional LSTM
Processes sequences in both directions:

Forward: left → right
Backward: right → left
Combines both for better context
🔬 Experiments
Run different configurations:

Bash

# Basic LSTM
python experiments/train.py --model lstm

# Bidirectional LSTM
python experiments/train.py --model bilstm

# GRU (alternative to LSTM)
python experiments/train.py --model gru
📈 Monitoring
Bash

# Start TensorBoard
tensorboard --logdir=runs

# Open browser to:
http://localhost:6006
🎓 Citation
bibtex

@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  publisher={MIT Press}
}
📝 License
MIT License