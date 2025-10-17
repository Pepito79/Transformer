# Transformer from Scratch (PyTorch)

This project implements a **Transformer model from scratch** using **PyTorch**, following the architecture introduced in the paper  
📘 *"Attention Is All You Need"* (Vaswani et al., 2017).

The goal is to understand and reproduce the core building blocks of a Transformer — without relying on pre-built modules like `nn.Transformer` — and to demonstrate how self-attention, positional encoding, and residual connections work under the hood.

---

## 🚀 Features

- Implemented **entirely from scratch** with PyTorch `nn.Module`
- Includes:
  - Scaled Dot-Product Attention  
  - Multi-Head Attention  
  - Positional Encoding  
  - Feed-Forward Network  
  - Layer Normalization  
  - Residual Connections  
  - Encoder and Decoder blocks  
- Modular design for experimentation and visualization
- Easy to extend for NLP or sequence-to-sequence tasks
