# penr-oz-multigram-predictor
Implementation of a Multigram Predictor with block context multiple character-level language model for understanding next character prediction algorithms

- Based on [makemore](https://github.com/karpathy/makemore)
- Using a [Neural Network service](https://github.com/derinworks/penr-oz-neural-network-torch)

## Quickstart Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/derinworks/penr-oz-neural-multigram-predictor.git
   cd penr-oz-multigram-predictor
   ```

2. **Create and Activate a Virtual Environment**:
   - **Create**:
     ```bash
     python -m venv venv
     ```
   - **Activate**:
     - On Unix or macOS:
       ```bash
       source venv/bin/activate
       ```
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Neural Network Service**:
   - **Follow instructions** on [penr-oz-neural-network-torch](https://github.com/derinworks/penr-oz-neural-network-torch?tab=readme-ov-file#quickstart-guide)

5. **Run**:
   ```bash
   python main.py
   ```
