# Amazon Prediction Pipeline for PyTorch Neural Network

An end-to-end machine-learning project that turns Amazon order history into product-category labels, trains a PyTorch classifier, and predicts a category from order timing and previous-order context.

## Technical Experiences

**Amazon Prediction Pipeline for PyTorch Neural Network**
Personal machine-learning project

- Addressed unlabeled product data by embedding product names with `all-MiniLM-L6-v2`, reducing the embeddings with PCA, and clustering with K-Means into 15 groups; manually reviewed the groups and mapped them to readable product categories.
- Built a supervised classification dataset with calendar, cyclical time, elapsed-time, and previous-category features, including a 16-value one-hot encoding for the previous cluster.
- Trained a PyTorch multilayer perceptron and saved its learned weights and `StandardScaler` for reuse during inference; stored the prepared feature data in SQLite.

## Pipeline Overview

1. **Load and embed order names.** The notebook reads `Your Orders/Your Amazon Orders/Order History.csv` and encodes product names into 384-dimensional vectors using `all-MiniLM-L6-v2`.
2. **Generate and review labels.** PCA and K-Means are used to explore product groupings. The selected notebook configuration reduces the embeddings to 7 dimensions and creates 15 clusters. The cluster assignments are manually reviewed and mapped to category names; they are then used as training labels.
3. **Build training features.** Orders are sorted by date. Features include day of week, month, day, hour, weekend flag, cyclical month/hour values, days since the previous order, and a one-hot previous-cluster value. The 26 input features are scaled with a `StandardScaler` fit on the training split.
4. **Train and save.** An 80/20 train/test split is used. The classifier trains for 160 epochs with cross-entropy loss and Adam, then the notebook saves `AmazonNeuralNet.pth` and `scaler.pkl`.
5. **Predict.** `backend.py` collects date/time inputs, builds the same 26-feature row, scales it, and runs the saved model to return one of 15 category labels.

The classifier is a feed-forward neural network, not a Transformer. The Sentence Transformer is used only to create product-name embeddings during label exploration.

## Backend: From Inputs to a Label

`backend.py` and `NNArchitecture.py` implement single-row inference:

1. **Load model weights.** `load_model()` constructs `AmazonNeuralNetwork` (26 inputs, hidden layers of 128 and 64 units, and 15 output logits) and loads the state dictionary from `AmazonNeuralNet.pth`. `map_location` selects the available device.
2. **Collect and encode time features.** The CLI validates weekday, month, day, and hour. It derives the weekend flag and sine/cosine encodings for month and hour.
3. **Look up previous-order context.** The current SQLite query is:

   ```sql
   SELECT Cluster_Encoding, day
   FROM Amazon_DataSet
   WHERE month = {month} AND day <= {day} AND hour <= {hour};
   ```

   The implementation interpolates these validated numeric inputs into the query and calls `fetchone()`. If no row matches, it uses cluster `-1` and `0` days since the previous order; otherwise it uses the returned cluster and subtracts its day from the entered day. It then creates a one-hot encoding in the fixed order `[-1, 0, 1, ..., 14]`.

   **Lookup limitation:** The query does not constrain the result to the actual previous chronological order and has no `ORDER BY`; SQLite may return any matching row. Consequently, previous-order features can be incorrect when multiple rows match.
4. **Scale and classify.** The input row is transformed with `scaler.pkl` and passed to the model in evaluation mode with gradient calculation disabled. `argmax` over the 15 output logits selects the category index, which is translated to a label:

   | Index | Predicted category |
   |---:|---|
   | 0 | Haircare |
   | 1 | Clothing accessories (watches, hats, etc.) |
   | 2 | Shoes or sandals |
   | 3 | Kitchen supplies |
   | 4 | Pantry snacks or kitchen goods |
   | 5 | Skincare |
   | 6 | Vitamins or nutritional supplements |
   | 7 | SD cards, chargers, or cables |
   | 8 | House goods |
   | 9 | Video games |
   | 10 | Shirts |
   | 11 | Styling products or makeup |
   | 12 | Electronics accessories or printer ink |
   | 13 | Women's clothing |
   | 14 | Miscellaneous |

The SQLite table is named `Amazon_DataSet`. The notebook writes the prepared feature DataFrame to `local.db`; the inference query expects this table to exist in the working directory.

## Run Locally

Run commands from the repository root. For inference, install the runtime dependencies and start the interactive CLI:

```bash
pip install torch numpy joblib scikit-learn
python backend.py
```

The notebook's label-generation and training workflow additionally uses `pandas`, `sentence-transformers`, and `matplotlib`.

The backend currently calls `.cuda()` for the input tensor even when its device selection falls back to CPU. As written, inference therefore requires a CUDA-enabled PyTorch environment; CPU-only inference needs that call changed to use the selected device.

## Data and Model Artifacts

The original notebook, order-history export, `local.db`, and `scaler.pkl` are ignored by Git and are not included in a repository clone. `AmazonNeuralNet.pth` is tracked, but inference also needs a matching scaler and SQLite database. Provide or generate those local files before running the backend. Do not commit personal order-history exports or other private account data.