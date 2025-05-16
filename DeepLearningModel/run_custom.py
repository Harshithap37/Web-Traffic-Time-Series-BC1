import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_custom import StreamingInformerDataset
from models.model import Informer  # assuming Informer model is defined here
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import gc
import os
# from tqdm import tqdm


def save_predictions_csv(preds, trues, sample_ids, filename='predictions_20.csv'):
    rows = []
    for i in range(len(sample_ids)):
        for t in range(preds.shape[1]):
            rows.append({
                'Sample': sample_ids[i],
                'Timestep': t,
                'Prediction': preds[i, t],
                'Actual': trues[i, t]
            })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✅ Saved predictions to {filename}")

def main():
    # ---------- CLEAN GPU MEMORY ---------- #
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cleared GPU cache.")

    # ---------- CONFIGURATION ---------- #
    data_path_train = 'informer_train_streamed.pkl'
    data_path_val = 'informer_val_streamed.pkl'
    data_path_test = 'informer_test_streamed.pkl'

    enc_len = 96
    dec_len = 24
    batch_size = 64
    epochs = 15
    learning_rate = 0.0005
    patience = 2

    # ---------- LOAD DATA ---------- #
    print("\n>> Indexing streamed training dataset...")
    train_dataset = StreamingInformerDataset(data_path_train, enc_len, dec_len)

    print(">> Indexing streamed validation dataset...")
    val_dataset = StreamingInformerDataset(data_path_val, enc_len, dec_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- MODEL SETUP ---------- #
    model = Informer(enc_in=1, dec_in=1, c_out=1, seq_len=enc_len, label_len=0, out_len=dec_len,
                     d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, dropout=0.05,
                     attn='full', embed='fixed', freq='d', activation='gelu', output_attention=False)
    # model = model.cuda() if torch.cuda.is_available() else model
    model = model.cuda() if torch.cuda.is_available() else model

    resume_path = "best_model_20.pth"
    if os.path.exists(resume_path):
        print(f"🔁 Resuming training from checkpoint: {resume_path}")
        model.load_state_dict(torch.load(resume_path))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_rmses, val_smapes = [], [], []
    best_smape = float('inf')
    bad_epoch_count = 0

    for epoch in range(epochs):
        print(f"\n▶️ Starting Epoch {epoch+1}/{epochs}...")
        start_time = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (x_enc, x_dec, y) in enumerate(train_loader):
            x_enc = x_enc.unsqueeze(-1).float()
            x_dec = x_dec.unsqueeze(-1).float()
            if isinstance(y, list):
                y = torch.tensor(y, dtype=torch.float32)

            if y.dim() == 2:  # shape: (batch_size, dec_len)
                y = y.unsqueeze(-1)  # -> (batch_size, dec_len, 1)

            y = y.float()

            if torch.cuda.is_available():
                x_enc, x_dec, y = x_enc.cuda(), x_dec.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x_enc, x_mark_enc=None, x_dec=x_dec, x_mark_dec=None)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"✅ Epoch {epoch+1} completed in {time.time() - start_time:.2f}s | Train Loss: {avg_train_loss:.6f}")

        # ---------- VALIDATION ---------- #
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x_enc, x_dec, y in val_loader:
                x_enc = x_enc.unsqueeze(-1).float()
                x_dec = x_dec.unsqueeze(-1).float()
                y = y.unsqueeze(-1).float()

                if torch.cuda.is_available():
                    x_enc, x_dec, y = x_enc.cuda(), x_dec.cuda(), y.cuda()

                out = model(x_enc, x_mark_enc=None, x_dec=x_dec, x_mark_dec=None)

                preds.append(out.cpu().numpy())
                trues.append(y.cpu().numpy())

        preds = np.concatenate(preds, axis=0).reshape(-1, dec_len)
        trues = np.concatenate(trues, axis=0).reshape(-1, dec_len)

        rmse = np.sqrt(mean_squared_error(trues, preds))
        denominator = np.clip(np.abs(preds) + np.abs(trues), 1e-6, None)
        smape = np.mean(2 * np.abs(preds - trues) / denominator) * 100
        # smape = np.mean(2 * np.abs(preds - trues) / (np.abs(preds) + np.abs(trues))) * 100

        val_rmses.append(rmse)
        val_smapes.append(smape)
        print(f"📊 Validation Results — RMSE: {rmse:.4f}, SMAPE: {smape:.2f}%")

        if smape < best_smape:
            best_smape = smape
            torch.save(model.state_dict(), 'best_model_20.pth')
            print("💾 Best model saved.")
            bad_epoch_count = 0
        else:
            bad_epoch_count += 1

        print(f"[Epoch {epoch+1}] Best SMAPE so far: {best_smape:.2f}%, Bad Epochs: {bad_epoch_count}/{patience}")

        if bad_epoch_count >= patience:
            print("⏹️ Early stopping triggered. Stopping training.")
            break

    # ---------- PLOT LEARNING CURVES ---------- #
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_rmses, label='Validation RMSE')
    plt.plot(epochs_range, val_smapes, label='Validation SMAPE')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves_20.png')
    plt.show()

    # ---------- FINAL TEST EVALUATION ---------- #
    print("\n>> Running final evaluation on test set...")
    test_dataset = StreamingInformerDataset(data_path_test, enc_len, dec_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model.load_state_dict(torch.load('best_model_20.pth'))
    model.eval()
    test_preds, test_trues, test_ids = [], [], []

    with torch.no_grad():
        for i, (x_enc, x_dec, y) in enumerate(test_loader):
            x_enc = x_enc.unsqueeze(-1).float()
            x_dec = x_dec.unsqueeze(-1).float()
            y = y.unsqueeze(-1).float()

            if torch.cuda.is_available():
                x_enc, x_dec, y = x_enc.cuda(), x_dec.cuda(), y.cuda()

            out = model(x_enc, x_mark_enc=None, x_dec=x_dec, x_mark_dec=None)
            test_preds.append(out.cpu().numpy())
            test_trues.append(y.cpu().numpy())
            test_ids.extend([f"sample_{i * batch_size + j}" for j in range(x_enc.shape[0])])

    test_preds = np.concatenate(test_preds, axis=0).reshape(-1, dec_len)
    test_trues = np.concatenate(test_trues, axis=0).reshape(-1, dec_len)

    test_rmse = np.sqrt(mean_squared_error(test_trues, test_preds))
    test_smape = np.mean(2 * np.abs(test_preds - test_trues) / (np.abs(test_preds) + np.abs(test_trues))) * 100
    print(f"Test RMSE: {test_rmse:.4f}, SMAPE: {test_smape:.2f}%")

    save_predictions_csv(test_preds[:10], test_trues[:10], test_ids[:10])


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
