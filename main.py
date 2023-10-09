import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
import urllib.request

from dataset import NSMCDataset


def train():
    losses = []
    accuracies = []

    for i in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        batches = 0

        model.train()

        for input_ids_batch, attention_masks_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_batch = y_batch.to(device)
            y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
            loss = F.cross_entropy(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y_batch).sum()
            total += len(y_batch)

            batches += 1
            if batches % 100 == 0:
                print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)

        losses.append(total_loss)
        accuracies.append(correct.float() / total)
        print("Train Loss:", total_loss, "Accuracy:", correct.float() / total)

def eval():

    model.eval()

    test_correct = 0
    test_total = 0

    for input_ids_batch, attention_masks_batch, y_batch in test_loader:
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        _, predicted = torch.max(y_pred, 1)
        test_correct += (predicted == y_batch).sum()
        test_total += len(y_batch)

    print("Accuracy:", test_correct.float() / test_total)


if __name__=='__main__':
    # -------------------------- dataset_download-------------------------------
    url1 = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
    url2 = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"

    local_path1 = "/storage/sjpark/NSMC/test.txt"
    local_path2 = "/storage/sjpark/NSMC/train.txt"

    # urllib.request.urlretrieve(url1, local_path1)
    # urllib.request.urlretrieve(url2, local_path2)
    # --------------------------------------------------------------------------

    gpu_id = '1'
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

    train_dataset = NSMCDataset("/storage/sjpark/NSMC/train.txt")
    test_dataset = NSMCDataset("/storage/sjpark/NSMC/test.txt")

    model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)

    epochs = 5
    batch_size = 16

    optimizer = AdamW(model.parameters(), lr=5e-6)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    train()
    eval()

