import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from Bio import SeqIO

from trievo.nn_utils import cross_entropy_loss
from trievo.tokenizer import KmerTokenizer


def predict_fasta(
    model, tokenizer, 
    input_fasta_path: str, 
    output_dir: str, 
    max_length, 
    device,
):
    # read fasta
    record = SeqIO.read(input_fasta_path, "fasta")

    # decode 
    tokenized_input = np.array(tokenizer.encode(str(record.seq)))

    total_length = len(tokenized_input)
    print(f"Total length of input sequence: {total_length}")

    model.eval()
    loss_fn = cross_entropy_loss

    # predict 
    if total_length - 1 > max_length:
        # use sliding window
        start_indices = list(np.arange(0, total_length, step=max_length))
        losses = []
        for start in start_indices:
            end = min(start + max_length, total_length)
            x = torch.from_numpy(tokenized_input[start:end-1]).long().to(device, non_blocking=True).unsqueeze(0) # add batch dimension
            y = torch.from_numpy(tokenized_input[start+1:end]).long().to(device, non_blocking=True).unsqueeze(0) # add batch dimension
            with torch.no_grad():
                logits = model(x)
                loss = loss_fn(logits, y, agg_mean=False)
                losses.append(loss)
        loss = np.concatenate([l.cpu().numpy() for l in losses], axis=1)
        perplexity = np.exp(loss)
        
    else: 
        x = torch.from_numpy(tokenized_input[:-1]).long().to(device, non_blocking=True).unsqueeze(0) # add batch dimension
        y = torch.from_numpy(tokenized_input[1:]).long().to(device, non_blocking=True).unsqueeze(0) # add batch dimension
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y, agg_mean=False).cpu().numpy()
        perplexity = np.exp(loss)

    # write to output file
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'loss.npy'), np.array(loss.squeeze()))
    np.save(os.path.join(output_dir, 'perplexity.npy'), np.array(perplexity.squeeze()))

def main():

    pass

if __name__ == "__main__":
    pass