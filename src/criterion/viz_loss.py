import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.ndimage import uniform_filter1d

def smooth_loss(loss, window=151):
    """对 per-token loss 做滑动平均，边界用边缘复制"""
    loss = np.asarray(loss, dtype=float)
    return uniform_filter1d(loss, size=window, mode='nearest')

def plot_loss(loss_dir, phage_region_path=None, resolution=500, kmer_size=3, stride=1):
    loss_path = os.path.join(loss_dir, 'loss.npy')
    # perplexity_path = os.path.join(loss_dir, 'perplexity.npy')

    loss = np.load(loss_path)
    if loss.ndim > 1:
        loss = loss.squeeze()
        np.save(loss_path, loss)  # Save the squeezed loss back to the file
    # perplexity = np.load(perplexity_path)
    # if perplexity.ndim > 1:
    #     perplexity = perplexity.squeeze()
    #     np.save(perplexity_path, perplexity)  # Save the squeezed perplexity back to the file
    
    length = len(loss)
    window_size = length // resolutuion 
    loss = smooth_loss(loss, window=window_size)
    loss_mean = np.mean(loss)
    loss_std = np.std(loss)
    # perplexity = smooth_loss(perplexity, window=window_size)
    # perplexity_mean = np.mean(perplexity)
    # perplexity_std = np.std(perplexity)

    if phage_region_path is not None:
        phage_regions = pd.read_csv(phage_region_path, header=0)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(loss, label='Loss')
    ax1.set_ylim([loss_mean - 3 * loss_std, loss_mean + 3 * loss_std])
    ax1.set_xlabel('Token Index')
    ax1.set_ylabel('Loss')
    ax1.set_title('Per-token Loss')
    ax1.legend()
    ax1.grid()
    if phage_region_path is not None:
        for _, row in phage_regions.iterrows():
            ax1.axvspan(row[1], row[2], color='red', alpha=0.3)
            ax1.text((row[1] + row[2]) / 2, loss_mean + 2*loss_std, row[0], color='red', ha='center', va='bottom')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(loss, label='loss (log-scale)')
    ax2.set_ylim([loss_mean - 3 * loss_std, loss_mean + 3 * loss_std])
    ax2.set_yscale('log')
    ax2.set_xlabel('Token Index')
    ax2.set_ylabel('Loss')
    ax2.set_title('Per-token Loss (log-scale)')
    ax2.legend()
    ax2.grid()
    if phage_region_path is not None:
        for _, row in phage_regions.iterrows():
            ax2.axvspan(row[1], row[2], color='red', alpha=0.3)
            ax2.text((row[1] + row[2]) / 2, loss_mean + 2*loss_std, row[0], color='red', ha='center', va='bottom')

    # fig.tight_layout()
    fig.savefig(os.path.join(loss_dir, f'loss_perplexity_plot_smoothed_res_{resolution}.png'))

def main():
    tokenizer = 'kmer3'
    # plot_loss(f'/srv/scratch/z3543429/LLM/TriEvo/dataset/bacteria/NC_000964.3/phage_seqs/phage_pbsx.fasta.{tokenizer}')
    # plot_loss(f'/srv/scratch/z3543429/LLM/TriEvo/dataset/bacteria/NC_000964.3/phage_seqs/phage_spbeta.fasta.{tokenizer}')
    plot_loss(
        f'/srv/scratch/z3543429/LLM/TriEvo/dataset/bacteria/NC_000964.3/sequence.fasta.{tokenizer}', 
        phage_region_path='/srv/scratch/z3543429/LLM/TriEvo/dataset/bacteria/NC_000964.3/phage_regions.csv',
        resolution=500,
        kmer_size=3,
        stride=1
    )

if __name__ == "__main__":
    main()