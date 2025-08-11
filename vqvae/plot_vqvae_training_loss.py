import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_losses(log_dir):
    """
    Plots multiple training and validation loss curves from CSV files in the given directory.
    Saves the plot as 'all_losses.png' in the same directory.
    """

    file_configs = [
        ("train_total_loss.csv", "Train Total Loss", "red",  "-", None),
        ("val_total_loss.csv",   "Val Total Loss",   "red",  "--", "^"),
        ("train_recon_loss.csv", "Train Recon Loss", "blue", "-", None),
        ("val_recon_loss.csv",   "Val Recon Loss",   "blue", "--", "^"),
        ("train_vq_loss.csv",    "Train VQ Loss",    "green","-", None),
        ("val_vq_loss.csv",      "Val VQ Loss",      "green","--", "^"),
        ("train_lpips_loss.csv", "Train LPIPS Loss", "orange", "-", None),
        ("val_lpips_loss.csv",   "Val LPIPS Loss",   "orange", "--", "^"),
    ]
    
    # plt.figure(figsize=(12, 6))

    # for filename, label, color, linestyle, marker in file_configs:
    #     path = os.path.join(log_dir, filename)
    #     if not os.path.exists(path):
    #         print(f"[Warning] File not found: {path}")
    #         continue

    #     df = pd.read_csv(path)
    #     df['Epoch'] = range(1, len(df) + 1)
    #     plt.plot(df['Epoch'], df['Value'], label=label, color=color,
    #              linestyle=linestyle, marker=marker, linewidth=2, markersize=5)

    # plt.xlabel("Epoch", fontsize=20)
    # plt.ylabel("Loss", fontsize=20)
    # plt.title("v128_z16_c64_b1_1248 Compound Loss", fontsize=24)
    # plt.ylim(0, 0.70)
    # plt.grid(True)
    # plt.legend(fontsize=16, loc='upper right')
    # plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # # Save plot
    # save_path = os.path.join(log_dir, "compound_loss_v128_z16_c64_b1_lpips.png")
    # plt.savefig(save_path)
    # print(f"Saved plot to {save_path}")
    # plt.close()
    
    # Plot the following compound loss separately
    compound_file_configs = [
        ("train_total_loss.csv", "Train Total Loss", "red",  "-", None),
        ("val_total_loss.csv",   "Val Total Loss",   "red",  "--", "^"),
        ("train_recon_loss.csv", "Train Recon Loss", "blue", "-", None),
        ("val_recon_loss.csv",   "Val Recon Loss",   "blue", "--", "^"),
        # ("train_vq_loss.csv",    "Train VQ Loss",    "green","-", None),
        # ("val_vq_loss.csv",      "Val VQ Loss",      "green","--", "^"),
        ("train_lpips_loss.csv", "Train LPIPS Loss", "orange", "-", None),
        ("val_lpips_loss.csv",   "Val LPIPS Loss",   "orange", "--", "^"),
        
        ("train_codebook_loss.csv", "Train Codebook Loss", "purple", "-", None),
        ("val_codebook_loss.csv",   "Val Codebook Loss",   "purple", "--", "^"),
        ("train_commitment_loss.csv", "Train Commit Loss", "cyan", "-", None),
        ("val_commitment_loss.csv",   "Val Commit Loss",   "cyan", "--", "^"),
        ("train_vq_loss.csv",    "Train VQ Loss",    "green","-", None),
        ("val_vq_loss.csv",      "Val VQ Loss",      "green","--", "^"),
    ]

    plt.figure(figsize=(6, 8))

    for filename, label, color, linestyle, marker in compound_file_configs:
        path = os.path.join(log_dir, filename)
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            continue

        df = pd.read_csv(path)
        df['Epoch'] = range(1, len(df) + 1)
        plt.plot(df['Epoch'], df['Value'], label=label, color=color,
                 linestyle=linestyle, marker=marker, linewidth=2, markersize=5)

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title("VQ-VAE Compound Losses", fontsize=20)
    plt.ylim(0, 0.70)
    plt.grid(True)
    plt.legend(fontsize=12, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(log_dir, "compound_losses.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

def main():
    log_directory = "/home/yuchenliu/VAR/local_output/vqvae_checkpoints_v128_z16_c64_b1_lpips/multiscale_vqvae_loss"
    plot_all_losses(log_directory)

if __name__ == "__main__":
    main()
