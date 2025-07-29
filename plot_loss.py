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
    ]

    plt.figure(figsize=(12, 6))

    for filename, label, color, linestyle, marker in file_configs:
        path = os.path.join(log_dir, filename)
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            continue

        df = pd.read_csv(path)
        df['Epoch'] = range(1, len(df) + 1)
        plt.plot(df['Epoch'], df['Value'], label=label, color=color,
                 linestyle=linestyle, marker=marker, linewidth=2, markersize=5)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("v128_z16_c64_b1_1248 Compound Loss", fontsize=20)
    plt.ylim(0, 0.30)
    plt.grid(True)
    plt.legend(fontsize=12, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(log_dir, "compound_loss_v128_z16_c64_b1.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

def main():
    log_directory = "/home/yuchenliu/VAR/local_output/vqvae_checkpoints_v128_z16_c64_b1/multiscale_vqvae_loss"
    plot_all_losses(log_directory)

if __name__ == "__main__":
    main()
