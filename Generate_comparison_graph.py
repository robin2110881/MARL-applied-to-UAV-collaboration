import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':


    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = os.path.join(SCRIPT_DIR, 'SingleAgent','sa_ppo_training_outputs')
    dates = ["[512, 256, 128, 64]", "[256, 128, 64]", "[64, 32, 16]", "[16, 8, 4]"]#"[1024, 512, 256, 128]", ,"[512, 256, 128]"
    
    score_history, length_history, value_loss_history, entropy_loss_history = [], [], [], []

    for date in dates:
        score_history.append(np.load(os.path.join(output_dir, f"score_history_{date}.npy")))
        length_history.append(np.load(os.path.join(output_dir, f"length_history_{date}.npy")))
        value_loss_history.append(np.load(os.path.join(output_dir, f"value_loss_history_{date}.npy")))
        entropy_loss_history.append(np.load(os.path.join(output_dir, f"entropy_loss_history_{date}.npy")))
    
    len_dates = len(dates)
    for i in range(len_dates):
        dates[i] = "SA " + dates[i]
    

    output_dir = os.path.join(SCRIPT_DIR, 'MultiAgent','ma_ppo_training_outputs')
    dates.extend(["[512, 256, 128, 64]", "[256, 128, 64]", "[64, 32, 16]", "[16, 8, 4]"])

    for i in range(len_dates, len(dates)):
        date = dates[i]
        score_history.append(np.load(os.path.join(output_dir, f"score_history_{date}.npy")))
        length_history.append(np.load(os.path.join(output_dir, f"length_history_{date}.npy")))
        value_loss_history.append(np.load(os.path.join(output_dir, f"value_loss_history_{date}.npy")))
        entropy_loss_history.append(np.load(os.path.join(output_dir, f"entropy_loss_history_{date}.npy")))
    
    for i in range(len_dates, len(dates)):
        dates[i] = "MA " + dates[i]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    for i in range(len(dates)):
        axs[0, 0].plot(score_history[i], alpha=0.3)
        #get the color of the score to amply for the smoothed line
        color = axs[0, 0].lines[-1].get_color()
        axs[0, 1].plot(length_history[i], alpha=0.3, color=color)
        axs[0, 0].plot(np.convolve(score_history[i], np.ones(100)/100, mode='valid'), color=color, label=dates[i])
        axs[0, 1].plot(np.convolve(length_history[i], np.ones(100)/100, mode='valid'), color=color, label=dates[i])

        axs[1, 0].plot(value_loss_history[i], color=color, label=dates[i])
        axs[1, 1].plot(entropy_loss_history[i], color=color, label=dates[i])
    
    axs[0, 0].set_title('PPO Reward Training Progress')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Mean Drone Reward')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    #Add a red line at y=960 
    axs[0, 1].axhline(y=960, color='red', linestyle='--', alpha=0.7)

    axs[0, 1].set_title('PPO Episode Length Training Progress')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Mean Length')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].set_title('PPO Value Loss During Training')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Value Loss')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 1].set_title('PPO Entropy Loss During Training')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Entropy Loss')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
