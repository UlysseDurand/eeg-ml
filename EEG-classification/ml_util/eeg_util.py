import matplotlib.pyplot as plt

def plotEEG(eeg, channels = range(32), totalTime = 25, title="", fileName=None, scaler=1):
    plt.figure()

    for i in channels:
        offset = i*1000/len(channels)
        plt.plot([j for j in range(eeg.shape[1])],eeg[i]*scaler+offset, color='blue')
        plt.axhline(offset, color='gray', linestyle='--', linewidth=0.5)
        plt.text(-2, offset, f'Ch {i+1}', va='center', ha='right', fontsize=8)
    plt.xlabel('Time units')
    plt.title(title)
    plt.gca().set_yticks([])
    plt.gca().set_yticklabels([])
    if (fileName):
        plt.savefig(fileName, dpi=300, bbox_inches='tight')