import matplotlib.pyplot as plt

def plotEEG(eeg, channels = range(32), totalTime = 25, title="", fileName=None):
    plt.figure()
    # First dimension for the number of electrodes
    # Second dimension for the data
    assert len(eeg.shape) == 2
    assert eeg.shape[0] == 32

    for i in channels:
        offset = i*1000/len(channels)
        plt.plot([j*totalTime/eeg.shape[1] for j in range(eeg.shape[1])],eeg[i]+offset, color='blue')
        plt.axhline(offset, color='gray', linestyle='--', linewidth=0.5)
        plt.text(-2, offset, f'Ch {i+1}', va='center', ha='right', fontsize=8)
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.gca().set_yticks([])
    plt.gca().set_yticklabels([])
    if (fileName):
        plt.savefig(fileName, dpi=300, bbox_inches='tight')