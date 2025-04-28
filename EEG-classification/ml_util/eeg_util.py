import matplotlib.pyplot as plt

def plotEEG(eeg, channels = range(32), totalTime = 25, title=""):
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
    plt.title('EEG of '+title)
    plt.gca().set_yticks([])
    plt.gca().set_yticklabels([])
    plt.show()

def plotTimeFreqEEG(sample):
    fig, axes = plt.subplots(32, 1)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.imshow(sample[i], cmap='plasma')
        ax.axis('off')  # Hide axes
        
    # plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig, axes