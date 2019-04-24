

def TBR(signal, factor=1):
    '''
    Threshold Based Representation
    Converts signal into spike train.
    INPUT: signal <np.ndarray>, factor <float>
    OUTPUT: spikes <np.ndarray>, threshold <float>
    '''
    import numpy as np
    signal_length = len(signal)
    diff = np.zeros(signal_length)
    
    for t in range(signal_length-1):
        diff[t+1] = signal[t+1] - signal[t]
    
    threshold = np.mean(diff) + factor * np.std(diff)
    
    spikes = np.zeros(signal_length)
    
    for t in range(signal_length):
        if diff[t] > threshold:
            spikes[t] = 1
        elif diff[t] < -threshold:
            spikes[t] = -1
            
    return spikes, threshold

def TBR_decode(spikes, initial_val, threshold=1):
    '''
    Threshold based representation
    INPUT: spikes <np.ndarray>, initial_val<float>, threshold <float>
    OUTPUT: signal <np.ndarray>
    '''
    import numpy as np
    signal_length = len(spikes)

    signal = np.full(signal_length, initial_val)

    for t in range(1,signal_length):
    
        if spikes[t] ==1:
            signal[t] = signal[t-1] + threshold
    
        elif spikes[t]==-1:
            signal[t] = signal[t-1] - threshold
    
        else:
            signal[t] = signal[t-1]
        
    return signal

def BSA(signal, filter, threshold=1):
    '''
    Ben's Spiking Algorithm
    Converts signal into spike train.
    INPUT: signal <np.ndarray>, filter <np.ndarray>, threshold <float>
    OUTPUT: spikes <np.ndarray>
    '''
    # from scipy.signal import firwin
    # filter = firwin(win_size,cutoff_freq)
    import numpy as np
    
    signal_length = len(signal)
    filter_length = len(filter)
    spikes = np.zeros(signal_length)

    for i in range(1,signal_length):
        
        error1 = 0
        error2 = 0
        
        for j in range(1,filter_length):
            if i+j-2 < signal_length:
                error1 += abs(signal[i+j-2]-filter[j])
                error2 += abs(signal[i+j-2])
        
        if error1 <= (error2-threshold):
            spikes[i] = 1
            for j in range(1,filter_length):
                if i+j-2 < signal_length:
                    signal[i+j-2] -= filter[j]

    return spikes


def BSA_decode(spikes, filter, initial_val=0):
    '''
    Threshold based representation
    INPUT: spikes <np.ndarray>, filter <np.ndarray>, initial_val <float>
    OUTPUT: signal <np.ndarray>
    '''
    import numpy as np

    signal_length = len(spikes)
    filter_length = len(filter)
    signal = np.full(signal_length, initial_val)

    for t in range(1,signal_length-filter_length+1):
        if spikes[t]==1:
            for j in range(1,filter_length):
                if t+j-2 < signal_length:
                    signal[t+j-2] += filter[j]
        
    return signal

def MW(signal, window, threshold=1):
    '''
    Moving Window
    Converts signal into spike train.
    INPUT: signal <np.ndarray>, window <float>, threshold <float>
    OUTPUT: spikes <np.ndarray>
    '''
    import numpy as np
    signal_length = len(signal)
    spikes = np.zeros(signal_length)
    base = np.mean(signal[:window+1])
    
    for t in range(window+1):
    
        if signal[t] > base+threshold:
            spikes[t] = 1
    
        elif signal[t] < base-threshold:
            spikes[t] = -1
    
    for t in range(window+1,signal_length):
    
        base = np.mean(signal[t-window-1:t-1]);
    
        if signal[t] > base+threshold:
            spikes[t] = 1
    
        elif signal[t] < base-threshold:
            spikes[t] = -1
    
    return spikes

def MW_decode(spikes, window, initial_val, threshold=1):
    '''
    Moving Window
    INPUT: spikes <np.ndarray>, window <float>, initial_val<float>, threshold <float>
    OUTPUT: signal <np.ndarray>
    '''
    import numpy as np

    signal_length = len(spikes)
    signal = np.zeros(signal_length)
    signal[0] = initial_val

    for t in range(1,signal_length):
    
        if spikes[t] == 1:
            signal[t] = signal[t-1] + threshold
    
        elif spikes[t] == -1:
            signal[t] = signal[t-1] - threshold
    
        else:
            signal[t] = signal[t-1]
        
    return signal

def SF(signal, threshold=1):
    '''
    Step Forward
    Converts signal into spike train.
    INPUT: signal <np.ndarray>, threshold <float>
    OUTPUT: spikes <np.ndarray>
    '''
    import numpy as np
    signal_length = len(signal)
    spikes = np.zeros(signal_length)
    base=signal[0]
    
    for t in range(1,signal_length):
        
        if signal[t] > base+threshold:
            spikes[t] = 1
            base += threshold
        
        elif signal[t] < base-threshold:
            spikes[t] = -1;
            base -= threshold

    return spikes

def SF_decode(spikes, initial_val, threshold=1):
    '''
    Step Forward
    INPUT: spikes <np.ndarray>, initial_val<float>, threshold <float>
    OUTPUT: signal <np.ndarray>
    '''
    import numpy as np
    
    signal_length = len(spikes)
    signal = np.zeros(signal_length)
    signal[0] = initial_val

    for t in range(1,signal_length):
    
        if spikes[t] == 1:
            signal[t] = signal[t-1] + threshold
    
        elif spikes[t] == -1:
            signal[t] = signal[t-1] - threshold
    
        else:
            signal[t] = signal[t-1]
        
    return signal
