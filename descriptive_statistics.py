from data import getData
X, y = getData()
def descriptive_statistics(data):
    # Importing the required libraries
    
    import numpy as np
    from scipy import stats
    # ignore all warnings

    for i in range(4):
        print("-----------------")
        mean = np.mean(data[:,i])
        median = np.median(data[:,i])
        mode = stats.mode(data[:,i])
        std_dev = np.std(data[:,i])
        variance = np.var(data[:,i])
        percentiles = np.percentile(data[:,i], [25, 50, 75])
        min_value = np.min(data[:,i])
        max_value = np.max(data[:,i])
        range_value = np.ptp(data[:,i])
        
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Mode: {mode}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Variance: {variance}")
        print(f"25th Percentile: {percentiles[0]}")
        print(f"50th Percentile (Median): {percentiles[1]}")
        print(f"75th Percentile: {percentiles[2]}")
        print(f"Minimum: {min_value}")
        print(f"Maximum: {max_value}")
        print(f"Range: {range_value}")
    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data).mode[0]
    std_dev = np.std(data)
    variance = np.var(data)
    percentiles = np.percentile(data, [25, 50, 75])
    min_value = np.min(data)
    max_value = np.max(data)
    range_value = np.ptp(data)
    
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print(f"25th Percentile: {percentiles[0]}")
    print(f"50th Percentile (Median): {percentiles[1]}")
    print(f"75th Percentile: {percentiles[2]}")
    print(f"Minimum: {min_value}")
    print(f"Maximum: {max_value}")
    print(f"Range: {range_value}")
descriptive_statistics(X)