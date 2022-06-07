from joblib import load, dump, Parallel, delayed

def get_n_spilt_index(data, n_spilit=5):
    index_list_in = np.arange((len(data)))
    output_index_out = []
    index_len = len(index_list_in)
    step = round(index_len/n_spilit)
    for i in range(n_spilit):
        output_index_out.append(i*step)
    return output_index_out