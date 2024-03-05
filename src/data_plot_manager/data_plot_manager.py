
class DataPlotManager:
    def __init__(self):
        pass

    @staticmethod
    def process_slices_to_compression_savings(plot_resolution_data):
        compression_savings = []
        for values in plot_resolution_data:
            temp = []
            for value in values:
                i_slice = value[0]
                p_slice = value[1]
                temp.append(round((i_slice / (i_slice + p_slice)) * 100, 2))
            compression_savings.append(temp)
        return compression_savings
