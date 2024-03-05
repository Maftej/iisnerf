import numpy as np
import statistics
import matplotlib.pyplot as plt

from file_manager.file_manager import FileManager
from data_plot_manager.data_plot_manager import DataPlotManager


class InstantNgpPlotManager:
    def __init__(self):
        self.file_manager = FileManager()

    @staticmethod
    def plot_step_eval_trajectory(plot_data, ylabel, plot_filename):
        fig, ax = plt.subplots()
        resolution_list = ["300x168", "500x280", "720x404", "1920x1080"]
        for i, compression_saving in enumerate(plot_data):
            x = range(1, len(compression_saving)+1)
            y = compression_saving

            ax.step(x, y, label=fr'{resolution_list[i]}')
            ax.plot(x, y, color='grey', alpha=0.3)

        plt.grid(axis='x', color='0.95')
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Indices of the image sequence')
        ax.set_title('Image quality of a custom trajectory')
        plt.savefig(fr"step_plot_eval_trajectory_{plot_filename}.png")
        plt.savefig(fr"step_plot_eval_trajectory_{plot_filename}.pdf")
        plt.show()

    @staticmethod
    def plot_eval_trajectory(plot_data, ylabel, plot_filename):
        x = range(1, len(plot_data) + 1)
        y = plot_data

        fig, ax = plt.subplots()
        markerline, stemlines, baseline = ax.stem(x, y)
        markerline.set_markerfacecolor('none')
        ax.set_xlabel('Indices of the image sequence')
        ax.set_ylabel(ylabel)
        ax.set_title('Custom trajectory')
        plt.savefig(fr"eval_custom_trajectory_{plot_filename}.pdf")
        plt.savefig(fr"eval_custom_trajectory_{plot_filename}.png")
        plt.show()

    @staticmethod
    def plot_eval_models(plot_data, ylabel, plot_filename):
        x = np.array(['1', '2', '4', '8', '16', '32', '64', '128'])
        avg_values = []
        for i in range(len(plot_data)):
            avg_values.append(statistics.mean(plot_data[i]))
            print("avg=", round(sum(plot_data[i]) / len(plot_data[i]), 2))

        pstdev_values = []
        for i in range(len(plot_data)):
            pstdev_values.append(statistics.pstdev(plot_data[i]))
            print("pstdev=", round(statistics.pstdev(plot_data[i]), 2))

        # Create the bar plot with variance
        fig, ax = plt.subplots()
        if ylabel == "SSIM":
            plt.ylim(top=1)
        else:
            plt.ylim(top=40)

        ax.bar(x, np.array(avg_values), yerr=np.array(pstdev_values), capsize=10, color='orange')

        ax.set_xlabel('AABB')
        ax.set_ylabel(ylabel)
        ax.set_title('Instant-NGP models')

        plt.savefig(fr"aabb_scale_{plot_filename}.pdf")
        plt.savefig(fr"aabb_scale_{plot_filename}.png")
        plt.show()

    @staticmethod
    def plot_bar_chart_absolute_sizes(pkt_sizes, plot_filename):
        i_slices = list(map(lambda x: x[0], pkt_sizes))
        p_slices = list(map(lambda x: x[1], pkt_sizes))

        to_kilobytes = 0.001

        i_slices = [to_kilobytes * i_slice for i_slice in i_slices]
        p_slices = [to_kilobytes * p_slice for p_slice in p_slices]

        species = range(0, 159)
        sex_counts = {
            'I-slice': i_slices,
            'P-slice': p_slices,
        }
        width = 0.6  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()
        bottom = np.zeros(len(i_slices))

        for sex, sex_count in sex_counts.items():
            p = ax.bar(species, sex_count, width, label=sex, bottom=bottom)
            bottom += sex_count

        ax.set_title('Sizes of I-slices and P-slices')
        ax.set_ylabel('Size [kB]')
        ax.set_xlabel('Images')
        max_i_slice = max(i_slices)
        max_p_slice = max(p_slices)
        ax.set_ylim(0, (max_i_slice + max_p_slice) + (max_i_slice + max_p_slice) * 0.4)
        ax.legend()

        plt.savefig(fr"{plot_filename}.png")
        plt.savefig(fr"{plot_filename}.pdf")
        plt.show()

    @staticmethod
    def plot_compression_savings(plot_resolution_data, plot_filename):
        fig, ax = plt.subplots()
        resolution_list = ["300x168", "500x280", "720x404", "1920x1080"]
        for i, compression_saving in enumerate(plot_resolution_data):
            x = range(1, len(compression_saving)+1)
            y = compression_saving

            ax.step(x, y, label=fr'{resolution_list[i]}')
            ax.plot(x, y, color='grey', alpha=0.3)

        plt.grid(axis='x', color='0.95')
        ax.legend()
        ax.set_ylabel('Saving [%]')
        ax.set_xlabel('Indices of the image sequence')
        ax.set_title('Compression savings')
        plt.savefig(fr"step_plot_{plot_filename}.png")
        plt.savefig(fr"step_plot_{plot_filename}.pdf")
        plt.show()

    @staticmethod
    def plot_bar_chart_compression_savings(plot_data, plot_filename):
        compression_savings = []
        for values in plot_data:
            temp = []
            for value in values:
                i_slice = value[0]
                p_slice = value[1]
                temp.append(round((i_slice / (i_slice + p_slice)) * 100, 2))
            compression_savings.append(temp)

        x = np.array(['300x168', '500x280', '720x404', '1920x1080'])
        avg_values = []
        for i in range(len(compression_savings)):
            avg_values.append(statistics.mean(compression_savings[i]))
            print("avg=", round(sum(compression_savings[i]) / len(compression_savings[i]), 2))

        pstdev_values = []
        for i in range(len(compression_savings)):
            pstdev_values.append(statistics.pstdev(compression_savings[i]))

        fig, ax = plt.subplots()
        ax.bar(x, np.array(avg_values), yerr=np.array(pstdev_values), capsize=10, color='orange')

        ax.set_xlabel('Resolutions')
        ax.set_ylabel("Saving [%]")
        ax.set_title('Savings using H264 codec')

        plt.savefig(fr"bar_chart_compression_savings_{plot_filename}.pdf")
        plt.savefig(fr"bar_chart_compression_savings_{plot_filename}.png")
        plt.show()


if __name__ == "__main__":
    file_manager = FileManager()

    file_path = r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\test\evaluated_models.json"
    data = file_manager.open_file(file_path)

    InstantNgpPlotManager.plot_eval_models(data["psnr"], "PSNR [dB]", "psnr_factory_robotics")
    InstantNgpPlotManager.plot_eval_models(data["ssim"], "SSIM", "ssim_factory_robotics")

    file_path = r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\eval\factory_robotics_eval\eval_trajectory_res300x168.json"
    data_res300x168 = file_manager.open_file(file_path)

    file_path = r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\eval\factory_robotics_eval\eval_trajectory_res500x280.json"
    data_res500x280 = file_manager.open_file(file_path)

    file_path = r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\eval\factory_robotics_eval\eval_trajectory_res720x404.json"
    data_res720x404 = file_manager.open_file(file_path)

    file_path = r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\eval\factory_robotics_eval\eval_trajectory.json"
    data_res1920x1080 = file_manager.open_file(file_path)

    InstantNgpPlotManager.plot_step_eval_trajectory([data_res300x168["psnr"], data_res500x280["psnr"], data_res720x404["psnr"], data_res1920x1080["psnr"]], "PSNR [dB]", "psnr")
    InstantNgpPlotManager.plot_step_eval_trajectory([data_res300x168["ssim"], data_res500x280["ssim"], data_res720x404["ssim"], data_res1920x1080["ssim"]], "SSIM", "ssim")

    resolution_list = ["300x168", "500x280", "720x404", "1920x1080"]

    resolution_data = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_veryslow_crf_18\pkt_size_res_{single_resolution}_preset_veryslow_crf_18.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_veryslow_crf_18")
        resolution_data.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data), "all_res_preset_veryslow_crf_18")

    resolution_data2 = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_veryslow_crf_23\pkt_size_res_{single_resolution}_preset_veryslow_crf_23.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_veryslow_crf_23")
        resolution_data2.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data2), "all_res_preset_veryslow_crf_23")

    resolution_data3 = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_veryslow_crf_28\pkt_size_res_{single_resolution}_preset_veryslow_crf_28.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_veryslow_crf_28")
        resolution_data3.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data3), "all_res_preset_veryslow_crf_28")

    resolution_data4 = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_medium_crf_18\pkt_size_res_{single_resolution}_preset_medium_crf_18.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_medium_crf_18")
        resolution_data4.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data4), "all_res_preset_medium_crf_18")

    resolution_data5 = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_medium_crf_23\pkt_size_res_{single_resolution}_preset_medium_crf_23.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_medium_crf_23")
        resolution_data5.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data5), "all_res_preset_medium_crf_23")

    resolution_data6 = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_medium_crf_28\pkt_size_res_{single_resolution}_preset_medium_crf_28.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_medium_crf_28")
        resolution_data6.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data6), "all_res_preset_medium_crf_28")

    resolution_data7 = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_veryfast_crf_18\pkt_size_res_{single_resolution}_preset_veryfast_crf_18.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_veryfast_crf_18")
        resolution_data7.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data7), "all_res_preset_veryfast_crf_18")

    resolution_data8 = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_veryfast_crf_23\pkt_size_res_{single_resolution}_preset_veryfast_crf_23.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_veryfast_crf_23")
        resolution_data8.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data8), "all_res_preset_veryfast_crf_23")


    resolution_data9 = []
    for single_resolution in resolution_list:
        print(single_resolution)
        file_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\factory_robotics\videos\factory_robotics_videos\res_{single_resolution}_preset_veryfast_crf_28\pkt_size_res_{single_resolution}_preset_veryfast_crf_28.json"
        data = file_manager.open_file(file_path)
        InstantNgpPlotManager.plot_bar_chart_absolute_sizes(data, f"res_{single_resolution}_preset_veryfast_crf_28")
        resolution_data9.append(data)

    InstantNgpPlotManager.plot_compression_savings(DataPlotManager.process_slices_to_compression_savings(resolution_data9), "all_res_preset_veryfast_crf_28")

    bar_compression_savings_data = []
    len_resolution_data = len(resolution_data7)
    for i in range(len_resolution_data):
        temp = []
        temp = resolution_data[i] + resolution_data2[i] + resolution_data3[i] + resolution_data4[i] + resolution_data5[i] + resolution_data6[i] + resolution_data7[i] + resolution_data8[i] + resolution_data9[i]
        bar_compression_savings_data.append(temp)

    InstantNgpPlotManager.plot_bar_chart_compression_savings(bar_compression_savings_data, "all_res")


    bar_compression_savings_data = []
    all_resolution_data = [resolution_data, resolution_data2, resolution_data3, resolution_data4, resolution_data5, resolution_data6, resolution_data7, resolution_data8, resolution_data9]
    all_processed_resolution_data = []
    for single_data in all_resolution_data:
        all_processed_resolution_data.append(DataPlotManager.process_slices_to_compression_savings(single_data))

    print("LEN=", len(all_processed_resolution_data))
    print("LEN2=", len(all_processed_resolution_data[0]))
    print(all_processed_resolution_data[0])

    res_300x168 = []
    res_500x280 = []
    res_720x404 = []
    res_1920_1080 = []
    output = []
    for single_resolution_data in all_processed_resolution_data:
        res_300x168.append(single_resolution_data[0])
        res_500x280.append(single_resolution_data[1])
        res_720x404.append(single_resolution_data[2])
        res_1920_1080.append(single_resolution_data[3])

    first_result = []
    all_resolutions = [res_300x168, res_500x280, res_720x404, res_1920_1080]
    output = []
    for single_res in all_resolutions:
        first_result = []
        for i in range(159):
            temp = []
            for single_first in single_res:
                temp.append(single_first[i])
            first_result.append(round(sum(temp) / len(temp), 2))
        output.append(first_result)
    print("*****")
    print(len(output))
    print(len(output[0]))
    print("*****")

    InstantNgpPlotManager.plot_compression_savings(output, "all_res_averaged")
