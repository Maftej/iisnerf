import matplotlib.pyplot as plt
import numpy as np
import statistics


from file_manager.file_manager import FileManager
from render_manager.render_manager import RenderManager


class DNeRFPlotManager:
    def __init__(self):
        pass

    @staticmethod
    def plot_eval_trajectory(plot_data, ylabel, plot_filename):
        fig, ax = plt.subplots()
        resolution_list = ["train", "test", "val"]
        for i, compression_saving in enumerate(plot_data):
            x = range(1, len(compression_saving) + 1)
            y = compression_saving

            ax.bar(x, y, label=fr'{resolution_list[i]}')
            ax.legend()

        plt.grid(axis='x', color='0.95')
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Indices of the image sequence')
        ax.set_title('Image quality of disparity maps')
        plt.savefig(fr"../plots/plot_dnerf_mytest_{plot_filename}.png")
        plt.savefig(fr"../plots/plot_dnerf_mytest_{plot_filename}.pdf")
        plt.show()

    @staticmethod
    def plot_step_eval_trajectory(plot_data, ylabel, plot_filename):
        fig, ax = plt.subplots()
        resolution_list = ["train", "test", "val"]
        for i, compression_saving in enumerate(plot_data):
            x = range(1, len(compression_saving) + 1)
            y = compression_saving

            ax.step(x, y, label=fr'{resolution_list[i]}')
            ax.plot(x, y, color='grey', alpha=0.3)

        plt.grid(axis='x', color='0.95')
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Indices of the image sequence')
        ax.set_title('Image quality')
        plt.savefig(fr"../plots/step_plot_dnerf_{plot_filename}.png")
        plt.savefig(fr"../plots/step_plot_dnerf_{plot_filename}.pdf")
        plt.show()

    @staticmethod
    def plot_eval_models(plot_data, ylabel, plot_filename):
        x = np.array(['train', 'test', 'val'])
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

        ax.bar(x, np.array(avg_values), yerr=np.array(pstdev_values), capsize=10,
               color=['#cccccc', '#38b5fd', '#ffd329'])

        ax.set_xlabel('Datasets')
        ax.set_ylabel(ylabel)
        ax.set_title('Image quality of disparity maps')

        plt.savefig(fr"../plots/plot_dnerf_{plot_filename}.pdf")
        plt.savefig(fr"../plots/plot_dnerf_{plot_filename}.png")
        plt.show()

    @staticmethod
    def violin_plot(data, ylabel, plot_filename):
        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value

        labels = ['train', 'test', 'val']
        colors = ['#cccccc', '#38b5fd', '#ffd329']

        violin_parts = plt.violinplot(data, showmedians=True, showextrema=False)

        # Customize the appearance of the violin plot components
        # for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        #     violin_parts[partname].set_edgecolor('#4d4d4dff')

        violin_parts['cmedians'].set_edgecolor('#4d4d4dff')
        # violin_parts['cmedians'].set_linewidth(4)
        # violin_parts['cmedians'].set_color('red')
        # violin_parts['cmedians'].set_alpha(0.7)

        for i, pc in enumerate(violin_parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor("#4d4d4dff")
            pc.set_linewidth(0.5)
            pc.set_alpha(0.7)

        quartile1 = []
        medians = []
        quartile3 = []
        for single_data in data:
            my_q1, my_mdn, my_q3 = np.percentile([single_data], [25, 50, 75], axis=1)
            quartile1.append(my_q1[0])
            medians.append(my_mdn[0])
            quartile3.append(my_q3[0])

        # quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
        # whiskers = np.array([
        #     adjacent_values(sorted_array, q1, q3)
        #     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        # whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        average_means = []
        for single_data in data:
            average_means.append(np.mean(single_data))

        inds = np.arange(1, len(medians) + 1)
        plt.scatter(inds, average_means, marker='o', color='#38b5fd', s=15, zorder=3)
        plt.vlines(inds, quartile1, quartile3, color='#4d4d4dff', linestyle='-', lw=5)
        # plt.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        plt.xlabel('Datasets')
        plt.ylabel(ylabel)
        plt.title('Image quality of disparity maps')
        plt.xticks(np.arange(1, len(labels) + 1), labels)

        plt.savefig(fr"../plots/violin_plot_dnerf_{plot_filename}.pdf")
        plt.savefig(fr"../plots/violin_plot_dnerf_{plot_filename}.png")
        plt.show()

    @staticmethod
    def plot_rgb_map(img, arm_robot_id):
        fig, ax = plt.subplots()
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        img = img[200: 625, 220: 620]

        plt.imshow(img)

        plt.savefig(fr"../plots/arm_robot_rgb_map_normalized_{arm_robot_id}.png")
        plt.savefig(fr"../plots/arm_robot_rgb_map_normalized_{arm_robot_id}.pdf")
        plt.show()

    @staticmethod
    def plot_acc_map(normalized_heatmap, arm_robot_id):
        fig, ax = plt.subplots()
        heatmap = ax.imshow(normalized_heatmap, cmap='viridis')

        cbar = plt.colorbar(heatmap)
        cbar.ax.yaxis.set_label_position('right')
        cbar.set_label('Opacity')

        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        plt.savefig(fr"../plots/arm_robot_acc_map_normalized_{arm_robot_id}.png")
        plt.savefig(fr"../plots/arm_robot_acc_map_normalized_{arm_robot_id}.pdf")
        plt.show()

    @staticmethod
    def plot_disp_map(normalized_heatmap, arm_robot_id):
        fig, ax = plt.subplots()
        heatmap = ax.imshow(normalized_heatmap, cmap='viridis')

        # Add colorbar on the right side
        cbar = plt.colorbar(heatmap)
        cbar.ax.yaxis.set_label_position('right')
        cbar.set_label('Depth')

        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        plt.savefig(fr"../plots/arm_robot_disparity_map_normalized_{arm_robot_id}.png")
        plt.savefig(fr"../plots/arm_robot_disparity_map_normalized_{arm_robot_id}.pdf")
        plt.show()

    @staticmethod
    def plot_all_maps(single_scenario):
        # time = 0.05  # in [0,1]
        # azimuth = -90  # in [-180,180]
        # elevation = -20  # in [-180,180]
        time = float(single_scenario["time"])
        azimuth = int(single_scenario["azimuth"])
        elevation = int(single_scenario["elevation"])
        render_manager = RenderManager()

        arm_robot_id = 0
        while time <= 1.0:
            img, disp, acc, depth = render_manager.render_maps(time, azimuth, elevation)

            DNeRFPlotManager.plot_rgb_map(img, arm_robot_id)

            masked = render_manager.process_acc_map(acc)
            normalized_heatmap = render_manager.create_normalized_heatmap(masked)
            normalized_heatmap = normalized_heatmap[200:625, 220:620]
            DNeRFPlotManager.plot_acc_map(normalized_heatmap, arm_robot_id)

            masked = render_manager.process_disp_map(disp, acc)
            normalized_heatmap = render_manager.create_normalized_heatmap(masked)
            normalized_heatmap = normalized_heatmap[200:625, 220:620]

            DNeRFPlotManager.plot_disp_map(normalized_heatmap, arm_robot_id)

            time = time + 0.9
            arm_robot_id = arm_robot_id + 1

    @staticmethod
    def plot_psnr_ssim_data(single_scenario):
        data_path = single_scenario["data_path"]
        file_manager = FileManager()
        data = file_manager.open_file(data_path)["data"]
        train_data = data["train_data"]
        test_data = data["test_data"]
        val_data = data["val_data"]

        psnr_data = [train_data["psnr"], test_data["psnr"], val_data["psnr"]]
        ssim_data = [train_data["ssim"], test_data["ssim"], val_data["ssim"]]

        DNeRFPlotManager.violin_plot(psnr_data, "PSNR [dB]", "psnr")
        DNeRFPlotManager.violin_plot(ssim_data, "SSIM", "ssim")
