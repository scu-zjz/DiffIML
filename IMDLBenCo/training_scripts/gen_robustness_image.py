import matplotlib.pyplot as plt
import json


# 定义一个函数来绘制图表
def plot_robustness(data, models, save_path, y_label):
    tests = ['Jpeg_Compression', 'Gaussian_Blur', 'Gaussian_Noise']
    markers = ['o', '^', 'v', '<', '>', '*', 'D', 'p']
    markers_idx = 0
    # 创建一个1行3列的子图布局
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    lines = []
    labels = []
    for i, test in enumerate(tests):
        ax = axs[i]
        markers_idx = 0
        for j, model_data in enumerate(data[test]):
            model_name = model_data['model_name']
            if model_name in models:
                x = list(model_data['score'].keys())
                y = list(model_data['score'].values())
                marker = markers[markers_idx]  # 使用预定义的标记形状
                markers_idx += 1
                line, = ax.plot(x, y, marker=marker, markersize=9)
                if i == 0:
                    lines.append(line)
                    labels.append(model_name)

        ax.set_xlabel(test.replace('_', ' '))
        ax.set_ylabel(y_label)

        ax.grid(True)

    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.985), ncol=len(models), framealpha=0.5,
               fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(save_path, dpi=350)


if __name__ == '__main__':
    # 用户自定义参数
    models_to_plot = ["Mantra-Net", "MVSS-Net", "CAT-Net", "NCL-IML", "PSCC-Net", "DiffIML"]  # 需要绘制的模型名称
    save_path = "./data/robustness_tests.png"  # 保存路径
    json_path = "./data/robustness.json"  # 保存数据的json路径
    y_label = "F1"  # 纵轴名称

    # 读取JSON数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 调用函数绘制图表并保存
    plot_robustness(data, models_to_plot, save_path, y_label)
