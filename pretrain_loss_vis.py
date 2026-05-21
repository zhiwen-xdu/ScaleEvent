import numpy as np
import matplotlib.pyplot as plt



loss_log = "/home/zhiyu/projects/DINOv3/scale_event/output/DINO21/train.log"

with open(loss_log) as f:
    loss_lines = [line.strip() for line in f.readlines()]

l1_loss = []
intra_structure_loss = []
cross_structure_loss = []

line_num = len(loss_lines)
for i in range(len(loss_lines)):
    if i % 1==0:
        line = loss_lines[i]
        l1_str = float(line.split(",")[3].split(":")[-1])
        intra_str = float(line.split(",")[4].split(":")[-1])
        cross_str = float(line.split(",")[5].split(":")[-1])

        l1_loss.append(l1_str)
        intra_structure_loss.append(intra_str)
        cross_structure_loss.append(cross_str)



window_size = 5
window = np.ones(window_size) / window_size
l1_loss_w = np.convolve(l1_loss, window, 'same')
intra_structure_loss_w = np.convolve(intra_structure_loss, window, 'same')
cross_structure_loss_w = np.convolve(cross_structure_loss, window, 'same')
# l1_loss_w[0],l1_loss_w[-1] = l1_loss[0],l1_loss[-1]
# intra_structure_loss_w[0],intra_structure_loss_w[-1] = intra_structure_loss[0],intra_structure_loss[-1]
# cross_structure_loss_w[0],cross_structure_loss_w[-1] = cross_structure_loss[0],cross_structure_loss[-1]


# 模拟与提供的损失图类似的数据
steps = np.arange(0, len(l1_loss_w[:-50]), 1)


# 绘制图形
plt.figure(figsize=(20, 8))

# 绘制8K和16K迭代次数分别带CFG和不带CFG的曲线
plt.plot(steps, l1_loss_w[:-50], label="L1 Loss", color='m')
plt.plot(steps, intra_structure_loss_w[:-50], label="Intra-modal Structure Loss", color='c')
plt.plot(steps, cross_structure_loss_w[:-50], label="Cross-modal Structure Loss", color='#3c9992')

# plt.plot(steps, l1_loss_w, color='m', linestyle='--')
# plt.plot(steps, intra_structure_loss, color='c', linestyle='--')
# plt.plot(steps, cross_structure_loss, color='#3c9992', linestyle='--')

# # 添加表示8K和16K迭代分界的虚线
# plt.axvline(x=800, color='black', linestyle='--')
# plt.axvline(x=1600, color='black', linestyle='--')
# plt.axvline(x=2400, color='black', linestyle='--')
#
# # 添加最终分数的文本
# plt.text(8500, 0.75, "(8K iterations) VBench Final Score\n78.58 (w/o CFG) / 79.86 (w/ CFG)", color='black')
# plt.text(17000, 0.75, "(16K iterations) VBench Final Score\n80.76 (w/o CFG) / 81.79 (w/ CFG)", color='black')
# plt.text(25500, 0.75, "(24K iterations) VBench Final Score\n80.76 (w/o CFG) / 81.79 (w/ CFG)", color='black')


# 设置标题和标签
plt.title("Distillation Loss - ViTL",fontsize=28)
plt.xlabel("Step",fontsize=24)
plt.ylabel("Loss",fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 显示图例
plt.legend(fontsize=22)

# 保存为高分辨率的PNG图像
plt.tight_layout()
plt.savefig('/home/zhiyu/projects/DINOv3/scale_event/output/vitl_loss_plot.png', dpi=400)

# 显示图形
plt.show()
