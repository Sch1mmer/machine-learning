import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']



# 数据
epochs = range(20)
data1 = [
    1.0773049205731435, 1.0495495052094672, 1.0422959084723407, 1.0113456852876457, 0.9991762835508698,
    0.9928255992330564, 0.9876350601007984, 0.989633291010644, 0.9894194713064061, 1.0037940315380218,
    0.9863616615344005, 0.9913203139214, 0.9879494685276299, 0.9943465719557112, 0.9832444282094385,
    0.9822459163939118, 0.9832496119152968, 0.9840646765793964, 0.9910085015236192, 0.9827241836839421
]
data2 = [
    1.0938449583876246, 1.0304936530491033, 1.059322571030821, 1.0594177611719686, 0.9923921998697348,
    0.9840596428694436, 0.9863304608165265, 0.9849528845506735, 0.9990257990246002, 0.9885521302588831,
    0.9879969234664601, 0.9852717450251595, 0.9802308383460243, 0.9885827240090781, 0.9814880505537453,
    0.9851816376558127, 0.9840090505231303, 0.9798121518982105, 0.9774572860699492, 0.9815872757198711
]

# 创建图形并确保没有错误
try:
    fig, ax = plt.subplots()
except Exception as e:
    print(f"Error creating subplots: {e}")
    raise

# 绘制准确度折线图
color1 = 'tab:red'
ax.set_xlabel('周期')
ax.set_ylabel('误差')
line1, = ax.plot(epochs, data1, color=color1, label='ResNet_4误差')
ax.tick_params(axis='y')

# Plot the second line (ResNet准确度) on the same y-axis
color2 = 'tab:blue'
line2, = ax.plot(epochs, data2, color=color2, label='ResNet误差')

# 添加图例
lines = [line1, line2]
ax.legend(lines, [line.get_label() for line in lines])

# 调整布局, 捕获可能的错误
try:
    fig.tight_layout()
except Exception as e:
    print(f"Error during tight_layout: {e}")
    raise

plt.title('ResNet与ResNet_4误差对比')
plt.show()