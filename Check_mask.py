import torch
import pandas as pd

# 加载之前保存的文件
m_real = torch.load('/home/yuhengfeng/Experiments/GTCRN/mask_log/m_real.pt')
m_imag = torch.load('/home/yuhengfeng/Experiments/GTCRN/mask_log/m_imag.pt')

# 去掉前面的1, 1两个维度
m_real_squeezed = m_real.squeeze(dim=0).squeeze(dim=0)
m_imag_squeezed = m_imag.squeeze(dim=0).squeeze(dim=0)

# 转换为DataFrame以便查看
df_real = pd.DataFrame(m_real_squeezed.numpy())
df_imag = pd.DataFrame(m_imag_squeezed.numpy())

# 将 DataFrame 保存为 CSV 文件
df_real.to_csv('/home/yuhengfeng/Experiments/GTCRN/mask_log/mask_real_part.csv', index=False)
df_imag.to_csv('/home/yuhengfeng/Experiments/GTCRN/mask_log/mask_imaginary_part.csv', index=False)

print("Real part of mask:")
print(df_real)

print("\nImaginary part of mask:")
print(df_imag)
