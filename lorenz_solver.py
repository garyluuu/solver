import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz_eq(sigma, beta, rho):
    def lorenz_solver(u_n, dt):
        x, y, z = u_n

        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z

        u_n_plus_1 = u_n + torch.tensor([dx_dt, dy_dt, dz_dt]) * dt

        return u_n_plus_1

    return lorenz_solver


# 定义初始条件和时间步长
u_0 = torch.ones(3)
dt = 0.001
T = 50.0

solver1 = lorenz_eq(10, 8/3, 28)

# 计算总步数
num_steps = int(T / dt)

# 初始化结果张量
trajectory = torch.zeros((num_steps + 1, 3))
trajectory[0] = u_0

# 迭代计算每个时间步的状态
for i in range(num_steps):
    u_t = solver1(trajectory[i], dt)
    trajectory[i + 1] = u_t

# 可视化结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(trajectory[:, 0].numpy(), trajectory[:, 1].numpy(), trajectory[:, 2].numpy())
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
