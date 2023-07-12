import torch

def lorenz_eq(sigma, beta, rho):
    def lorenz_solver(u_n, dt):
        x , y , z = u_n

        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z

        u_n_plus_1 = u_n + torch.tensor([dx_dt, dy_dt, dz_dt]) * dt

        return u_n_plus_1

    return  lorenz_solver

u=torch.ones(3)
dt=0.1

solver1 = lorenz_eq(10,8/3,28)
u_plus1=u + solver1(u , dt)
print(u_plus1)