#!/usr/bin/env python

import time

import numpy as np
from numpy import sin, sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.integrate import odeint

def smooth_x(x):
    E = 1.8
    return np.power(np.abs(x),E)*np.sign(x)

def smooth_dx(dx,x):
    x1 = 0.01
    return dx*(-2*(-x/x1)**3 + 3*(-x/x1)**2)

class JointParam(object):
    def __init__(self, J=1.5, K=2000.0):
        self.J = J
        self.K = K

        self.Dl = 10

        self.P = 0.0
        self.D = 0.0

        self.Tjump = 0.3
        self.g = 9.8
        self.l = 0.4/sqrt(2)
        self.m = 70.0*0.5 # for double support phase

        # self.a,self.b = 0.3*1e-4,0.7 # not negative
        # self.a,self.b = 2.3e-4,0.7 # EC-max
        self.a,self.b = 7e-5,0.8 # EC-4pole
        # self.a,self.b = 0.3,2 # not negative
        self.safety_factor = 1.0 # safety factor

class JointImpactAnalyzer(object):
    def __init__(self):
        self.param = JointParam()

    def calc_reducer_torque(self, q, x, dq, dx):
        # fin = K * smooth_x(q-x/l) +self.param.Dl * smooth_dx(dq-dx/l,q-x/l)
        fin_l = self.param.K * (q-x/self.param.l) + self.param.Dl * (dq-dx/self.param.l)
        return fin_l if fin_l > 0 else 0

    def calc_ref_tau(self, t, x, dx, dq):
        # Tland = 0.5
        # Tland = 0.2
        Tland = 0.1
        # Tland = 0.05
        # Tland = 0.02
        # Tland = 0.01

        # alpha = 0.0
        # alpha = 0.3
        # alpha = 0.1
        # alpha = 0.8
        # alpha = 1.0
        # alpha = 1.1
        alpha = 1.15

        # return 0
        # return m*l*g*alpha
        # return min(m*l*g*alpha, m*l*g*alpha*t/Tland)
        # return m*l*(g/Tland * t if dx < 0 else g)

        return - self.param.D*dq # only D control

        # 2-order function x0=0 dx0=-gTjump/2 dx1=0 ddx0=-g ddx1=0
        # a = -3*g/Tland**2 * (1 + Tjump/Tland)
        # return m*l*( (t-Tland)*(a*t+g/Tland) + g if t < Tland else g )

    # def state_equation_func(xvec,t):
    def state_equation_func(self, xvec, t):
        param = self.param

        eta = 1.0

        ret = [0,0,0,0]
        # xvec = [q x dq dx]
        q = xvec[0]
        x = xvec[1]
        dq = xvec[2]
        dx = xvec[3]

        ret[0] = xvec[2]# dq = dq
        ret[1] = xvec[3]# dx = dx

        tau = self.calc_reducer_torque(q,x,dq,dx)
        ref_tau = self.calc_ref_tau(t,x,dx,dq)
        # ret[2] = -param.P/param.J * q -param.D/param.J * dq -    eta/param.J* tau +ref_tau/param.J # ddq
        ret[2] = -param.P/param.J * q                 -    eta/param.J* tau +ref_tau/param.J # ddq including -D*dq into ref_tau
        ret[3] = 1.0/(param.m*param.l)* tau - param.g # ddx

        return ret

    # def calc_ode(self):
    def calc_ode(self):
        duration = 0.5
        # frame_rate = 1000
        frame_rate = 100000
        dt = 1.0/frame_rate
        self.t_vec = np.linspace(0,duration,duration*frame_rate)
        self.x_mat = odeint(self.state_equation_func, [0,0,0,-self.param.g*self.param.Tjump*0.5], self.t_vec)

    def calc_variables(self):
        self.calc_ode()
        self.tau_vec = [self.calc_reducer_torque(xvec[0],xvec[1],xvec[2],xvec[3]) for xvec in self.x_mat]
        self.ref_tau_vec = [self.calc_ref_tau(t,x,dx,dq) for t,(x,dq,dx) in zip(self.t_vec,self.x_mat[:,1:4])]

    def plot_answer(self, title='Joint Impact'):
        deg_scale = 1
        tau_scale = 0.1
        plt.figure(0, figsize=(18,15))
        plt.cla()
        plt.title(title)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.ylim(-100,200)
        # plt.axes(axisbg='white')
        plt.grid(True, color='gray', linestyle='dashed')
        plt.plot(self.t_vec, np.rad2deg(self.x_mat[:,0])*deg_scale,    label='q *'+str(deg_scale)+'[deg]')
        plt.plot(self.t_vec, np.rad2deg(self.x_mat[:,1]/self.param.l)*deg_scale,  label='x/l *'+str(deg_scale)+'[deg]')
        # plt.plot(t, K*(x[:,0] - x[:,1]/l)*tau_scale, label='tau*'+str(tau_scale)+'[Nm]')
        # plt.plot(t, [K*smooth_x(val)*tau_scale for val in x[:,0] - x[:,1]/l], label='tau*'+str(tau_scale)+'[Nm]')
        # plt.plot(t, [(K*smooth_x(delq)+Dl*smooth_dx(deldq,delq))*tau_scale for delq,deldq in zip(x[:,0]-x[:,1]/l, x[:,2]-x[:,3]/l)], label='tau*'+str(tau_scale)+'[Nm]')
        plt.plot(self.t_vec, [val*tau_scale for val in self.tau_vec],     label=    'tau*'+str(tau_scale)+'[Nm]')
        plt.plot(self.t_vec, [val*tau_scale for val in self.ref_tau_vec], label='ref_tau*'+str(tau_scale)+'[Nm]')
        plt.legend(loc='upper right', frameon=True)
        plt.pause(0.5)
        # plt.savefig('x-t.png', dpi=300, facecolor='white', transparent=False, format="png")

class ExhaustiveSearchInterface(object):
    def __init__(self):
        self.jia = JointImpactAnalyzer()
        self.initialize_plot()

    def initialize_plot(self):
        # 3D
        self.cmap = 'hsv'
        self.fig = plt.figure()
        # self.ax = Axes3D(self.fig)
        self.Lax = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.Rax = self.fig.add_subplot(1, 2, 2, projection='3d')
        self.axes = [self.Lax, self.Rax]

        self.fig.set_size_inches((16.0,8.0))
        self.fig.subplots_adjust(left=-0.05,right=0.95, bottom=0.02,top=1, wspace=0.1, hspace=1)
        self.fontsize = 35

        self.lx_max = 1.0
        self.rx_max = 500
        self.rz_max = 1000

        for ax in self.axes:
            # ticks
            tics_fontsize_rate = 0.8
            # ax.axes.tick_params(labelsize=self.fontsize*tics_fontsize_rate)
            ax.tick_params(labelsize=self.fontsize*tics_fontsize_rate)

            # margin between tics and axis label
            labelpad_rate = 0.6
            ax.axes.xaxis.labelpad=self.fontsize*labelpad_rate
            ax.axes.yaxis.labelpad=self.fontsize*labelpad_rate
            ax.axes.zaxis.labelpad=self.fontsize*labelpad_rate

            # select tics position
            ax.axes.xaxis.tick_top()
            ax.axes.yaxis.tick_bottom()
            ax.axes.zaxis.tick_top()

    def update_plot_conf(self):
        # label
        label_fontsize_rate = 1.1
        self.Lax.set_xlabel(self.value_list[0][0],fontsize=self.fontsize*label_fontsize_rate)
        self.Lax.set_ylabel(self.value_list[1][0],fontsize=self.fontsize*label_fontsize_rate)
        self.Lax.set_zlabel(r'$\tau_{\mathrm{max}}$: [Nm]',fontsize=self.fontsize*label_fontsize_rate)

        self.Rax.set_xlabel('m [kg]',fontsize=self.fontsize*label_fontsize_rate)
        self.Rax.set_ylabel(self.value_list[1][0],fontsize=self.fontsize*label_fontsize_rate)
        self.Rax.set_zlabel(r'$\tau_{\mathrm{design}}$: [Nm]',fontsize=self.fontsize*label_fontsize_rate)

    def plot_2d_map(self, value_list):
        self.sweep_variables(value_list=value_list, sleep_time=0, plot_2d=False)

        x_grid,y_grid,z_grid = self.x_grid, self.y_grid, self.z_grid
        x,y,z = x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
        # x,y,z = self.plot_data[value_list[0][0]], self.plot_data[value_list[1][0]], self.plot_data['max_tau']

        # # 2D
        # plt.scatter(self.plot_data[value_list[0][0]], self.plot_data[value_list[1][0]], c=self.plot_data['max_tau'])
        # plt.colorbar()

        lx_max = self.lx_max
        self.Lax.set_xlim3d(0,min(lx_max,np.max(x_grid)))
        z_limited = np.where(x>lx_max, 0,z)
        p = self.Lax.scatter(np.clip(x, 0,lx_max),y,z_limited, c=z_limited, cmap=self.cmap, alpha=0.7)
        self.Lax.plot_surface(np.clip(x_grid, 0,lx_max),
                              y_grid,
                              np.where(x_grid>lx_max, 0,z_grid),
                              cmap=self.cmap, linewidth=0.3, alpha=0.3, edgecolors='black')
        # self.Lax.contourf(x_grid, y_grid, z_grid, cmap=self.cmap, offset=-10.0)
        # self.fig.colorbar(p, shrink=0.6, aspect=10, orientation='horizontal')
        rx_max,rz_max = self.rx_max,self.rz_max
        # self.Rax.set_xlim3d(0,rx_max)
        m_grid,design_tau_grid = np.clip(self.m_grid, 0,rx_max), np.clip(self.design_tau_grid, 0,rz_max)
        # m_grid = np.log(m_grid)
        m,design_tau = m_grid.flatten(), design_tau_grid.flatten()
        # surface
        p = self.Rax.scatter(m, y, design_tau, c=design_tau, cmap=self.cmap, alpha=0.7)
        self.Rax.plot_surface(m_grid, y_grid, design_tau_grid, cmap=self.cmap, linewidth=0.3, alpha=0.3, edgecolors='black')

        plt.pause(0.5)

    def sweep_variables(self, value_list=None, sleep_time=0.2, plot_2d=False):
        self.value_list = (('K', [1000,2000,3000,6000,25000,47000,110000]), ('Dl', np.linspace(0,30, 10, dtype=int))) if value_list is None else value_list

        self.update_plot_conf()

        # self.plot_data = {}
        # for key_str,val_list in value_list:
        #     # setattr(self, key_str+'_list', [])
        #     self.plot_data[key_str] = []
        # self.plot_data['max_tau'] = []

        # self.sweep_variables_impl(value_list, sleep_time)

        x_key,x_values = self.value_list[0]
        y_key,y_values = self.value_list[1]
        self.x_grid,self.y_grid = np.meshgrid(x_values,y_values)
        self.z_grid = np.zeros(self.x_grid.shape)
        self.m_grid = np.zeros(self.x_grid.shape)
        self.design_tau_grid = np.zeros(self.x_grid.shape)
        jia = self.jia
        for i,x_value in enumerate(x_values):
            setattr(jia.param, x_key, x_value)
            for j,y_value in enumerate(y_values):
                setattr(jia.param, y_key, y_value)
                jia.calc_variables()
                impact_tau = np.max(jia.tau_vec)

                self.z_grid[j][i] = impact_tau

                design_tau = impact_tau*jia.param.safety_factor
                self.m_grid[j][i] = ( jia.param.J/(jia.param.a*design_tau**2) )**(-1.0/jia.param.b)
                self.design_tau_grid[j][i] = design_tau

                J,K,Dl = jia.param.J,jia.param.K,jia.param.Dl
                print (J,K,Dl), ' m:', self.m_grid[j][i], ' tau:', impact_tau
                if plot_2d: self.jia.plot_answer(title='J:'+str(J)+' K:'+str(K)+' Dl:'+str(Dl))

                time.sleep(sleep_time)
            print ''
        print ''

    # def sweep_variables_impl(self, value_list, sleep_time):
    #     if len(value_list) < 1:
    #         self.jia.calc_variables()
    #         impact_tau = np.max(self.jia.tau_vec)
    #         print (J,K,Dl), impact_tau
    #         # self.jia.plot_answer(title='J:'+str(J)+' K:'+str(K)+' Dl:'+str(Dl))

    #         time.sleep(sleep_time)
    #         return impact_tau
    #     else:
    #         key_str,values = value_list[0]
    #         for value in values:
    #             getattr(self, key_str+'_list').append(value)
    #             exec(key_str + '=' + str(value), globals())
    #             self.sweep_variables_impl(value_list[1:], sleep_time)

    #         print ''

if __name__ == '__main__':
    # jia = JointImpactAnalyzer()
    # jia.calc_variables()
    # jia.plot_answer()

    esi = ExhaustiveSearchInterface()
    # esi.sweep_variables()

    # esi.plot_2d_map( (('K', [1000,2000,3000,6000,25000,47000,110000]), ('Dl', np.linspace(0,30, 10, dtype=int))) )

    esi.plot_2d_map( (('J', np.round(np.linspace(0.1,3, 10),1)), ('K', [1000,2000,3000,6000,25000,47000,110000])) )
