#!/usr/bin/env python

import time

import numpy as np
from numpy import sin, sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.integrate import odeint

from logger import *

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
        self.F = 0.0

        self.P = 0.0
        self.D = 0.0

        # self.Tjump = 0.3
        # self.Tjump = 0.4
        self.Tjump = 0.5
        self.g = 9.8
        self.l = 0.4/sqrt(2)
        self.m = 70.0*0.5 # for double support phase

        # self.a,self.b = 0.3*1e-4,0.7 # not negative
        # self.a,self.b = 2.3e-4,0.7 # EC-max
        self.a,self.b = 7e-5,0.8 # EC-4pole
        # self.a,self.b = 0.3,2 # not negative
        # self.design_torque_factor = 1.35 # max design torque / continuous joint torque
        self.design_torque_factor = 1.5 # max design torque / continuous joint torque (updated knee)

    def param_dict(self):
        return {
            'm':self.m,
            'leg':self.l,
            'Tjump':self.Tjump,
            'Dj':self.Dl,
            'Fj':self.F,
            'alpha':self.design_torque_factor,
        }

class JointSample(object):
    def __init__(self, motor_name='EC-4pole 30 200W 36V 300g', Jm=33.3*1e-7, motor_max_tq=0.104, gear_ratio=229):
        self.motor_name = motor_name
        self.Jm = Jm
        self.motor_max_tq = motor_max_tq
        self.gear_ratio = gear_ratio
        self.data = None

    def Jtau_coeff(self):
        return self.Jm/self.motor_max_tq**2

    def Jjoint(self):
        return self.Jm*(self.gear_ratio**2)

class JointImpactAnalyzer(object):
    def __init__(self):
        self.param = JointParam()

    def calc_reducer_torque(self, q, x, dq, dx):
        # fin = K * smooth_x(q-x/l) +self.param.Dl * smooth_dx(dq-dx/l,q-x/l)
        # fin_l = self.param.K * (q-x/self.param.l) + self.param.Dl * (dq-dx/self.param.l)
        fin_l = self.param.K * (q-x/self.param.l) + self.param.Dl * dq + self.param.F
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
        # duration = 0.5
        # frame_rate = 1000
        duration = 2.0
        frame_rate = 5000
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

        self.joint_samples = {}

    def initialize_plot(self):
        # 3D
        self.cmap = 'hsv'
        self.color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        mpl.rc('text', usetex=True)

        self.fontsize = 35

        self.lx_max = 0.5
        self.rx_max = 500
        self.rz_max = 800

    def update_plot_conf(self, clear=True):
        # clear list
        if type(clear) == list:
            self.clear_list = clear if len(clear) == len(self.axes) else [clear[0] for ax in self.axes]
        else:
            self.clear_list = [clear for ax in self.axes]
        # color index list
        if (not hasattr(self,'color_index_list')) or len(self.color_index_list) != len(self.axes): self.color_index_list = [-1 for ax in self.axes]
        # 3D
        for idx,(ax,clear) in enumerate(zip(self.axes, self.clear_list)):
            # clear
            if clear:
                ax.cla()
                self.color_index_list[idx] = 0
            else:
                self.color_index_list[idx] = (self.color_index_list[idx] + 1) % len(self.color_list)
            # ticks
            tics_fontsize_rate = 0.8
            # ax.axes.tick_params(labelsize=self.fontsize*tics_fontsize_rate)
            ax.tick_params(labelsize=self.fontsize*tics_fontsize_rate)

            # margin between tics and axis label
            labelpad_rate = 0.6
            ax.axes.xaxis.labelpad=self.fontsize*labelpad_rate
            ax.axes.yaxis.labelpad=self.fontsize*labelpad_rate
            if hasattr(ax.axes,'zaxis'): ax.axes.zaxis.labelpad=self.fontsize*labelpad_rate

            # select tics position
            if hasattr(ax.axes,'zaxis'):
                ax.axes.xaxis.tick_bottom()
                ax.axes.yaxis.tick_bottom()
                ax.axes.zaxis.tick_bottom()
                ax.view_init(30,-55) # rotate view

    def plot_3d_map(self, value_list, num_tau=1, update=True, clear=True):
        if update: self.sweep_variables(value_list=value_list, sleep_time=0, plot_2d=False, num_tau=num_tau)

        # axis
        if not hasattr(self,'Lax'): self.Lax = plt.figure(figsize=(8,8)).gca(projection='3d')
        if not hasattr(self,'Rax'): self.Rax = plt.figure(figsize=(8,8)).gca(projection='3d')
        self.axes = [self.Lax, self.Rax]
        self.update_plot_conf(clear=clear)
        # figure
        map((lambda ax: ax.figure.subplots_adjust(left=-0.1,right=0.94, bottom=0.05,top=1, wspace=0.1, hspace=1)), self.axes)
        # label
        label_fontsize_rate = 0.9
        # self.Lax.set_xlabel(self.value_list[0][0],fontsize=self.fontsize*label_fontsize_rate)
        # self.Lax.set_ylabel(self.value_list[1][0],fontsize=self.fontsize*label_fontsize_rate)
        self.Lax.set_xlabel('$'+self.value_list[0][0]+'_\mathrm{jnt}$ [kgm$^2$]',fontsize=self.fontsize*label_fontsize_rate)
        self.Lax.set_ylabel('$'+self.value_list[1][0]+'_\mathrm{jnt}$ [Nm/rad]',fontsize=self.fontsize*label_fontsize_rate)
        self.Lax.set_zlabel(r'$_{\mathrm{max}}\tau_{\mathrm{jnt}}$ [Nm]',fontsize=self.fontsize*label_fontsize_rate)

        # self.Rax.set_xlabel('m [kg]',fontsize=self.fontsize*label_fontsize_rate)
        # self.Rax.set_ylabel(self.value_list[1][0],fontsize=self.fontsize*label_fontsize_rate)
        self.Rax.set_xlabel('$M_\mathrm{mot}$ [kg]',fontsize=self.fontsize*label_fontsize_rate)
        self.Rax.set_ylabel('$'+self.value_list[1][0]+'_\mathrm{jnt}$ [Nm/rad]',fontsize=self.fontsize*label_fontsize_rate)
        self.Rax.set_zlabel(r'$_{\mathrm{dsn}}\tau_{\mathrm{jnt}}$ [Nm]',fontsize=self.fontsize*label_fontsize_rate)

        x_grid,y_grid,z_grid = self.x_grid, self.y_grid, self.impact_tau_grid
        # rz_grid = self.cont_tau_grid
        rz_grid = self.impact_tau_grid

        # log settings
        y_grid = np.log10(y_grid)
        max_exp = np.ceil(np.log10(np.max(self.y_grid[:,0])))
        min_exp = np.floor(np.log10(max(1,np.min(self.y_grid[:,0]))))
        self.Lax.set_yticklabels(['    $10^{'+'{0:.1f}'.format(val)+'}$' for val in np.linspace(min_exp,max_exp, max_exp-min_exp+1)])
        self.Rax.set_yticklabels(['    $10^{'+'{0:.1f}'.format(val)+'}$' for val in np.linspace(min_exp,max_exp, max_exp-min_exp+1)])

        # use log scale in J map
        z_grid = np.log10(z_grid)
        max_exp = np.ceil(np.log10(np.max(self.impact_tau_grid)))
        min_exp = -1.0
        self.Lax.set_zticklabels(['    $10^{'+'{0:.1f}'.format(val)+'}$' for val in np.linspace(min_exp,max_exp, max_exp-min_exp+1)])

        x,y,z = x_grid.flatten(), y_grid.flatten(), z_grid.flatten()

        # J map
        lx_max = self.lx_max
        self.Lax.set_xlim3d(0,min(lx_max,np.max(x_grid)))
        # z_limited = np.where(x>lx_max, 0,z)
        # p = self.Lax.scatter(np.clip(x, 0,lx_max),y,z_limited, c=z_limited, cmap=self.cmap, alpha=0.7)
        # self.Lax.plot_surface(np.clip(x_grid, 0,lx_max),
        #                       y_grid,
        #                       np.where(x_grid>lx_max, 0,z_grid),
                              # cmap=self.cmap, linewidth=0.3, alpha=0.3, edgecolors='gray')
        x_max_idx = np.max(np.where(self.x_grid[0] <= lx_max))+1
        self.Lax.plot_surface(x_grid[:,0:x_max_idx],
                              y_grid[:,0:x_max_idx],
                              z_grid[:,0:x_max_idx],
                              # color=self.color_list[self.color_index_list[0]], linewidth=0.3, alpha=0.3, edgecolors='gray', shade=True)
                              color=self.color_list[self.color_index_list[0]], linewidth=1, alpha=0, edgecolors=self.color_list[self.color_index_list[0]])

        rx_max,rz_max = self.rx_max,self.rz_max
        m_grid,rz_grid = np.clip(self.m_grid, 0,rx_max), np.clip(rz_grid, 0,rz_max)
        # m_grid = np.log10(m_grid)
        m,rz = m_grid.flatten(), rz_grid.flatten()

        # legend by dummy plot
        linestyle='-'; linewidth=2
        if self.Lax.legend_ is None:
            tmp_fake_plot_list = []
            tmp_text_list = []
        else:
            tmp_fake_plot_list = [mpl.lines.Line2D([0],[0], linestyle=linestyle, linewidth=linewidth, c=handle.get_color()) for handle in self.Lax.legend_.legendHandles]
            tmp_text_list = [text.get_text() for text in self.Lax.legend_.texts]
        tmp_fake_plot_list.append( mpl.lines.Line2D([0],[0], linestyle=linestyle, linewidth=linewidth, c=self.color_list[self.color_index_list[0]]) )
        tmp_text_list.append( '{0}={1} Nm/(rad/s), {2}={3} Nm'.format(r'$D_\mathrm{jnt}$',self.jia.param.Dl, '$F_\mathrm{jnt}$',self.jia.param.F ) )
        self.Lax.legend(tmp_fake_plot_list, tmp_text_list, numpoints=1, fontsize=self.fontsize*0.6)

        # M map
        # p = self.Rax.scatter(m, y, rz, c=rz, cmap=self.cmap, alpha=0.7)
        self.Rax.plot_surface(m_grid, y_grid, rz_grid, cmap=self.cmap, linewidth=0.3, alpha=0.3, edgecolors='gray')

        plt.pause(0.5)

    def plot_motor_map(self, value_list, update=True, fname=None):
        if update: self.sweep_variables(value_list=value_list, sleep_time=0, plot_2d=False)

        # axis
        if not hasattr(self,'motor_ax'): self.motor_ax = plt.figure(figsize=(18,6)).gca()
        self.axes = [self.motor_ax]
        self.update_plot_conf()
        # figure
        self.motor_ax.figure.subplots_adjust(left=0.1,right=0.98, bottom=0.13,top=0.95, wspace=0.1, hspace=1)
        # label
        label_fontsize_rate = 0.9
        self.motor_ax.set_xlabel('$'+self.value_list[1][0]+'_\mathrm{jnt}$ [Nm/rad]',fontsize=self.fontsize*label_fontsize_rate)
        self.motor_ax.set_ylabel(r'$_{\mathrm{max}}\tau_{\mathrm{jnt}}$ [Nm]',fontsize=self.fontsize*label_fontsize_rate)

        # margin
        self.motor_ax.xaxis.labelpad=-5
        self.motor_ax.yaxis.labelpad=0
        # grid
        self.motor_ax.grid()
        # limit
        # self.motor_ax.set_xlim(10,2000a0)
        # self.motor_ax.set_ylim(10,800)
        # scale
        self.motor_ax.set_xscale('log')

        # plot
        for sample_idx, sample in enumerate(self.joint_samples):
            # self.motor_ax.plot(self.K_values, self.impact_tau_grid[:,sample_idx], '-o', label='{}(J={}) {}:1'.format(sample.motor_name, sample.Jm, sample.gear_ratio))
            self.motor_ax.plot(self.K_values, self.impact_tau_grid[:,sample_idx], '-', label='{}(J={}) {}:1'.format(sample.motor_name, sample.Jm, sample.gear_ratio))

        # legend
        self.motor_ax.legend(fontsize=self.fontsize*0.7, loc='upper left')

        if fname is not None: self.motor_ax.figure.savefig(fname)

        plt.pause(0.5)

    def plot_sample_values(self, value_list, update=True, fname=None):
        if update: self.sweep_variables(value_list=value_list, sleep_time=0, plot_2d=False)

        # axis
        if not hasattr(self, 'sample_fig'): self.sample_ax = plt.figure(figsize=(18,6)).gca()
        self.axes = [self.sample_ax]
        self.update_plot_conf()
        # figure
        self.sample_ax.figure.subplots_adjust(left=0.1,right=0.98, bottom=0.13,top=0.95, wspace=0.1, hspace=1)
        # label
        # self.sample_ax.set_xlabel(self.value_list[1][0],fontsize=self.fontsize)
        self.sample_ax.set_xlabel('$'+self.value_list[1][0]+'_\mathrm{jnt}$ [Nm/rad]',fontsize=self.fontsize)
        self.sample_ax.set_ylabel(r'$_{\mathrm{dsn}}\tau_{\mathrm{jnt}}$ [Nm]',fontsize=self.fontsize)
        # margin
        self.sample_ax.xaxis.labelpad=-5
        self.sample_ax.yaxis.labelpad=0
        # grid
        self.sample_ax.grid()
        # limit
        self.sample_ax.set_xlim(10,20000)
        self.sample_ax.set_ylim(10,800)
        # scale
        self.sample_ax.set_xscale('log')

        # plot
        reference_data_list = [np.sum(sample.data) for sample in self.joint_samples]
        sorted_data = np.sort(reference_data_list)[::-1].tolist()
        for reference_data,joint_sample in zip(reference_data_list,self.joint_samples):
            # self.sample_ax.plot(self.K_values, joint_sample.data, '-', label='{coef_name}={coef:.2e} ({motor})'.format(coef_name=r'$C_{\mathrm{IT}^2}$', coef=joint_sample.Jtau_coeff(), motor=joint_sample.motor_name))
            # self.sample_ax.plot(self.K_values, joint_sample.data, '-o', label='{coef_name}={coef:.2e} ({motor})'.format(coef_name=r'$C_{\mathrm{IT}^2}$', coef=joint_sample.Jtau_coeff(), motor=joint_sample.motor_name))
            bottom_idx = reference_data_list.index(sorted_data[(sorted_data.index(reference_data)+1)%len(reference_data_list)])
            bottom_data = self.joint_samples[bottom_idx].data if reference_data_list[bottom_idx] != sorted_data[0] else np.zeros_like(self.K_values)
            self.sample_ax.fill_between(self.K_values, joint_sample.data, bottom_data, alpha=0.5, label='{coef_name}={coef:.2e} ({motor})'.format(coef_name=r'$C_{\mathrm{IT}^2}$', coef=joint_sample.Jtau_coeff(), motor=joint_sample.motor_name))

        # legend
        self.sample_ax.legend(fontsize=self.fontsize*0.7, loc='upper left')

        plt.pause(0.5)

        # save figure
        if fname is not None: self.sample_ax.figure.savefig(fname)

    def sweep_variables(self, value_list=None, sleep_time=0.2, plot_2d=False, num_tau=1):
        self.value_list = (('K', [1000,2000,3000,6000,25000,47000,110000]), ('Dl', np.linspace(0,30, 10, dtype=int))) if value_list is None else value_list

        for key_str,values in value_list:
            if type(values) == list:
                setattr(self, key_str+'_values', np.array(values))
            else:
                setattr(self, key_str+'_values', values)

        x_key,x_values = self.value_list[0]
        y_key,y_values = self.value_list[1]
        self.x_grid,self.y_grid = np.meshgrid(x_values,y_values)
        self.impact_tau_grid = np.zeros(self.x_grid.shape)
        self.m_grid = np.zeros(self.x_grid.shape)
        self.cont_tau_grid = np.zeros(self.x_grid.shape)

        for joint_sample in self.joint_samples:
            joint_sample.data = np.empty_like(y_values)

        jia = self.jia
        for j,y_value in enumerate(y_values): # y loop
            setattr(jia.param, y_key, y_value)
            for i,x_value in enumerate(x_values): # x loop
                setattr(jia.param, x_key, x_value)
                jia.calc_variables()
                impact_tau = np.max(jia.tau_vec)

                self.impact_tau_grid[j][i] = impact_tau

                cont_tau = impact_tau/jia.param.design_torque_factor
                self.m_grid[j][i] = ( jia.param.J/(jia.param.a*cont_tau**2) )**(-1.0/jia.param.b)
                self.cont_tau_grid[j][i] = cont_tau

                J,K,Dl = jia.param.J,jia.param.K,jia.param.Dl
                # logger.info('({},{},{}) m: tau:'.format(J, K, Dl, self.m_grid[j][i], impact_tau))
                if plot_2d: self.jia.plot_answer(title='J:'+str(J)+' K:'+str(K)+' Dl:'+str(Dl)+'   max tau:'+str(np.round(impact_tau,1)))

                time.sleep(sleep_time)

            logger.info('y: {}'.format(y_value))
            for joint_sample in self.joint_samples:
                impact_tau_vec = self.impact_tau_grid[j]
                estimated_tau_vec = jia.param.design_torque_factor * ( self.J_values/joint_sample.Jtau_coeff() )**0.5 # = alpha * ( Jk*(max_tq^2/Jm) )^0.5
                delta_vec = np.clip(abs(impact_tau_vec)*0.05, 6,np.inf)
                idx = np.append(np.where( abs(impact_tau_vec - estimated_tau_vec) < delta_vec )[0],0).max()
                logger.info('{}: {}'' J: {} tq: {}'.format(joint_sample.motor_name, idx, self.J_values[idx], np.round(impact_tau_vec[:idx+1],1)))
                joint_sample.data[j] = impact_tau_vec[idx]

            logger.info('')
        logger.info('')

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

def export_JKM_map(ext='.pdf'):
    format_str = '{motor_name}_m{m:.0f}_l{leg:.2g}_Dj{Dj:g}_Fj{Fj:g}_Tjump{Tjump}_alpha{alpha}'
    # value_range = (('J', np.round(np.linspace(0.1,1, 20),3)), ('K', [1000,2000,3000,6000,15000,25000,35000,47000]))
    # value_range = (('J', np.round(np.hstack([np.linspace(0.05**0.5,1**0.5, 10)**2, np.linspace(2**0.5,1000**0.5, 10)**2]),2)),
    value_range = (('J', np.round(np.hstack([np.linspace(0.005**0.5,0.5**0.5, 10)**2, [0.7,0.85], np.linspace(1**0.5,50**0.5, 10)**2, np.linspace(60**0.5,1000**0.5, 5)**2]),3)),
    # value_range = (('J', np.round(np.hstack([np.linspace(0.05**0.5,1**0.5, 5)**2, np.linspace(2**0.5,50**0.5, 10)**2, np.linspace(60**0.5,1000**0.5, 5)**2]),2)),
    # value_range = (('J', np.round(np.hstack([np.linspace(0.05**0.5,1**0.5, 10)**2, np.linspace(1.1**0.5,2**0.5, 5)**2, np.linspace(2.1**0.5,1000**0.5, 5)**2]),2)),
                   # ('K', [1000,2000,3000,6000,15000,25000,35000,47000]))
                   ('K', [1,10,100,500,1000,2000,3000,6000,15000,25000,35000,47000]))
    rx_max = 0.6
    Dl = 0.07 # 292*240(gear ratio)*2e-20 from JAXON's fsharp
    F = 12.0 # 820*240(gear ratio)*2e-14 from JAXON's fsharp

    # EC
    esi0 = ExhaustiveSearchInterface()
    motor_name = 'EC'
    param = esi0.jia.param
    param.a, param.b = 2.2e-4, 0.91
    esi0.rx_max = rx_max

    param.Dl = 0.0
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi0.plot_3d_map( value_range )
    esi0.Rax.figure.savefig('M-K-map_'+param_str+ext)

    param.Dl = Dl; param.F = F
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi0.plot_3d_map( value_range, clear=[False,True] )
    esi0.Rax.figure.savefig('M-K-map_'+param_str+ext)

    # EC-max
    esi1 = ExhaustiveSearchInterface()
    motor_name = 'EC-max'
    param = esi1.jia.param
    param.a, param.b = 2.3e-4, 0.69
    esi1.rx_max = rx_max

    param.Dl = 0.0
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi1.plot_3d_map( value_range )
    esi1.Rax.figure.savefig('M-K-map_'+param_str+ext)

    param.Dl = Dl; param.F = F
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi1.plot_3d_map( value_range, clear=[False,True] )
    esi1.Rax.figure.savefig('M-K-map_'+param_str+ext)

    # EC-4pole
    esi2 = ExhaustiveSearchInterface()
    motor_name = 'EC-4pole'
    param = esi2.jia.param
    param.a, param.b = 7.1e-5, 0.80
    esi2.rx_max = rx_max
    param.Dl = 0.0
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi2.plot_3d_map( value_range )
    esi2.Rax.figure.savefig('M-K-map_'+param_str+ext)

    param.Dl = Dl; param.F = F
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi2.plot_3d_map( value_range, clear=[False,True] )
    esi2.Rax.figure.savefig('M-K-map_'+param_str+ext)
    param_str = str.replace('{motor_name}_m{m:.0f}_Tjump{Tjump}'.format(motor_name=motor_name, **param.param_dict()), '.','-')
    esi2.Lax.figure.savefig('J-K-map_'+param_str+ext)

    # EC-i
    esi3 = ExhaustiveSearchInterface()
    motor_name = 'EC-i'
    param = esi3.jia.param
    param.a, param.b = 7.6e-5, 0.56
    esi3.rx_max = rx_max
    param.Dl = 0.0
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi3.plot_3d_map( value_range )
    esi3.Rax.figure.savefig('M-K-map_'+param_str+ext)

    param.Dl = Dl; param.F = F
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi3.plot_3d_map( value_range, clear=[False,True] )
    esi3.Rax.figure.savefig('M-K-map_'+param_str+ext)

    # ILM
    esi4 = ExhaustiveSearchInterface()
    motor_name = 'ILM'
    param = esi4.jia.param
    param.a, param.b = 1.3e-5, 0.71
    esi4.rx_max = rx_max
    param.Dl = 0.0
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi4.plot_3d_map( value_range )
    esi4.Rax.figure.savefig('M-K-map_'+param_str+ext)

    param.Dl = Dl; param.F = F
    param_str = str.replace(format_str.format(motor_name=motor_name, **param.param_dict()), '.','-')
    logger.critical(param_str)
    esi4.plot_3d_map( value_range, clear=[False,True] )
    esi4.Rax.figure.savefig('M-K-map_'+param_str+ext)

def export_joint_sample_map(ext='.pdf'):
    # sample joint
    air_cooling_current = 10.0
    # air_cooling_current = 30.0
    water_cooling_current = 50.0

    head_fname = 'spesific-motor-joint-impact-map'
    param_format = '_m{m:.0f}_l{leg:.2g}_Tjump{Tjump}_alpha{alpha}_Dj{Dj:.0f}_Fj{Fj:.0f}'

    # value_range = (('J', np.round(np.hstack([np.linspace(0.01**0.5,5**0.5, 10)**2, np.linspace(6**0.5,10**0.5, 10)**2, np.linspace(10**0.5,1000**0.5, 5)**2]),2)),
    # value_range = (('J', np.round(np.hstack([np.linspace(0.05,3, 10), np.linspace(3.5**0.5,30**0.5, 10)**2, np.linspace(35**0.5,100**0.5, 10)**2, np.linspace(110**0.5,1000**0.5, 20)**2]),3)),
    value_range = (('J', np.round(np.hstack([np.linspace(0.2,3, 10), np.linspace(3.5**0.5,30**0.5, 10)**2, np.linspace(35**0.5,100**0.5, 10)**2, np.linspace(110**0.5,1000**0.5, 20)**2]),3)),
    # value_range = (('J', np.round(np.hstack([np.linspace(0.001,3, 20), np.linspace(3.5**0.5,30**0.5, 10)**2, np.linspace(35**0.5,100**0.5, 10)**2, np.linspace(110**0.5,1000**0.5, 20)**2]),3)),
                   # ('K', [1,10,100,500,1000,2000,3000,6000,15000,25000,35000,47000]))
                   # ('K', [7,8,10,15,30,60,100,200,300,1000,2000,3000,6000,15000,25000]))
                   # ('K', [7,8,10,15,30,60,100,120,150,200,300,1000,2000,3000,6000,15000,25000]))
                   ('K', [10,20,50,100,200,300,500,700,1000,2000,11000,25000,30000]))

    esi5 = ExhaustiveSearchInterface()
    param = esi5.jia.param
    param.Dl = 0
    param_str = str.replace(param_format.format(**param.param_dict()), '.','-')

    # EC-4pole
    esi5.joint_samples = (
        JointSample(motor_name='EC-4pole 30 100W 36V 175g', Jm=8.91*1e-7, motor_max_tq=0.0564), # coef=0.00280
        JointSample(motor_name='EC-4pole 30 100W 36V 210g', Jm=18.3*1e-7, motor_max_tq=0.0780), # coef=0.00301
        JointSample(motor_name='EC-4pole 30 200W 36V 300g', Jm=33.3*1e-7, motor_max_tq=0.104), # coef=0.00306
        # JointSample(motor_name='EC-4pole 30 200W 36V double motor 600 g', Jm=2*33.3*1e-7, motor_max_tq=2*0.104),
        # JointSample(motor_name='EC-4pole 30 200W 36V ({0:.0f}A) 300g'.format(air_cooling_current), Jm=33.3*1e-7, motor_max_tq=0.0205*air_cooling_current),
        # JointSample(motor_name='EC-4pole 30 200W 36V ({0:.0f}A) 300g x 2'.format(air_cooling_current), Jm=2*33.3*1e-7, motor_max_tq=2*0.0205*air_cooling_current),
        JointSample(motor_name='EC-4pole 30 200W 36V ({0:.0f}A) 300g'.format(water_cooling_current), Jm=33.3*1e-7, motor_max_tq=0.0205*water_cooling_current),
        JointSample(motor_name='EC-4pole 30 200W 36V ({0:.0f}A) 300g x 2'.format(water_cooling_current), Jm=2*33.3*1e-7, motor_max_tq=2*0.0205*water_cooling_current),
    )
    esi5.plot_sample_values(value_range, fname=head_fname+'_EC-4pole'+param_str+ext)

    # EC-i
    esi5.joint_samples = (
        # JointSample(motor_name='EC-i 40  50W 36V 150g', Jm= 8.1*1e-7, motor_max_tq=0.0742), # coef=0.020
        JointSample(motor_name='EC-i 40  50W 36V 180g', Jm=12.8*1e-7, motor_max_tq=0.0742), # coef=0.023
        JointSample(motor_name='EC-i 40  70W 36V 242g', Jm=15.5*1e-7, motor_max_tq=0.106), # coef=0.0014
        JointSample(motor_name='EC-i 40  70W 36V 250g', Jm=23.0*1e-7, motor_max_tq=0.126), # coef=0.0015
        JointSample(motor_name='EC-i 40 100W 36V 390g', Jm=44.0*1e-7, motor_max_tq=0.204), # coef=0.0011
        JointSample(motor_name='EC-i 52 180W 36V 823g', Jm=170.0*1e-7, motor_max_tq=0.436), # coef=0.00089
    )
    esi5.plot_sample_values(value_range, fname=head_fname+'_EC-i'+param_str+ext)

    # EC-max
    esi5.joint_samples = (
        JointSample(motor_name='EC-max 30 60W 36V 195g', Jm=11.0*1e-7, motor_max_tq=0.0357), # coef=0.0086
        JointSample(motor_name='EC-max 30 60W 36V 305g', Jm=21.9*1e-7, motor_max_tq=0.0675), # coef=0.0048
        JointSample(motor_name='EC-max 40 60W 36V 460g', Jm=51.2*1e-7, motor_max_tq=0.102), # coef=0.0049
    )
    esi5.plot_sample_values(value_range, fname=head_fname+'_EC-max'+param_str+ext)

    # ILM
    esi5.joint_samples = (
        # JointSample(motor_name='ILM50x14 135g', Jm=0.086*1e-4, motor_max_tq=0.50), # coef=0.0000344
        JointSample(motor_name='ILM70x10 230g', Jm= 0.21*1e-4, motor_max_tq=0.74), # coef=0.0000383
        JointSample(motor_name='ILM70x18 340g', Jm= 0.34*1e-4, motor_max_tq=1.25), # coef=0.0000218
        # JointSample(motor_name='ILM85x04 200g', Jm= 0.28*1e-4, motor_max_tq=0.43), # coef=0.000154
        JointSample(motor_name='ILM85x13 370g', Jm= 0.61*1e-4, motor_max_tq=1.43), # coef=0.0000298
        JointSample(motor_name='ILM85x23 550g', Jm= 0.98*1e-4, motor_max_tq=2.3 ), # coef=0.0000185
        JointSample(motor_name='ILM85x26 590g', Jm= 1.15*1e-4, motor_max_tq=2.6 ), # coef=0.0000170
    )
    esi5.plot_sample_values(value_range, fname=head_fname+'_ILM'+param_str+ext)

def export_K_map(ext='.pdf'):
    head_fname = 'spesific-motor-gear-joint-impact-map'
    format_str = 'm{m:.0f}_l{leg:.2g}_Dj{Dj:g}_Fj{Fj:g}_Tjump{Tjump}_alpha{alpha}'
    joint_samples = (
        # JointSample(motor_name='EC-4pole 30 100W 36V 175g', Jm=8.91*1e-7, motor_max_tq=0.0564, gear_ratio=229),
        # JointSample(motor_name='EC-4pole 30 100W 36V 210g', Jm=18.3*1e-7, motor_max_tq=0.0780, gear_ratio=229),
        JointSample(motor_name='EC-4pole 30 200W 36V 300g', Jm=33.3*1e-7, motor_max_tq=0.104, gear_ratio=229),
        # JointSample(motor_name='EC-4pole 30 200W 36V double motor 600 g', Jm=2*33.3*1e-7, motor_max_tq=2*0.104, gear_ratio=229),
    )

    value_range = (('J', [joint_sample.Jjoint() for joint_sample in joint_samples]),
                   ('K', [1.0,10,100,500,1000,2000,3000,6000,15000,25000,35000,47000]))

    Dl = 0.07 # 292*240(gear ratio)*2e-20 from JAXON's fsharp
    F = 12.0 # 820*240(gear ratio)*2e-14 from JAXON's fsharp

    global esi
    esi = ExhaustiveSearchInterface()
    esi.joint_samples = joint_samples
    param = esi.jia.param
    param.Dl = 0.0
    param_str = str.replace(format_str.format(**param.param_dict()), '.','-')
    logger.critical(param_str)
    esi.plot_motor_map( value_range, fname=head_fname+param_str+ext)

    param.Dl = Dl; param.F = F
    # param_str = str.replace(format_str.format(**param.param_dict()), '.','-')
    # logger.critical(param_str)
    # esi.plot_motor_map( value_range, fname=head_fname+param_str+ext)

if __name__ == '__main__':
    # ext = '.svg'
    ext = '.pdf'

    # logger.setLevel(CRITICAL)
    # export_JKM_map(ext=ext)

    logger.setLevel(INFO)
    export_joint_sample_map(ext=ext)

    export_K_map()
