import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy

def main(plots = False, directory = None):
    filename_base = 'centerline.'

    if directory is not None:
        filename_base = os.path.join(directory, filename_base)
    else:
        if len(sys.argv) < 2:
            raise ValueError('Usage: python <filename> <directory of probes> <plotting probes (min, max, step), optional> <forcing file (optional)> <forcing write step>')
        filename_base =os.path.join(sys.argv[1], filename_base)
    try:
        inds = sys.argv[2]
        inde = sys.argv[3]
        indstep = sys.argv[4]
    except:
        inds=0
        inde=1
        indstep=1

    solutions = {}
    rho = 1.225
    c = 340.29
    dt = 1e-6

    # read README to check probe positions
    README = filename_base + 'README'

    with open(README, 'r') as f:
        for line in f:
            if line.startswith('Parameter line: PROBE'):
                parts = line.split()
                # GEOM LINE x0 y0 dx y1 y1 dy n
                geom_idx = parts.index('GEOM')
                if parts[geom_idx + 1] == 'LINE':
                    x0 = float(parts[geom_idx + 2])
                    y0 = float(parts[geom_idx + 3])
                    z0 = float(parts[geom_idx + 4])
                    x1 = float(parts[geom_idx + 5])
                    y1 = float(parts[geom_idx + 6])
                    z1 = float(parts[geom_idx + 7])
                    n = int(parts[geom_idx + 8])

    x = np.linspace(x0, x1, n)
    y = np.linspace(y0, y1, n)
    z = np.linspace(z0, z1, n)

    print(f'parsing {n} probe positions: (x0, x1, y0, y1, z0, z1) = {x0:.3f}, {x1:.3f}, {y0:.3f}, {y1:.3f}, {z0:.3f}, {z1:.3f}')
    dx = x[1] - x[0] # uniform

    for label in ['P','U-X','U_AVG-X', 'P_AVG']:
        filename = filename_base + label
        data = np.loadtxt(filename, delimiter=None, skiprows=2).T[3:, :]
        time = np.loadtxt(filename, delimiter=None, skiprows=2).T[1, :]

        solutions[label] = data

    # print(time)
    # print(time.shape)
    # print(data.shape)

    solutions['P_AVG'] = solutions['P_AVG'][-1, :]
    solutions['U_AVG-X'] = solutions['U_AVG-X'][-1, :]
    # print(mean_u_y)

    f = 0.5 * ((solutions['P']-solutions['P_AVG']) / rho / c + (solutions['U-X']-solutions['U_AVG-X']))
    g = 0.5 * ((solutions['P']-solutions['P_AVG']) / rho / c - (solutions['U-X']-solutions['U_AVG-X']))

    # print(f.shape)

    f_func = scipy.interpolate.interp1d(time, f, axis=1, fill_value=(np.zeros(f.shape[0]), np.zeros(f.shape[0])), bounds_error=False)
    g_func = scipy.interpolate.interp1d(time, g, axis=1, fill_value=(np.zeros(g.shape[0]), np.zeros(g.shape[0])), bounds_error=False)


    # print(f_func(time - 40 * dy / c).shape)

    # TODO: triple check
    # fa = np.sum(
    #     np.array([[f_func(time - i * dy / (c+mean_u_y))[index-i, :] if n>index-i>=0 else np.zeros(time.shape) for i in range(0, n)] for index in range(n)]),
    #     axis=0
    # )

    # fas = [[f_func(time - i * dy / (c+mean_u_y))[index-i, :] if n>index-i>=0 else np.zeros(time.shape) for i in range(0, n)] for index in range(n)]

    # fa = fa / np.tile(np.arange(1, n+1), (fa.shape[1], 1)).T


    # EXPLICIT
    # fa = np.zeros((n, time.shape[0]))
    # for index in range(n):
    #     probe_count = 0
    #     for shift in range(index+1):
    #         probe_count += 1
    #         timeshift = shift * dx / (c)
    #         fa[index, :] += f_func(time - timeshift)[index-shift, :]  # shape (1, time.shape)
    #     fa[index, :] = fa[index, :] / probe_count # mean over the counted probes

    # ga = np.zeros((n, time.shape[0]))
    # for index in range(n):
    #     probe_count = 0
    #     for shift in range(0, n-index):
    #         probe_count += 1
    #         timeshift = shift * dx / (c)
    #         ga[index, :] += g_func(time - timeshift)[index+shift, :]  # shape (1, time.shape)
    #     ga[index, :] = ga[index, :] / probe_count # mean over the counted probes

    #VECTORIZED
    T = time.shape[0]
    shifts = np.arange(n)
    timeshifts = shifts * dx / (c)  # shape (n,)

    # build shifted time arrays: shape (n, T)
    time_shifted = time[None, :] - timeshifts[:, None]  # (n, T)

    # evaluate f_func and g_func at each shifted time
    # -> (n, n, T): [shift, probe, time]
    F = np.array([f_func(t)[None, :, :] for t in time_shifted])  # shape (n, 1, n, T)
    F = np.squeeze(F, 1)  # (n, n, T)

    G = np.array([g_func(t)[None, :, :] for t in time_shifted])
    G = np.squeeze(G, 1)  # (n, n, T)

    # --- fa ---
    fa = np.zeros((n, T))
    for idx in range(n):
        valid_shifts = np.arange(idx+1)
        fa[idx, :] = F[valid_shifts, idx - valid_shifts, :].mean(axis=0)

    # --- ga ---
    ga = np.zeros((n, T))
    for idx in range(n):
        valid_shifts = np.arange(n - idx)
        ga[idx, :] = G[valid_shifts, idx + valid_shifts, :].mean(axis=0)


    # ga = np.sum(
    #     np.array([[g_func(time - i * dy / (c-mean_u_y))[index+i, :] if 0<=index+i<n else np.zeros(time.shape) for i in range(0, n)] for index in range(n)]),
    #     axis=0
    # )
    # ga = ga / np.tile(np.arange(1, n+1), (ga.shape[1], 1)).T

    ft  = f - fa
    gt = g - ga

    ua = fa - ga
    pa = rho * c * (fa + ga)

    # ut = (ft - gt)
    # pt = (ft + gt) * ( rho * c)
    
    # for var in [solutions['U-X']-solutions['U_AVG-X'], ua, ut, solutions['P']-solutions['P_AVG'], pa, pt]:
    #     print(f'min/mean/max: {var.min():.8f}/{var.mean():.8f}/{var.max():.8f}')

    if not plots:
        return ua, pa, fa, ga, f, g, solutions['P']-solutions['P_AVG'], solutions['U-X']-solutions['U_AVG-X'], time, x, y, z


    fig, ax = plt.subplots()
    for ffa, gaa, ff, gg, color, xx in zip(fa[int(inds):int(inde):int(indstep), :], ga[int(inds):int(inde):int(indstep), :],
                                        f[int(inds):int(inde):int(indstep), :], g[int(inds):int(inde):int(indstep), :],
                                        plt.cm.viridis(np.linspace(0,1,len(range(int(inds),int(inde),int(indstep))))), x[int(inds):int(inde):int(indstep)]):
        ax.plot(time, ffa, color=color, label=f'x={xx:.5f}')
        ax.plot(time, gaa, color=color, linestyle='dotted')
        # ax.plot(time, ff, color=color, linestyle='dashed')
        # ax.plot(time, gg, color=color, linestyle='dashed')

        # ax.plot(time, fa, label='$f_a$', color='r')
        # ax.plot(time, ga, label='$g_a$', color='b')
        # ax.plot(time, f, label='$f$', color='r', linestyle='dashed')
        # ax.plot(time, g, label='$g$', color='b', linestyle='dashed')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [m/s]')
    ax.grid()
    ax.legend()

    if plots==True:
        plt.show()
    plt.close(fig)

    #plot fa and ga separately on imshow maps on the y-t plane
    for dataframe, title in zip([f, g, solutions['U-X']-solutions['U_AVG-X'],solutions['P']-solutions['P_AVG'], fa, ga, ua, pa], ['$f$', '$g$', '$u$', '$p$', '$f_a$', '$g_a$', '$u_a$', '$p_a$']):
        fig, ax = plt.subplots()
        cbar = ax.imshow(np.abs(dataframe), extent=[time[0], time[-1], x[0], x[-1]], aspect='auto', origin='lower', cmap='RdBu', norm=matplotlib.colors.LogNorm())

        Nlines = 10
        for index in range(Nlines):
            ax.plot(time[0] + x/(c) + (time[-1] - time[0])/ Nlines * index, x, linestyle='dashed', color='k', alpha=0.3)
            ax.plot(time[0] + (x1- x)/(c) + (time[-1] - time[0])/ Nlines * index, x, linestyle='dashed', color='k', alpha=0.3)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Probe Position [m]')
        fig.colorbar(cbar, ax=ax, orientation='vertical', extend='both')
        fig.suptitle(title)
        plt.show()
        plt.close(fig)



    fig, ax = plt.subplots()
    for uua,uuaref, ppa, ff, gg, color, xx in zip(ua[int(inds):int(inde):int(indstep), :], solutions['U-X'][int(inds):int(inde):int(indstep), :], pa[int(inds):int(inde):int(indstep), :],
                                        f[int(inds):int(inde):int(indstep), :], g[int(inds):int(inde):int(indstep), :],
                                        plt.cm.viridis(np.linspace(0,1,len(range(int(inds),int(inde),int(indstep))))), x[int(inds):int(inde):int(indstep)]):
        ax.plot(time, uua, label=f'x={xx:.5f}', color=color)
        # ax.plot(time, ppa / rho / c, color=color, linestyle='dashed')
        ax.plot(time, uuaref - solutions['U_AVG-X'], color=color, linestyle='dashed')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')

    ax.legend()
    ax.grid()
    if plots==True:
        plt.show()
    plt.close(fig)

    try:
        forcing_file = sys.argv[5]
        # save_interval = int(sys.argv[6])
    except:
        return ua, pa, fa, ga, f, g, solutions['P']-solutions['P_AVG'], solutions['U-X']-solutions['U_AVG-X'], time, x, y, z



    s = np.loadtxt(forcing_file, delimiter=',')[np.array(time//dt, dtype=int)]

    fig, ax = plt.subplots(figsize=(4,3))
    for uua,uuaref, ppa, ff, gg, color, xx in zip(ua[int(inds):int(inde):int(indstep), :], solutions['U-X'][int(inds):int(inde):int(indstep), :], pa[int(inds):int(inde):int(indstep), :],
                                        f[int(inds):int(inde):int(indstep), :], g[int(inds):int(inde):int(indstep), :],
                                        plt.cm.viridis(np.linspace(0,1,len(range(int(inds),int(inde),int(indstep))))), x[int(inds):int(inde):int(indstep)]):
        ax.plot(time - (xx)/c, uua, label=f'x={xx:.5f}', color=color)

    ax.plot(time, s, label='forcing', color='k', linestyle='dashed')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')

    ax.legend()
    ax.grid()
    if plots==True:
        plt.show()
    plt.close(fig)


    # fig, ax = plt.subplots(figsize=(4,3))
    # for uua,uuaref, ppa, ff, gg, color, yy in zip(ua[int(inds):int(inde):int(indstep), :], solutions['U-X'][int(inds):int(inde):int(indstep), :], pa[int(inds):int(inde):int(indstep), :],
    #                                     f[int(inds):int(inde):int(indstep), :], g[int(inds):int(inde):int(indstep), :],
    #                                     plt.cm.viridis(np.linspace(0,1,len(range(int(inds),int(inde),int(indstep))))), y[int(inds):int(inde):int(indstep)]):
    #     ax.plot(time, uuaref - solutions['U_AVG-X'], color=color, linestyle='dashed')


    # ax.plot(time, s, label='forcing', color='k', linestyle='dashed')

    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Amplitude')

    # ax.legend()
    # ax.grid()
    # if plots==True:
    #     plt.show()
    # plt.close(fig)

    return ua, pa, fa, ga, f, g, solutions['P']-solutions['P_AVG'], solutions['U-X']-solutions['U_AVG-X'], time, x, y, z

if __name__=="__main__":
    main(plots=True)

