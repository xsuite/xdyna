import matplotlib.pyplot as plt



def plot_particles(DA, ax=None, at_turn=None, type_plot="polar", seed=None,
                   closses="red", csurviving="blue", size_scaling="log", **kwargs):
    """Scatter plot of the lost and surviving particles.

Parameters
----------
ax:           Plot axis.
at_turn:      All particles surviving at least this number of turns are considered as surviving.
seed:         In case of multiseed simulation, the seed number must be specified (Default=None).
type_plot:    x-y for cartesian, ang-amp for polar (Default="polar").
csurviving:   Color of surviving dots (Default="blue"). Use "" to disable.
closses:      Color of losses dots (Default="red"). Use "" to disable.
size_scaling: Type of losses dot scaling (Default="log"). There are 3 options: "linear", "log", None.
"""

    if ax is None:
        ax = plt.subplots(1,1,figsize=(10,10))[1]

    if DA.survival_data is None:
        raise ValueError('Run the simulation before using plot_particles.')
    if DA.meta.nseeds>0 and (seed==None or seed=='stat'):
        raise ValueError('For multiseed simulation, please specify a seed number.')

    # Clean kwargs and initiallize parameters
    if 'c' in kwargs:
        kwargs.pop('c')
        print("Warning: ignoring parameter 'c'. Use 'closses' and 'csurviving' instead.")
    if 'color' in kwargs:
        kwargs.pop('color')
        print("Warning: ignoring parameter 'color'. Use 'closses' and 'csurviving' instead.")
    if 'label' in kwargs:
        print("Warning: ignoring parameter 'label'.")
        kwargs.pop('label')

    if at_turn is None:
        at_turn=DA.max_turns
    if DA.meta.nseeds==0:
        data = DA.survival_data.copy()
    else:
        data = DA.survival_data[DA.survival_data.seed==seed].copy()

    if type_plot=="polar":
        if "angle" not in data.columns or "amplitude" not in data.columns:
            data['angle']    = np.arctan2(data['y'],data['x'])*180/np.pi
            data['amplitude']= np.sqrt(data['x']**2+data['y']**2)

        if csurviving is not None and csurviving!='':
            surv=data.loc[data['nturns']>=at_turn,:]
            ax.scatter(surv['angle'], surv['amplitude'], color=csurviving, \
                       label="Surv.", **kwargs)

        if closses is not None and closses!='':
            loss=data.loc[data['nturns']<at_turn,:]
            if size_scaling in ["linear","log"]:
                if size_scaling=="linear":
                    size=(loss['nturns'].to_numpy()/at_turn) * plt.rcParams['lines.markersize']
                elif size_scaling=="log":
                    loss=loss.loc[loss['nturns']>0,:]
                    size=(np.log10(loss['nturns'].to_numpy())/np.log10(at_turn)) *\
                            plt.rcParams['lines.markersize']
                ax.scatter(loss['angle'], loss['amplitude'], size**2, color=closses, \
                           label="Loss.", **kwargs)
            else:
                ax.scatter(loss['angle'], loss['amplitude'], color=closses, \
                           label="Loss.", **kwargs)
            ax.set_xlabel(r'Angle [$^{\circ}$]')
            ax.set_ylabel(r'Amplitude [$\sigma$]')

    elif type_plot=="cartesian":
        if "x" not in data.columns or "y" not in data.columns:
            data['x']= data['amplitude']*np.cos(data['angle']*np.pi/180)
            data['y']= data['amplitude']*np.sin(data['angle']*np.pi/180)

        if csurviving is not None and csurviving!='':
            surv=data.loc[data['nturns']>=at_turn,:]
            ax.scatter(surv['x'], surv['y'], color=csurviving, label="Surv.", \
                       **kwargs)

        if closses is not None and closses!='':
            loss=data.loc[data['nturns']<at_turn,:]
            if size_scaling in ["linear","log"]:
                if size_scaling=="linear":
                    size=(loss['nturns'].to_numpy()/at_turn) * plt.rcParams['lines.markersize']
                elif size_scaling=="log":
                    loss=loss.loc[loss['nturns']>0,:]
                    size=(np.log10(loss['nturns'].to_numpy())/np.log10(at_turn)) *\
                            plt.rcParams['lines.markersize']
                ax.scatter(loss['x'], loss['y'], size**2, color=closses, label="Loss.", **kwargs)
            else:
                ax.scatter(loss['x'], loss['y'], color=closses, label="Loss.", **kwargs)
            ax.set_xlabel(r'x [$\sigma$]')
            ax.set_ylabel(r'y [$\sigma$]')

    else:
        raise ValueError('type_plot can only be either "polar" or "cartesian".')


def plot_da_border(DA, ax=None, at_turn=None, seed=None, type_plot="polar", clower="blue", cupper="red", **kwargs):
    """Plot the . border.

Parameters
----------
ax:        Plot axis.
at_turn:   All particles surviving at least this number of turns are considered as surviving.
seed:      In case of multiseed simulation, the seed number must be specified (Default=None).
type_plot: x-y for cartesian, ang-amp for polar (Default="polar").
clower:    Color of the lower DA estimation (Default="blue"). Use "" to disable.
cupper:    Color of the upper DA estimation (Default="red"). Use "" to disable.
"""

#         if DA.meta.pairs_shift != 0:
#             raise NotImplementedError("The DA computing methods have not been implemented for pairs yet!")

    if ax is None:
        ax = plt.subplots(1,1,figsize=(10,10))[1]

    if DA.meta.nseeds>0 and seed==None:
        raise ValueError('For multiseed simulation, please specify a seed number.')
    if seed=='stat':
        raise ValueError('"stat" border is not computed yet.')

    # Clean kwargs and initiallize parameters
    if 'c' in kwargs:
        kwargs.pop('c')
        print("Warning: ignoring parameter 'c'. Use 'closses' and 'csurviving' instead.")
    if 'color' in kwargs:
        kwargs.pop('color')
        print("Warning: ignoring parameter 'color'. Use 'closses' and 'csurviving' instead.")
    label = kwargs.pop('label', '')

    if at_turn is None:
        at_turn=DA.max_turns
    if DA._lower_davsturns is None:
        DA.calculate_da(at_turn=at_turn,angular_precision=1,smoothing=True)

    if "angle" not in DA.survival_data.columns:
        angle= np.arctan2(DA.survival_data['y'],DA.survival_data['x'])*180/np.pi
    else:
        angle= np.array(DA.survival_data.angle)
    ang_range=(min(angle),max(angle))

    if DA.meta.nseeds==0:
        lower_da=DA._lower_davsturns
        upper_da=DA._upper_davsturns
    else:
        lower_da=DA._lower_davsturns[seed]
        upper_da=DA._upper_davsturns[seed]

    at_turn=max(lower_da.turn[lower_da.turn<=at_turn])

    fit_min=fit_DA(lower_da.loc[at_turn,'border'][0].angle, 
                   lower_da.loc[at_turn,'border'][0].amplitude, ang_range)
    fit_max=fit_DA(upper_da.loc[at_turn,'border'][0].angle, 
                   upper_da.loc[at_turn,'border'][0].amplitude, ang_range)

    amplitude_min=fit_min(angle)
    amplitude_max=fit_max(angle)
    sort = np.argsort(angle)
    angle= angle[sort]; amplitude_min = amplitude_min[sort]; amplitude_max = amplitude_max[sort]
    if type_plot=="polar":
        if clower is not None and clower!='':
            ax.plot(angle,amplitude_min,color=clower,label=label+' (min)',**kwargs)

        if cupper is not None and cupper!='':
            ax.plot(angle,amplitude_max,color=cupper,label=label+' (max)',**kwargs)

        ax.set_xlabel(r'Angle [$^{\circ}$]')
        ax.set_ylabel(r'Amplitude [$\sigma$]')

    elif type_plot=="cartesian":
        if clower is not None and clower!='':
            x= amplitude_min*np.cos(angle*np.pi/180)
            y= amplitude_min*np.sin(angle*np.pi/180)
            ax.plot(x,y,color=clower,label=label+' (min)',**kwargs)

        if cupper is not None and cupper!='':
            x= amplitude_max*np.cos(angle*np.pi/180)
            y= amplitude_max*np.sin(angle*np.pi/180)
            ax.plot(x,y,color=cupper,label=label+' (max)',**kwargs)

        ax.set_xlabel(r'x [$\sigma$]')
        ax.set_ylabel(r'y [$\sigma$]')

    else:
        raise ValueError('type_plot can only be either "polar" or "cartesian".')


def plot_davsturns_border(DA, ax=None, from_turn=1e3, to_turn=None, seed=None, clower="blue", cupper="red", 
                          show_seed=True, show_Nm1=True, **kwargs):
    """Plot the DA as a function of turns.

Parameters
----------
ax:        Plot axis.
from_turn: Lower turn range (Default: from_turn=1e3).
at_turn:   Upper turn range (Default: at_turn=max_turns).
seed:      In case of multiseed simulation, the seed number must be specified (Default=None).
clower:    Color of the lower da vs turns stat. Set to '' will not show the plot (Default: "blue").
cupper:    Color of the upper da vs turns stat. Set to '' will not show the plot (Default: "red").
show_seed: Plot seeds (Default: True).
show_Nm1:  Plot davsturns as a stepwise function (Default: True).
"""

    if ax is None:
        ax = plt.subplots(1,1,figsize=(10,10))[1]

    if DA.meta.nseeds>0 and seed==None:
        raise ValueError('For multiseed simulation, please specify the seed.')
    if DA.meta.nseeds==0 and seed=='stat':
        raise ValueError('"stat" is only available for multiseed simulation.')

    # Clean kwargs and initiallize parameters
    if 'c' in kwargs:
        kwargs.pop('c')
        print("Warning: ignoring parameter 'c'. Use 'closses' and 'csurviving' instead.")
    if 'color' in kwargs:
        kwargs.pop('color')
        print("Warning: ignoring parameter 'color'. Use 'closses' and 'csurviving' instead.")
    label = kwargs.pop('label', '')
    label = kwargs.pop('alpha', 1)

    if to_turn is None:
        to_turn=DA.max_turns

    if DA._lower_davsturns is None:
        DA.calculate_davsturns(from_turn=from_turn,to_turn=to_turn)

    if DA.meta.nseeds==0:
        lower_da=DA._lower_davsturns
        upper_da=DA._upper_davsturns
    else:
        lower_da=DA._lower_davsturns[seed]
        upper_da=DA._upper_davsturns[seed]

    # Select the range of data
    lturns_data=np.array([t for t in lower_da.turn if t>=from_turn and t<=to_turn])
    lturns_data=lturns_data[np.argsort(lturns_data)]
    lturns_prev=[t-1 for t in lturns_data if t>from_turn and t<=to_turn]

    if cupper is not None and cupper!='':
        # Load Data
        davsturns_avg=upper_da.loc[lturns_data,'avg'] ;
        davsturns_min=upper_da.loc[lturns_data,'min'] ;
        davsturns_max=upper_da.loc[lturns_data,'max'] ;

        # Add step at turns-1
        if show_Nm1:
            for prev,turn in zip(lturns_prev, lturns_data[0:-1]):
                davsturns_avg[prev]=davsturns_avg[turn]
                davsturns_min[prev]=davsturns_min[turn]
                davsturns_max[prev]=davsturns_max[turn]

        lturns=np.array(davsturns_avg.index.tolist())
        lturns=lturns[np.argsort(lturns)]
        y_avg=np.array(davsturns_avg[lturns], dtype=float)
        y_min=np.array(davsturns_min[lturns], dtype=float)
        y_max=np.array(davsturns_max[lturns], dtype=float)

        # Plot the results
        ax.plot(lturns,y_avg,ls="-.",label=label,color=cupper,alpha=alpha,**kwargs);
        ax.plot(lturns,y_min,ls="-", label='',   color=cupper,alpha=alpha,**kwargs);
        ax.plot(lturns,y_max,ls="-", label='',   color=cupper,alpha=alpha,**kwargs);

        ax.fill_between(lturns,y_min, y_max,color=cupper,alpha=alpha*0.1,**kwargs)

        if seed=='stat' and show_seed:
            for s in range(1,DA.meta.nseeds+1):
                # Select the range of data
                slturns_data=np.array([t for t in DA._upper_davsturns[s].turn if t>=from_turn and t<=to_turn])
                slturns_data=slturns_data[np.argsort(slturns_data)]
                slturns_prev=[t-1 for t in slturns_data if t>from_turn and t<=to_turn]

                # Load Data
                davsturns_avg=DA._upper_davsturns[s].loc[slturns_data,'avg'] ;

                # Add step at turns-1
                if show_Nm1:
                    for prev,turn in zip(slturns_prev, slturns_data[0:-1]):
                        davsturns_avg[prev]=davsturns_avg[turn]

                lturns=np.array(davsturns_avg.index.tolist())
                lturns=lturns[np.argsort(lturns)]
                y_avg=np.array(davsturns_avg[lturns], dtype=float)

                # Plot the results
                ax.plot(lturns,y_avg,ls="-.",lw=1,label='',color=cupper,alpha=alpha*0.3,**kwargs);


    if clower is not None and clower!='':
        # Load Data
        davsturns_avg=lower_da.loc[lturns_data,'avg'] ;
        davsturns_min=lower_da.loc[lturns_data,'min'] ;
        davsturns_max=lower_da.loc[lturns_data,'max'] ;

        # Add step at turns-1
        if show_Nm1:
            for prev,turn in zip(lturns_prev, lturns_data[0:-1]):
                davsturns_avg[prev]=davsturns_avg[turn]
                davsturns_min[prev]=davsturns_min[turn]
                davsturns_max[prev]=davsturns_max[turn]

        lturns=np.array(davsturns_avg.index.tolist())
        lturns=lturns[np.argsort(lturns)]
        y_avg=np.array(davsturns_avg[lturns], dtype=float)
        y_min=np.array(davsturns_min[lturns], dtype=float)
        y_max=np.array(davsturns_max[lturns], dtype=float)

        # Plot the results
        ax.plot(lturns,y_avg,ls="-.",label=label,color=clower,alpha=alpha,**kwargs);
        ax.plot(lturns,y_min,ls="-", label='',   color=clower,alpha=alpha,**kwargs);
        ax.plot(lturns,y_max,ls="-", label='',   color=clower,alpha=alpha,**kwargs);

        ax.fill_between(lturns,y_min, y_max,color=clower,alpha=alpha*0.1,**kwargs)

        if seed=='stat' and show_seed:
            for s in range(1,DA.meta.nseeds+1):
                # Select the range of data
                slturns_data=np.array([t for t in DA._lower_davsturns[s].turn if t>=from_turn and t<=to_turn])
                slturns_data=slturns_data[np.argsort(slturns_data)]
                slturns_prev=[t-1 for t in slturns_data if t>from_turn and t<=to_turn]

                # Load Data
                davsturns_avg=DA._lower_davsturns[s].loc[slturns_data,'avg'] ;

                # Add step at turns-1
                if show_Nm1:
                    for prev,turn in zip(slturns_prev, slturns_data[0:-1]):
                        davsturns_avg[prev]=davsturns_avg[turn]

                lturns=np.array(davsturns_avg.index.tolist())
                lturns=lturns[np.argsort(lturns)]
                y_avg=np.array(davsturns_avg[lturns], dtype=float)

                # Plot the results
                ax.plot(lturns,y_avg,ls="-.",lw=1,label='',color=clower,alpha=alpha*0.3,**kwargs);

    ax.set_xlabel(r'Turns [1]')
    ax.set_ylabel(r'Amplitude [$\sigma$]')
