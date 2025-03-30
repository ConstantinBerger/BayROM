from imports import *


class Field:

    def __init__(self, position_space, matern, prior_type):
        self.matern = matern
        self.m_pars = None
        self.fl_pars = None

        self.get_mpars(prior_type)
        self.get_flpars(position_space, prior_type)

        self.cf = ift.CorrelatedFieldMaker(self.m_pars['prefix'])
        self.cf.set_amplitude_total_offset(self.m_pars['offset_mean'], self.m_pars['offset_std'])
        self.cf.add_fluctuations_matern(**self.fl_pars) if matern else self.cf.add_fluctuations(**self.fl_pars)

        self.correlated_field = self.cf.finalize(prior_info=0)
        self.pspec = self.cf.power_spectrum

    def get_mpars(self, prior_type):
        if prior_type == 'log-normal':
            self.m_pars = {'offset_mean': np.log10(1e-1), 'offset_std': (1e-1, 1e-7), 'prefix': ''}
        elif prior_type == 'sigmoid-normal':
            self.m_pars = {'offset_mean': 0, 'offset_std': (1e-16, 1e-16), 'prefix': ''}

    def get_flpars(self, position_space, prior_type):
        if self.matern:
            if prior_type == 'log-normal':
                self.fl_pars = {'target_subdomain': position_space, 'scale': (2.5e-2, 2e-2), 'cutoff': (5e0, 5e-1),
                                'loglogslope': (-6.5e0, 5e-1)}
            elif prior_type == 'sigmoid-normal':
                self.fl_pars = {'target_subdomain': position_space, 'scale': (2.5e-1, 5e-1), 'cutoff': (1e1, 1e1),
                                'loglogslope': (-6.5e0, 5e-1)}
        else:
            self.fl_pars = {'target_subdomain': position_space, 'fluctuations': (1e0, 1e-1),
                            'flexibility': (5e-1, 1e-1), 'asperity': (5e-1, 1e-1), 'loglogavgslope': (-3.5e0, 1e0)}
