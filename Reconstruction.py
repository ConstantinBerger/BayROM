from imports import *


class Optimizer():

    def __init__(self, data, noise, signal_response, reconstruction_type):
        self.reconstruction_type = reconstruction_type
        self.likelihood_energy = (ift.GaussianEnergy(data, inverse_covariance=noise.inverse) @ signal_response)

        self.samples = None
        self.mean = None
        self.var = None

    def reconstruct_image(self, directory, signal):
        if self.reconstruction_type == 'full':
            n_iterations = 11
            n_samples = lambda iiter: 4 if iiter < 10 else 32
            self.samples = ift.optimize_kl(self.likelihood_energy, n_iterations, n_samples, minimizer, ic_sampling,
                                           None, export_operator_outputs={'signal': signal}, output_directory=directory,
                                           comm=None, resume=True)
            self.mean, self.var = self.samples.sample_stat(signal)
        if self.reconstruction_type == 'fast':
            n_iterations = 5
            n_samples = lambda iiter: 4 if iiter < 4 else 8
            self.samples = ift.optimize_kl(self.likelihood_energy, n_iterations, n_samples, minimizer_fast,
                                           ic_sampling_fast, None, export_operator_outputs={'signal': signal},
                                           output_directory=directory, comm=None, resume=True)
            self.mean, self.var = self.samples.sample_stat(signal)


def minimizer(iiter):
    if iiter < 5:
        minimizer_obj = ift.NewtonCG(
            ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, convergence_level=2, iteration_limit=10))
    elif iiter < 10:
        minimizer_obj = ift.NewtonCG(
            ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, convergence_level=2, iteration_limit=20))
    else:
        minimizer_obj = ift.NewtonCG(
            ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, convergence_level=2, iteration_limit=30))
    return minimizer_obj


def minimizer_fast(iiter):
    if iiter < 4:
        minimizer_obj = ift.NewtonCG(
            ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, convergence_level=2, iteration_limit=7))
    else:
        minimizer_obj = ift.NewtonCG(
            ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, convergence_level=2, iteration_limit=10))
    return minimizer_obj


def ic_sampling(iiter):
    if iiter < 5:
        ic_sampling_obj = ift.AbsDeltaEnergyController(name='Sampling (linear)', deltaE=0.05, iteration_limit=50)
    elif iiter < 10:
        ic_sampling_obj = ift.AbsDeltaEnergyController(name='Sampling (linear)', deltaE=0.05, iteration_limit=100)
    else:
        ic_sampling_obj = ift.AbsDeltaEnergyController(name='Sampling (linear)', deltaE=0.05, iteration_limit=200)
    return ic_sampling_obj


def ic_sampling_fast(iiter):
    ic_sampling_obj = ift.AbsDeltaEnergyController(name='Sampling (linear)', deltaE=0.05, iteration_limit=100)

    return ic_sampling_obj


def minimizer_sampling_nl(iiter):
    if iiter < 2:
        minimizer_sampling_nl_obj = ift.NewtonCG(
            ift.AbsDeltaEnergyController(name='Sampling (nonlin)', deltaE=0.5, iteration_limit=2, convergence_level=2))
    else:
        minimizer_sampling_nl_obj = ift.NewtonCG(
            ift.AbsDeltaEnergyController(name='Sampling (nonlin)', deltaE=0.5, iteration_limit=2, convergence_level=5))
    return minimizer_sampling_nl_obj
