from imports import *


class Reconstruction:

    def __init__(self, args):
        self.name = args.target_file[args.target_file.rfind('/') + 1:]
        self.image_data = loadmat(args.target_file + '.mat')

        if args.continue_reconstruction is not None:
            self.output_dir = args.continue_reconstruction + '/'
        else:
            self.output_dir = 'data/output' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S" + '/')
            temp_filename = self.name
            temp_filename_counter = 1
            while True:
                try:
                    self.output_dir = 'data/' + temp_filename + '/'
                    os.mkdir(self.output_dir)
                    break

                except:
                    temp_filename = self.name + '_' + str(temp_filename_counter).zfill(2)
                    temp_filename_counter += 1

        self.save = args.save

        self.ground_truth = None
        if args.ground_truth_file is not None:
            self.ground_truth = loadmat(args.ground_truth_file + '.mat')['micrograph']

        self.a_lines = None
        self.pos_data = None
        self.pp_image_data = None

        self.target_resolution = None
        self.x = None
        self.y = None
        self.low_res_reconstruction = None

        self.operator = None
        self.zero_padding_operator = None
        self.psf_response_operator = None

        self.data_space = None
        self.data = None

        self.non_padded_position_space = None
        self.position_space = None
        self.signal = None

        self.correlated_field = None
        self.optimizer = None

        self.reconstruction = None
        self.uncertainty = None

    def run_initialization(self, target_resolution):
        self.pp_image_data = self.image_data['micrograph']

        img_limits_x = (self.image_data['xdata'][0][0] / 1e3, self.image_data['xdata'][0][-1] / 1e3)
        img_limits_y = (self.image_data['ydata'][0][0] / 1e3, self.image_data['ydata'][0][-1] / 1e3)

        self.x = np.linspace(*img_limits_x, target_resolution, endpoint=True)
        self.y = np.linspace(*img_limits_y, target_resolution, endpoint=True)
        self.target_resolution = target_resolution

        self.low_res_reconstruction = np.zeros(shape=(target_resolution, target_resolution))
        self.low_res_reconstruction[::int(target_resolution / self.pp_image_data.shape[0]), :] = self.pp_image_data

        plot_image(self.low_res_reconstruction, self.output_dir + 'sparse_data', fov_size=args.fov_size, cmap='bone')

        if self.ground_truth is not None:
            plot_image(self.ground_truth, self.output_dir + 'ground_truth', fov_size=args.fov_size)

    def construct_operators(self, psf_std, fov_size):
        self.operator = Operator()
        self.operator.generate_operator_matrices(self.pp_image_data)
        self.operator.generate_gaussian_psf(self.x, psf_std, filename=self.output_dir + 'psf')

        position_space = ift.RGSpace([self.x.shape[0] + self.operator.gaussian_psf.shape[0] - 1,
                                      self.y.shape[0] + self.operator.gaussian_psf.shape[1] - 1],
                                     distances=0.002 * fov_size)
        self.non_padded_position_space = position_space

        self.zero_padding_operator = ift.FieldZeroPadder(position_space, new_shape=(
            position_space.shape[0] + int(position_space.shape[0] / 8),
            position_space.shape[1] + int(position_space.shape[1] / 8)))
        padded_position_space = self.zero_padding_operator.target[0]

        data_space = ift.RGSpace([self.pp_image_data.shape[0], self.pp_image_data.shape[1]])
        self.data = ift.makeField(data_space, self.pp_image_data)

        resolution_factor = int(self.target_resolution / self.pp_image_data.shape[0])
        self.psf_response_operator = PSFResponseOperator(position_space, data_space, resolution_factor,
                                                         self.operator.sparse_mats, self.operator.gaussian_psf)

        ift.extra.check_linear_operator(self.psf_response_operator)

        backprojection_image = self.psf_response_operator.adjoint_times(self.data).val
        plot_image(backprojection_image, self.output_dir + 'backprojection', fov_size=args.fov_size, cmap='bone')

        self.data_space = data_space
        self.position_space = padded_position_space

    def initialize_correlated_field(self, matern, prior_type):
        self.correlated_field = Field(self.position_space, matern, prior_type)

        if prior_type == 'log-normal':
            self.signal = (self.zero_padding_operator.adjoint @ self.correlated_field.correlated_field).exponentiate(10)
        elif prior_type == 'sigmoid-normal':
            contraction = ift.ContractionOperator(self.non_padded_position_space, None)
            normal = ift.NormalTransform(0.5, 2.5e-1, key='shift')
            self.signal = ((contraction.adjoint @ normal) + (
                    self.zero_padding_operator.adjoint @ self.correlated_field.correlated_field)).sigmoid().scale(
                np.max(self.pp_image_data) + 0.01)

        fields = [self.signal(ift.from_random(self.signal.domain)).val for _ in range(4)]
        for i, field in enumerate(fields):
            plot_image(field, self.output_dir + 'prior_sample' + '_' + str(i), fov_size=args.fov_size,
                       norm=matplotlib.colors.Normalize(vmin=np.min(self.pp_image_data), vmax=np.max(self.pp_image_data)))

    def reconstruct_image(self, dark_noise, reconstruction_type):
        noise = ift.ScalingOperator(self.data_space, dark_noise, np.float64)

        self.optimizer = Optimizer(self.data, noise, self.psf_response_operator @ self.signal, reconstruction_type)
        self.optimizer.reconstruct_image(self.output_dir + 'optimization', self.signal)
        self.optimizer.likelihood_energy = None

        samples = [self.signal(self.optimizer.samples.local_item(i)).val for i in range(4)]
        for i, field in enumerate(samples):
            plot_image(field, self.output_dir + 'posterior_sample' + '_' + str(i), fov_size=args.fov_size,
                       norm=matplotlib.colors.Normalize(vmin=np.min(field), vmax=np.max(field)))
        plot_image(self.optimizer.mean.val, self.output_dir + 'posterior_mean', fov_size=args.fov_size)
        plot_image(self.optimizer.var.val, self.output_dir + 'posterior_var', fov_size=args.fov_size)

    def postprocess_reconstruction(self):
        temp = self.signal(self.optimizer.samples.local_item(0)).val
        self.reconstruction = temp[int(self.operator.gaussian_psf.shape[0] / 2):-int(
            self.operator.gaussian_psf.shape[0] / 2), int(self.operator.gaussian_psf.shape[0] / 2):-int(
            self.operator.gaussian_psf.shape[0] / 2)]

        self.reconstruction = self.optimizer.mean.val[int(self.operator.gaussian_psf.shape[0] / 2):-int(
            self.operator.gaussian_psf.shape[0] / 2), int(self.operator.gaussian_psf.shape[0] / 2):-int(
            self.operator.gaussian_psf.shape[0] / 2)]

        self.uncertainty = self.optimizer.var.val[
                           int(self.operator.gaussian_psf.shape[0] / 2):-int(self.operator.gaussian_psf.shape[0] / 2),
                           int(self.operator.gaussian_psf.shape[0] / 2):-int(self.operator.gaussian_psf.shape[0] / 2)]

        if self.ground_truth is not None:
            x = np.linspace(0, 1, self.reconstruction.shape[0])
            y = np.linspace(0, 1, self.reconstruction.shape[1])

            final_int = RegularGridInterpolator((x, y), self.reconstruction)
            x = np.linspace(0, 1, self.ground_truth.shape[0])
            y = np.linspace(0, 1, self.ground_truth.shape[1])
            X, Y = np.meshgrid(x, y)
            self.reconstruction = np.transpose(final_int((X, Y)))

            x = np.linspace(0, 1, self.uncertainty.shape[0])
            y = np.linspace(0, 1, self.uncertainty.shape[1])

            final_int = RegularGridInterpolator((x, y), self.uncertainty)
            x = np.linspace(0, 1, self.ground_truth.shape[0])
            y = np.linspace(0, 1, self.ground_truth.shape[1])
            X, Y = np.meshgrid(x, y)
            self.uncertainty = np.transpose(final_int((X, Y)))

        plot_image(self.reconstruction, self.output_dir + 'reconstruction', fov_size=args.fov_size)
        plot_image(self.uncertainty, self.output_dir + 'uncertainty', fov_size=args.fov_size)

    def compute_metrics(self):
        if self.ground_truth is not None:
            pcc = np.corrcoef(np.ravel(self.reconstruction), np.ravel(self.ground_truth))[0, 1]
            rmse = np.sqrt(np.square(np.subtract(self.reconstruction, self.ground_truth)).mean())
            psnr = 20 * np.log10(np.max(self.ground_truth) / rmse)
            ssim = structural_similarity(self.reconstruction, self.ground_truth, data_range=np.max(self.ground_truth))

            console_output = 'Pearson Correlation Coefficient (PCC): {:.4}\nRoot Mean Squared Error (RMSE):        {:.4}\nPeak SNR (PSNR):                       {:.4} dB\nStructural Similarity (SSIM):          {:.4}'.format(pcc, rmse, psnr, ssim)

            print('\n' + console_output)

            overlay_images(self.ground_truth, self.reconstruction, self.output_dir + 'overlay', fov_size=args.fov_size)

            return console_output


def control(args):
    if args.existing_reconstruction:
        file = open(args.existing_reconstruction + '/optimization/reconstruction.obj', 'rb')
        reconstruction = pickle.load(file)
        file.close()

        reconstruction.save = False
        reconstruction.output_dir = args.existing_reconstruction + '/'
        reconstruction.run_initialization(target_resolution=args.reconstruction_res)
        reconstruction.construct_operators(args.psf_std, args.fov_size)
        reconstruction.initialize_correlated_field(matern=args.matern, prior_type=args.prior_type)

        samples = [reconstruction.signal(reconstruction.optimizer.samples.local_item(i)).val for i in range(4)]
        for i, field in enumerate(samples):
            plot_image(field, reconstruction.output_dir + 'posterior_sample' + '_' + str(i), fov_size=args.fov_size,
                       norm=matplotlib.colors.Normalize(vmin=np.min(reconstruction.low_res_reconstruction),
                                                        vmax=np.max(reconstruction.low_res_reconstruction)))
        plot_image(reconstruction.optimizer.mean.val, reconstruction.output_dir + 'posterior_mean',
                   fov_size=args.fov_size)
        plot_image(reconstruction.optimizer.var.val, reconstruction.output_dir + 'posterior_var',
                   fov_size=args.fov_size)

        reconstruction.postprocess_reconstruction()
        quality_metrics_string = reconstruction.compute_metrics()

    else:
        reconstruction = Reconstruction(args)
        reconstruction.run_initialization(target_resolution=args.reconstruction_res)
        reconstruction.construct_operators(args.psf_std, args.fov_size)
        reconstruction.initialize_correlated_field(matern=args.matern, prior_type=args.prior_type)

        t_start = time.time()
        reconstruction.reconstruct_image(args.dark_noise, args.reconstruction)
        reconstruction.postprocess_reconstruction()
        t_end = time.time()

        quality_metrics_string = reconstruction.compute_metrics()
        comp_time_string = 'Time needed for reconstruction:        {}s'.format(np.round(t_end - t_start, 2))
        print('\n' + comp_time_string)

        with open(reconstruction.output_dir + 'Details.txt', 'w') as text_file:
            text_file.write(comp_time_string)
            text_file.write('\n\n' + quality_metrics_string)

        if args.save:
            file = open(reconstruction.output_dir + 'optimization/reconstruction.obj', 'wb')
            pickle.dump(reconstruction, file)
            file.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('-existing_reconstruction', type=str, default=None, help='load reconstruction')
    p.add_argument('-continue_reconstruction', type=str, default=None, help='load and continue reconstruction')
    p.add_argument('-target_file', type=str, default='data/sample-image_2856_sparse', help='file containing image data')
    p.add_argument('-ground_truth_file', type=str, default='data/sample-image_2856', help='file containing ground truth image')
    p.add_argument('-reconstruction_res', type=int, default=400, help='pixel count of posterior')
    p.add_argument('-psf_std', type=float, default=0.005, help='PSF standard deviation')
    p.add_argument('-dark_noise', type=float, default=5e-06, help='dark noise of the system')
    p.add_argument('-matern', type=bool, default=True, help='use of matern kernel')
    p.add_argument('-fov_size', type=float, default=1, help='fov size in mm')
    p.add_argument('-prior_type', type=str, default='sigmoid-normal', help='use of log-normal or sigmoid-normal priors')
    p.add_argument('-reconstruction', type=str, default='fast', help='fast or full reconstruction')
    p.add_argument('-save', type=bool, default=False, help='saving the results can be disabled')

    args = p.parse_args()
    control(args)

    print('\nFinished [' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ']')
