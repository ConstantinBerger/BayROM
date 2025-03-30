from imports import *


class Operator:

    def __init__(self, offset=-0.0015, psf_resolution=6):
        self.offset = offset
        self.psf_res = psf_resolution
        self.sparse_mats = []
        self.gaussian_psf = None

    def generate_operator_matrices(self, image_data):
        row_ind = []
        col_ind = []
        resolution_factor = int(image_data.shape[1] / image_data.shape[0])

        for i in range(image_data.shape[0]):
            row_ind.append(i * resolution_factor)
            col_ind.append(i)

        forward_mat = scipy.sparse.csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)),
                                              shape=(image_data.shape[1], image_data.shape[0]))
        backward_mat = forward_mat.transpose(copy=False).tocsr(copy=True)

        self.sparse_mats = (forward_mat, backward_mat)

    def generate_gaussian_psf(self, x, psf_std, filename=None):
        pixel_size = x[1] - x[0]
        abs_lim = self.psf_res * pixel_size

        x = np.linspace(0, abs_lim, self.psf_res + 1, endpoint=True)
        x -= x[-1] / 2
        y = np.linspace(0, abs_lim, self.psf_res + 1, endpoint=True)
        y -= y[-1] / 2

        xv, yv = np.meshgrid(x, y)
        pos = np.empty(xv.shape + (2,))
        pos[:, :, 0] = xv
        pos[:, :, 1] = yv

        if psf_std > 0:
            mv = multivariate_normal([0, 0], psf_std ** 2)
            const = np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(np.diag([psf_std ** 2, psf_std ** 2])))
            self.gaussian_psf = const * mv.pdf(pos)
            self.gaussian_psf /= np.sum(self.gaussian_psf)
        else:
            self.gaussian_psf = np.zeros((x.shape[0], y.shape[0]))
            self.gaussian_psf[int(x.shape[0] / 2), int(y.shape[0] / 2)] = 1

        if filename:
            fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True, dpi=200)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            norm = matplotlib.colors.Normalize(vmin=np.min(self.gaussian_psf), vmax=np.max(self.gaussian_psf))
            cmap = 'bone'
            ax.imshow(self.gaussian_psf, extent=(x[0], x[-1], y[-1], y[0]), cmap=cmap, norm=norm)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical',
                         label=r'$I_{\mathrm{PSF}}(x, y)$ [a. u.]')

            ax.invert_yaxis()
            ax.set_xlabel(r'$x$ [mm]')
            ax.set_ylabel(r'$y$ [mm]')
            ax.set_aspect('equal')

            plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0, dpi=200)

            ax.set_title(filename[filename.rfind('/') + 1:])
            plt.show()


class PSFResponseOperator(ift.LinearOperator):
    def __init__(self, domain, target, resolution_factor, sparse_mats, gaussian_psf):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        self.resolution_factor = resolution_factor
        self.sparse_mats = sparse_mats
        self.gaussian_psf = gaussian_psf
        self.pad = int(gaussian_psf.shape[0] / 2)

        self.PSFop = pylops.signalprocessing.Convolve2D((*self.domain.shape,), self.gaussian_psf,
                                                        offset=(self.pad, self.pad))

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = self.times_func(x)
        else:
            res = self.adjoint_times_func(x)

        return ift.Field(self._tgt(mode), res)

    def times_func(self, x):
        signal = (self.PSFop * x.val)[self.pad:-self.pad, self.pad:-self.pad]
        res = self.sparse_mats[1] @ signal

        return res

    def adjoint_times_func(self, x):
        res = self.sparse_mats[0] @ x.val
        res = np.pad(res, self.pad)
        res = self.PSFop.H * res

        return res
