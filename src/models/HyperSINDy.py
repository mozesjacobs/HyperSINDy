import torch
import torch.nn as nn
from src.models.HyperNet import HyperNet
from src.utils.model_utils import library_size, sindy_library


class Net(nn.Module):
    """A HyperSINDy model.

    The HyperSINDy model that uses a hypernetwork to output SINDy coefficients.

    Attributes:
        self.z_dim: The spatial dimension (int) of the data.
        self.poly_order: The order (int) of the polynomials in the data.
        self.include_constant: Iff True (bool), a constant term is included in
            the SINDy library.
        self.include_sine: Iff True (bool), sine is included in the SINDy
            library.
        self.statistic_batch_size: An integer indicating the default number of
            samples to draw when sampling coefficients if not specified.
        self.prior_type: A string denoting what type of prior should be imposed
            on the SINDy coefficents. Options: {"normal", "laplace"}. The
            option "normal" refers to N(0,1) KLD regularization, while
            "laplace" refers to using a laplace distribution with loc = 0 and
            scale = 1 instead.
        self.hypernet_hidden_dim: An int of the size of the Linear layers in
            the hypernetwork.
        self.noise_dim: An int of the size of the random noise input to the
            hypernetwork.
        self.library_dim: The number (int) of terms in the SINDy library.
        self.hypernet: A HyperNet (from HyperNet.py) that generates
            coefficients using random noise.
        self.soft_threshold: Sampled coefficients whose absolute value is less
            than this (float) value are set to 0.
        self.hard_threshold_mask: A torch.Tensor of shape (library_dim x z_dim) 
            used to permanently zero out SINDy coefficients.
        self.prior: If self.prior_type == "laplace", a
            torch.distribution.Laplace object with loc = 0 and scale = 1 to
            generate samples with. If self.prior_type != "laplace",
            self.prior = None, since no generator object is required. Instead,
            N(0, 1) noise is used.
    """

    def __init__(self, args, hyperparams):
        """Initalizes the network.

        Initializes the HyperSINDy network using the given args and hyperparameters.

        Args:
            args: The argparser object return by parse_args() in the file
                cmd_line.py.
            hyperparams: The argparser object returned by parse_hyperparams() in 
                the file cmd_line.py

        Returns:
            A Net().
        """
        super(Net, self).__init__()
        
        self.z_dim = args.z_dim
        self.poly_order = args.poly_order
        self.include_constant = args.include_constant
        self.include_sine = args.include_sine
        self.statistic_batch_size = args.statistic_batch_size
        self.prior_type = hyperparams.prior
        self.hypernet_hidden_dim = hyperparams.hidden_dim
        self.noise_dim = hyperparams.noise_dim

        self.library_dim = library_size(self.z_dim, self.poly_order,
            include_constant=self.include_constant, use_sine=self.include_sine)
        self.hypernet = HyperNet(self.noise_dim, (self.library_dim, self.z_dim),
            [self.hypernet_hidden_dim for _ in range(4)])

        self.soft_threshold = hyperparams.soft_threshold
        self.hard_threshold_mask = nn.Parameter(torch.ones(self.library_dim, self.z_dim),
            requires_grad=False)

        if self.prior_type == "laplace":
            self.prior = torch.distributions.Laplace(
                torch.zeros([self.library_dim * self.z_dim]),
                torch.ones([self.library_dim * self.z_dim])
            )
        else:
            self.prior = None
        
        
    def forward(self, x, x_lib=None, device=0):
        """Runs the forward pass.

        Runs the forward pass, calculating the derivatives using randomly
        sampled coefficients from the hypernetwork.

        Args:
            x: The data (torch.Tensor of shape (batch_size x z_dim)) to
                calculate derivatives with.
            x_lib: The sindy_library form of x. Default is None. If None,
                creates a SINDy library out of x.
            device: The cpu or gpu device to do calculations with. To use cpu,
                device must be "cpu". To use gpu, specify which gpu to use as
                an integer (i.e.: 0 or 1 or 2 or 3). 
        
        Returns:
            A tuple of (tensor_a and tensor_b), where tensor_a is the
            calculated derivative as a torch.Tensor, and tensor_b are the
            SINDy coefficients (as a torch.Tensor) used to do the calculation.
            The shape of tensor_a is: (batch_size x z_dim).
            The shape of tensor_b is: (batch_size x library_dim x z_dim).
        """
        x = x.type(torch.FloatTensor).to(device)
        if x_lib is None:
            x_lib = self.make_library(x)
        else:
            x_lib = x_lib.type(torch.FloatTensor).to(device)

        coeffs = self.get_masked_coefficients(batch_size=x.size(0), device=device)
        return self.dx(x_lib, coeffs), coeffs
    
    def dx(self, library, coefs):
        """Calculate the derivative.

        Given the library terms and the SINDy coefficients, calculate the
        derivative.

        Args:
            library: The SINDy library terms as a torch.Tensor of shape
                (batch_size x library_dim).
            coefs: The SINDy coefficients as a torch.Tensor of shape
                (batch_size x library_dim x z_dim).

        Returns:
            The calculated derivatives as a torch.Tensor of shape
            (batch_szie x z_dim).
        """
        return torch.bmm(library.unsqueeze(1), coefs).squeeze(1)

    def sample_coeffs(self, n=None, batch_size=None, device=0):
        """Samples coefficients.

        Samples coefficients from the hypernetwork.

        Args:
            n: The torch.Tensor to feed into the hypernetwork. Should be random
                N(0, 1) noise. If None, samples from a N(0, 1) distribution.
            batch_size: If n is None, samples a vector of shape 
                (batch_size x self.noise_dim) from a N(0, 1) distribution.
            device: If n is None, the sampled noise vector uses this 
                cpu or gpu device. To use cpu, device must be "cpu". To use
                gpu, specify which gpu to use as an integer (i.e.: 0 or 1 or 2
                or 3). 

        Returns:
            Sampled SINDy coefficients as a torch.Tensor of shape
            (n.size(0) x self.noise_dim), or if n is None, of shape
            (batch_size x self.noise_dim).
        """
        if batch_size is None:
            batch_size = self.statistic_batch_size
        if n is None:
            n = torch.randn([batch_size, self.noise_dim], device=device)
        return self.hypernet(n)

    def get_masked_coefficients(self, n=None, batch_size=None, device=0):
        """Samples thresholded coefficients (masked coefficients).

        Samples coefficients from the hypernetwork and thresholds them with
        the soft threshold and the hard threshold mask.

        Args:
            n: The torch.Tensor to feed into the hypernetwork. Should be random
                N(0, 1) noise. If None, samples from a N(0, 1) distribution.
            batch_size: If n is None, samples a vector of shape 
                (batch_size x self.noise_dim) from a N(0, 1) distribution.
            device: If n is None, the sampled noise vector uses this 
                cpu or gpu device. To use cpu, device must be "cpu". To use
                gpu, specify which gpu to use as an integer (i.e.: 0 or 1 or 2
                or 3). 

        Returns:
            Sampled SINDy coefficients as a torch.Tensor of shape
            (n.size(0) x self.noise_dim), or if n is None, of shape
            (batch_size x self.noise_dim).
        """
        coefs = self.sample_coeffs(n, batch_size, device)
        soft_mask = torch.abs(coefs) > self.soft_threshold
        return coefs * soft_mask * self.hard_threshold_mask

    def update_threshold_mask(self, threshold, device):
        """Updates the threshold mask.

        Samples coefficients using self.get_masked_coefficients. Gets the
        indices of any SINDy coefficients that are less than the given
        threshold. Sets self.threshold_mask[indices] = 0.

        Args:
            threshold: The threshold (float) to use.
            device: The cpu or gpu device to sample coefficients with. To use
                cpu, device must be "cpu". To use gpu, specify which gpu to
                use as an integer (i.e.: 0 or 1 or 2 or 3). 
        
        Returns:
            None
        """
        if threshold is not None:
            coefs = torch.mean(self.get_masked_coefficients(device=device), dim=0)
            self.hard_threshold_mask[torch.abs(coefs) < threshold] = 0
    
    # KL function taken from:
    # https://github.com/pawni/BayesByHypernet_Pytorch/blob/master/model.py
    def kl(self, num_samples, device):
        """Calculates the KL divergence of the coefficients.

        Samples SINDy coefficients and then calculates the KL divergence
        between those coefficients and the networks prior.

        Args:
            num_samples: The (int) batch size of the coefficients to sample.
            device: The cpu or gpu device to sample coefficients with. To use
                cpu, device must be "cpu". To use gpu, specify which gpu to
                use as an integer (i.e.: 0 or 1 or 2 or 3).

        Returns:
            The KL divergence (as a torch.FloatTensor of shape (,)), with
            a mean over the batch dimension and sum over the other dimensions.
        """
        #num_samples, device = sindy_coeffs.size(0), sindy_coeffs.device
        #masked_coeffs = sindy_coeffs.reshape(num_samples, -1) # 250 x 60
        #gen_weights = masked_coeffs.transpose(1, 0) # 60 x 250
        coefs = self.sample_coeffs(batch_size=num_samples, device=device)
        gen_weights = coefs.reshape(num_samples, -1).transpose(1, 0)

        if self.prior_type == "laplace":
            prior_samples = self.prior.rsample(torch.Size([num_samples])).T.to(device)
        elif self.prior_type == "normal":
            prior_samples = torch.randn_like(gen_weights)
        else:
            print("ERROR: args.prior should be laplace or normal, not " + self.prior_type)
            exit()
        
        eye = torch.eye(num_samples, device=device) # 250 x 250
        wp_distances = (prior_samples.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2  # 60 x 250 x 250
        ww_distances = (gen_weights.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2    # 60 x 250 x 250

        # zero out indices that were thresholded so kl isn't calculated for them
        #wp_distances = wp_distances * self.hard_threshold_mask.reshape(-1, 1, 1)
        #ww_distances = ww_distances * self.hard_threshold_mask.reshape(-1, 1, 1)
        
        wp_distances = torch.sqrt(torch.sum(wp_distances, 0) + 1e-8) # 250 x 250
        wp_dist = torch.min(wp_distances, 0)[0] # 250
        ww_distances = torch.sqrt(torch.sum(ww_distances, 0) + 1e-8) + eye * 1e10 # 250 x 250
        ww_dist = torch.min(ww_distances, 0)[0] # 250

        # mean over samples
        kl = torch.mean(torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8))
        kl *= gen_weights.shape[0]
        kl += torch.log(torch.tensor(float(num_samples) / (num_samples - 1)))
        return kl

    def make_library(self, x):
        """Constructs a SINDy library.

        Creates a SINDy library out of the given tensor.

        Args:
            The torch.Tensor (batch_size x z_dim) to construct
            a SINDy library with.

        Returns:
            A torch.Tensor of shape (batch_size x z_dim).
        """
        return sindy_library(x, self.poly_order,
                             include_constant=self.include_constant,
                             include_sine=self.include_sine)