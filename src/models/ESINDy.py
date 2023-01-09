import torch
import torch.nn as nn
from src.models.HyperNet import HyperNet
from src.utils.model_utils import library_size, sindy_library


class Net(nn.Module):
    """An ESINDy model.

    An ESINDy model that is trained through gradient descent.

    Attributes:
        self.z_dim: The spatial dimension (int) of the data.
        self.poly_order: The order (int) of the polynomials in the data.
        self.include_constant: Iff True (bool), a constant term is included in
            the SINDy library.
        self.include_sine: Iff True (bool), sine is included in the SINDy
            library.
        self.num_ensemble: The number (int) of models in the ensemble.
        self.prior: A string denoting what type of prior should be imposed
            on the SINDy coefficents. Options: {"normal", "laplace"}. The
            option "normal" refers to l2 norm loss, while "laplace" refers to
            using a lasso (l1) norm.
        self.library_dim: The number (int) of terms in the SINDy library.
        self.sindy_coefficients: A torch.Tensor of shape
            (num_ensemble x library_dim x z_dim) that is multiplied by terms
            in the SINDy library.
        self.threshold_mask: A torch.Tensor of shape
            (num_ensemble x library_dim x z_dim) used to permanently zero out
            SINDy coefficients.
    """
    def __init__(self, args, hyperparams):
        """Initalizes the network.

        Initializes the ESINDy network using the given args and hyperparameters.

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

        self.num_ensemble = hyperparams.num_ensemble
        self.prior = hyperparams.prior

        self.library_dim = library_size(self.z_dim, self.poly_order,
            include_constant=self.include_constant, use_sine=self.include_sine)
        
        self.sindy_coefs = nn.Parameter(
            #torch.normal(0, 0.1, (self.num_ensemble, self.library_dim, self.z_dim)),
            torch.ones(self.num_ensemble, self.library_dim, self.z_dim),
            requires_grad=True)

        self.threshold_mask = nn.Parameter(
            torch.ones(self.num_ensemble, self.library_dim, self.z_dim),
            requires_grad=False)
    
    # TODO: make x_lib actually be optional. currently, None option only works
    # for shape (batch_size x z_dim), not (batch_size x num_ensemble x z_dim).
    def forward(self, x, x_lib=None, device=0):
        """Runs the forward pass.

        Runs the forward pass, calculating the derivatives.

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
                (batch_size x num_ensemble x library_dim).
            coefs: The SINDy coefficients as a torch.Tensor of shape
                (um_ensemble x library_dim x z_dim).
        """
        return torch.matmul(library.unsqueeze(2), coefs).squeeze(2)

    def get_masked_coefficients(self, n=None, batch_size=None, device=0):
        """Get the masked coefficients.

        Applies the coefficient threshold mask to the SINDy coefficients.

        Args:
            None

        Returns:
            The thresholded SINDy coefficients as a torch.Tensor.
        """
        return self.sindy_coefs * self.threshold_mask

    def update_threshold_mask(self, threshold, device):
        """Updates the threshold mask.

        Gets the indices of any SINDy coefficients that are less than the given
        threshold. Sets self.threshold_mask[indices] = 0.

        Args:
            threshold: The threshold (float) to use.
        
        Returns:
            None
        """
        self.threshold_mask[torch.abs(self.sindy_coefs) < threshold] = 0

    # TODO: make this function work for an ensemble. currently only works
    # for shape (batch x z_dim).
    def make_library(self, x):
        """Constructs a SINDy library.

        Creates a SINDy library out of the given tensor.

        Args:
            The torch.Tensor (batch_size x num_ensemble x z_dim) to construct
            a SINDy library with. Can also have shape (batch_size x z_dim).

        Returns:
            A torch.Tensor of shape (batch_size x num_ensemble x z_dim)
            or of shape (batch_size x z_dim), depending on input shape.
        """
        return sindy_library(x, self.poly_order,
                             include_constant=self.include_constant,
                             include_sine=self.include_sine)