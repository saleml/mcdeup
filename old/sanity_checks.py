import torch
from torch import nn
from networks import DropoutModel
from old.dropout import RegularDropout, RegularDropoutPerLayer, MultiplicativeGaussian, MultiplicativeGaussianPerLayer

torch.set_default_tensor_type(torch.DoubleTensor)

x = torch.randn(5, 1)
y = torch.randn(5, 1)


def parameter_names(model):
    return [x for x, y in model.named_parameters()]


def analyze(model, logit=True, logsigma=False, mu=False, noise_generator=False, x_input=False):
    x = torch.randn(5, 1)
    y = torch.randn(5, 1)
    print(parameter_names(model))
    optimizer = torch.optim.SGD([y for x, y in model.named_parameters() if 'dropout' in x and 'mu' not in x], lr=1e-1)
    if logsigma:
        print("Before update:", model.model.dropout1.logsigma.detach().numpy(),
              model.model.dropout2.logsigma.detach().numpy())
    if logit:
        print("Before update:", model.model.dropout1.logit.detach().numpy(),
              model.model.dropout2.logit.detach().numpy())
    if mu:
        print("Before update:", model.model.dropout1.mu.detach().numpy(),
              model.model.dropout2.mu.detach().numpy())
    if noise_generator:
        print("Before update:", sum(list(map(lambda x: torch.mean(x).detach().item(), model.model.dropout1.noise_generator.parameters()))),
              sum(list(map(lambda x: torch.mean(x).detach().item(), model.model.dropout2.noise_generator.parameters()))))
    optimizer.zero_grad()
    loss = nn.MSELoss()(model.train()(x, parametric_noise=noise_generator, noise_input=x if x_input else None), y)
    loss.backward()
    optimizer.step()
    if logsigma:
        print("After update:", model.model.dropout1.logsigma.detach().numpy(),
              model.model.dropout2.logsigma.detach().numpy())
    if logit:
        print("After update:", model.model.dropout1.logit.detach().numpy(),
              model.model.dropout2.logit.detach().numpy())
    if mu:
        print("After update:", model.model.dropout1.mu.detach().numpy(),
              model.model.dropout2.mu.detach().numpy())
    if noise_generator:
        print("After update:", sum(list(map(lambda x: torch.mean(x).detach().item(), model.model.dropout1.noise_generator.parameters()))),
              sum(list(map(lambda x: torch.mean(x).detach().item(), model.model.dropout2.noise_generator.parameters()))))
    print('\n\n')


def analyze_fixed_dropout(soft=False):
    print("Defining model with the same dropout value for all neurons")
    logit = nn.Parameter(torch.tensor(-2.))
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=RegularDropout, logit=logit, tau=1., soft=soft)
    analyze(model)

    print("Defining model with different value for all neurons but shared amongst layers")
    logit = nn.Parameter(torch.randn(3))
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=RegularDropout, logit=logit, tau=1., soft=soft)
    analyze(model)

    print("Defining model with the same dropout value for all neurons in the same layer")
    logit = [nn.Parameter(torch.tensor(-2.)), nn.Parameter(torch.tensor(-3.))]
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=RegularDropoutPerLayer, logit=logit, tau=1., soft=soft)
    analyze(model)

    print("Defining model with different dropout value for each neuron")
    logit = [nn.Parameter(torch.randn(3)), nn.Parameter(torch.randn(3))]
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=RegularDropoutPerLayer, logit=logit, tau=1., soft=soft)
    analyze(model)


def analyze_multiplicative_gaussian(use_mu=False):
    print("Defining model with the same dropout value for all neurons")
    logsigma = nn.Parameter(torch.tensor(-2.))
    mu = None
    if use_mu:
        mu = nn.Parameter(torch.tensor(2.))
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=MultiplicativeGaussian, mu=mu, logsigma=logsigma)
    analyze(model, logit=False, logsigma=True, mu=use_mu)

    print("Defining model with different value for all neurons but shared amongst layers")
    logsigma = nn.Parameter(torch.randn(3))
    mu = None
    if use_mu:
        mu = nn.Parameter(torch.randn(3))
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=MultiplicativeGaussian, mu=mu, logsigma=logsigma)
    analyze(model, logit=False, logsigma=True, mu=use_mu)

    print("Defining model with the same dropout value for all neurons in the same layer")
    logsigma = [nn.Parameter(torch.tensor(-2.)), nn.Parameter(torch.tensor(-3.))]
    mu = None
    if use_mu:
        mu = [nn.Parameter(torch.tensor(2.)), nn.Parameter(torch.tensor(1.))]
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=MultiplicativeGaussianPerLayer, mu=mu, logsigma=logsigma)
    analyze(model, logit=False, logsigma=True, mu=use_mu)

    print("Defining model with different dropout value for each neuron")
    mu = None
    if use_mu:
        mu = [nn.Parameter(torch.randn(3)), nn.Parameter(torch.randn(3))]
    logsigma = [nn.Parameter(torch.randn(3)), nn.Parameter(torch.randn(3))]
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=MultiplicativeGaussianPerLayer, mu=mu, logsigma=logsigma)
    analyze(model, logit=False, logsigma=True, mu=use_mu)


def analyze_learned_multiplicative_gaussian(x_input=False):
    print("Defining model with the same dropout value for all neurons")
    logsigma = nn.Parameter(torch.tensor(-2.))
    mu = None
    noise_generator = nn.Linear(3, 1) if not x_input else nn.Linear(1, 1)
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=MultiplicativeGaussian, mu=mu, logsigma=logsigma, noise_generator=noise_generator)
    analyze(model, logit=False, noise_generator=True, x_input=x_input)

    print("Defining model with different value for all neurons but shared amongst layers")
    logsigma = nn.Parameter(torch.randn(3))
    mu = None
    noise_generator = nn.Linear(3, 3) if not x_input else nn.Linear(1, 3)
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=MultiplicativeGaussian, mu=mu, logsigma=logsigma, noise_generator=noise_generator)
    analyze(model, logit=False, noise_generator=True, x_input=x_input)

    print("Defining model with the same dropout value for all neurons in the same layer")
    logsigma = [nn.Parameter(torch.tensor(-2.)), nn.Parameter(torch.tensor(-3.))]
    noise_generator = [nn.Linear(3, 1) if not x_input else nn.Linear(1, 1),
                       nn.Linear(3, 1) if not x_input else nn.Linear(1, 1)]
    mu = None
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=MultiplicativeGaussianPerLayer, mu=mu, logsigma=logsigma, noise_generator=noise_generator)
    analyze(model, logit=False, noise_generator=True, x_input=x_input)

    print("Defining model with different dropout value for each neuron")
    mu = None
    logsigma = [nn.Parameter(torch.randn(3)), nn.Parameter(torch.randn(3))]
    noise_generator = [nn.Linear(3, 3) if not x_input else nn.Linear(1, 3),
                       nn.Linear(3, 3) if not x_input else nn.Linear(1, 3)]
    model = DropoutModel(input_dim=1, n_hidden=3, hidden_layers=2, output_dim=1,
                         dropout_module=MultiplicativeGaussianPerLayer, mu=mu, logsigma=logsigma, noise_generator=noise_generator)
    analyze(model, logit=False, noise_generator=True, x_input=x_input)


analyze_fixed_dropout()
analyze_multiplicative_gaussian()
analyze_learned_multiplicative_gaussian()
