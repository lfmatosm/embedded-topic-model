import torch
import torch.nn.functional as F
from torch import nn
from embedded_topic_model.core.layer import LinearSVD


class BaseModel(nn.Module):
    def __init__(
        self, num_topics: int, vocab_size: int, rho_size: int,
        train_embeddings: bool, embeddings = None
    ) -> None:
        super().__init__()
        
        # define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.rho_size = rho_size

        # define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            self.rho = embeddings.clone().float().to(self.device)
            
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            act = nn.Tanh()
            if self.debug_mode:
                print('Defaulting to tanh activation')
        return act
    
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu
        
    def get_beta(self, *args):
        pass
    
    def get_theta(self, *args):
        pass
    
    def encode(self, *args):
        pass
    
    def decode(self, *args):
        pass


class Etm(BaseModel):
    def __init__(
            self,
            vocab_size: int,
            num_topics: int = 50,
            t_hidden_size: int = 800,
            rho_size: int = 100,
            theta_act: str = 'relu',
            train_embeddings=True,
            embeddings=None,
            enc_drop=0.5,
            debug_mode=False
    ) -> None:
        super().__init__(
            num_topics=num_topics, vocab_size=vocab_size, rho_size=rho_size, 
            train_embeddings=train_embeddings, embeddings=embeddings
        )

        # define hyperparameters
        self.t_hidden_size = t_hidden_size
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.debug_mode = debug_mode
        self.theta_act = self.get_activation(theta_act)

        # define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            self.rho = embeddings.clone().float().to(self.device)

        # define the matrix containing the topic embeddings
        # nn.Parameter(torch.randn(rho_size, num_topics))
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)

        # define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, t_hidden_size),
            self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * \
            torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight)
        except BaseException:
            logit = self.alphas(self.rho)
        logit = logit.transpose(1, 0)  
        beta = F.softmax(logit, dim=0)          # softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def forward(self, bows, normalized_bows, theta=None):
        # get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        # get \beta
        beta = self.get_beta()

        # get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        recon_loss = recon_loss.mean()
        kld_loss = kld_theta
        return recon_loss, kld_loss


class ProdEtm(Etm):
    def __init__(
            self,
            vocab_size: int,
            num_topics: int = 50,
            t_hidden_size: int = 800,
            rho_size: int = 100,
            theta_act: str = 'relu',
            train_embeddings=True,
            embeddings=None,
            enc_drop=0.5,
            debug_mode=False
    ) -> None:
        super().__init__(
            vocab_size, num_topics, t_hidden_size, rho_size,
            theta_act, train_embeddings, embeddings, enc_drop, debug_mode
        )
        
    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight)
        except BaseException:
            logit = self.alphas(self.rho)
        beta = logit.transpose(1, 0)  
        return beta
    
    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        theta = self.reparameterize(mu_theta, logsigma_theta)
        return theta, kld_theta
    
    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        res = F.softmax(res, dim=-1)
        preds = torch.log(res + 1e-6)
        return preds
    
    
class DropProdEtm(Etm):
    def __init__(
            self,
            vocab_size: int,
            num_topics: int = 50,
            t_hidden_size: int = 800,
            rho_size: int = 100,
            theta_act: str = 'relu',
            train_embeddings=True,
            embeddings=None,
            enc_drop=0.5,
            debug_mode=False
    ) -> None:
        super().__init__(
            vocab_size, num_topics, t_hidden_size, rho_size,
            theta_act, train_embeddings, embeddings, enc_drop, debug_mode
        )
        self.alphas = LinearSVD(rho_size, num_topics, bias=False)
        
    def forward(self, bows, normalized_bows, theta=None):
        # get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        # get \beta
        beta = self.get_beta()

        # get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        recon_loss = recon_loss.mean()
        kld_loss = kld_theta + self.alphas.kl_loss
        return recon_loss, kld_loss
