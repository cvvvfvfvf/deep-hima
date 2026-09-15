"""
Deep Partially Linear Cox Model with SCAD Regularization
"""

import torch
import torch.nn.functional as F
import numpy as np
from sksurv.metrics import concordance_index_censored
from .linear_fit import fit_scad_cox, standardize
from .nonlinear_fit import fit_nonlinear_confounding
from sksurv.linear_model import CoxnetSurvivalAnalysis
from .cox_loss import coxph_loss_scad_like


class Net_nonlinear(torch.nn.Module):
    """
    Feedforward neural network for nonlinear function approximation.
    """

    def __init__(self, n_feature, n_hidden1, n_hidden2,
                 dropout_rate1, dropout_rate2, n_output):
        super(Net_nonlinear, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1).double()
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2).double()
        self.dropout1 = torch.nn.Dropout(p=dropout_rate1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate2)
        self.out = torch.nn.Linear(n_hidden2, n_output).double()

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = F.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x


class CoxMediationModel:
    """
    Deep Partially Linear Cox model with SCAD regularization.
    """

    def __init__(self, x_dim, m_dim, z_dim,
                 neuron1, neuron2,
                 dropout1, dropout2,
                 gamma,
                 scad_LAMBDA=[0.07], scad_a=3.7,
                 learning_rate=0.01, weight_decay=0.01,
                 dnn_epoch=100, scad_dnn_epoch=10,
                 random_state=None):

        self.random_state = random_state
        self.x_dim = x_dim
        self.m_dim = m_dim
        self.z_dim = z_dim
        self.gamma = gamma
        self.linear_dim = x_dim + m_dim
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.learning_rate = learning_rate
        self.scad_a = scad_a
        self.scad_LAMBDA = scad_LAMBDA
        self.weight_decay = weight_decay
        self.dnn_epoch = dnn_epoch
        self.scad_dnn_epoch = scad_dnn_epoch
        self.selected_lambda = None
        self.best_model_id = None
        self.BETA_ORIG = None

    def get_params(self, deep=True):
        """Get model parameters (sklearn-style interface)."""
        return {
            "x_dim": self.x_dim,
            "m_dim": self.m_dim,
            "z_dim": self.z_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "scad_a": self.scad_a,
            "scad_lam": self.scad_LAMBDA,
            "dnn_epoch": self.dnn_epoch,
            "scad_dnn_epoch": self.scad_dnn_epoch,
            "neuron1": self.neuron1,
            "neuron2": self.neuron2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2
        }

    def set_params(self, **parameters):
        """Set model parameters (sklearn-style interface)."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, data_x, data_y):
        """
        Fit the Deep Partially Linear Cox model.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)


        self.sample_size = data_x.shape[0]

        # Sort by observed time (required for Cox partial likelihood)
        t_ord = np.argsort(data_y.iloc[:, 0])
        order_data_x = data_x.iloc[t_ord, :]
        order_data_y_np = data_y.iloc[t_ord, :].to_numpy()
        order_data_y = torch.tensor(
            data_y.iloc[t_ord, :].to_numpy(),
            dtype=torch.double
        )

        # Extract X, M, Z components
        if self.x_dim > 0:
            order_data_x_X = torch.tensor(
                order_data_x.iloc[:, :self.x_dim].to_numpy(),
                dtype=torch.double
            )
        else:
            order_data_x_X = torch.zeros((order_data_x.shape[0], 0), dtype=torch.double)

        order_data_x_M = torch.tensor(
            order_data_x.iloc[:, self.x_dim:self.x_dim + self.m_dim].to_numpy(),
            dtype=torch.double
        )

        order_data_x_Z = torch.tensor(
            order_data_x.iloc[:, self.x_dim + self.m_dim:].to_numpy(),
            dtype=torch.double
        )

        # Compute fixed X effect prediction
        if self.gamma is not None and self.x_dim > 0:
            gamma_tensor = torch.tensor(self.gamma, dtype=torch.double).view(-1, 1)
            fixed_x_predict = torch.matmul(order_data_x_X, gamma_tensor)
        else:
            fixed_x_predict = torch.zeros((order_data_x.shape[0], 1), dtype=torch.double)


        # Standardize M and Z (store scaling parameters for test set)
        std_order_linear_data_list = standardize(order_data_x_M)
        std_order_linear_data_M = std_order_linear_data_list['X']

        std_order_nonlinear_data_list = standardize(order_data_x_Z)
        std_order_nonlinear_data_Z = std_order_nonlinear_data_list['X']

        # Store training set scaling parameters
        # Test set MUST use these (not its own statistics) to avoid:
        # - Beta coefficients being incorrectly weighted
        # - DNN receiving inputs outside training distribution
        # - C-index degradation on test set
        self.center = std_order_linear_data_list['center']
        self.scale = std_order_linear_data_list['scale']
        self.center_Z = std_order_nonlinear_data_list['center']
        self.scale_Z = std_order_nonlinear_data_list['scale']

        # Initialize beta via Elastic Net (Lasso, alpha=1.0)
        X_initial_beta = np.concatenate((
            std_order_linear_data_M.cpu().numpy(),
            std_order_nonlinear_data_Z.cpu().numpy()
        ), axis=1)

        # Prepare outcome for sksurv format: (event, time)
        y_initial_beta = np.zeros_like(order_data_y.numpy())
        y_initial_beta[:, 0] = order_data_y.numpy()[:, 1]  # event
        y_initial_beta[:, 1] = order_data_y.numpy()[:, 0]  # time

        censor_initial_beta = np.core.records.fromarrays(
            y_initial_beta.transpose(),
            names='Status, Survival_in_days',
            formats='bool, f8'
        )

        # Lambda grid for initialization: exp(linspace(log(1), log(0.1), 10))
        alpha = list(np.exp(np.linspace(np.log(1), np.log(0.1), 10)))

        est_initial_beta = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alphas=alpha
        ).fit(X_initial_beta, censor_initial_beta)

        # Extract beta from smallest alpha (strongest regularization)
        beta0_np = np.expand_dims(
            est_initial_beta._get_coef(None)[0][:self.m_dim],
            axis=1
        )
        beta0 = torch.from_numpy(beta0_np).type(torch.double)


        # Initialize storage for all lambda values
        lambda_len = len(self.scad_LAMBDA)
        self.LOSS = np.zeros((lambda_len,), dtype=np.float64)
        self.BETA = np.zeros((self.m_dim, lambda_len), dtype=np.float64)
        self.MODEL = []
        lam_id = 0

        # Fit model for each lambda
        for l in self.scad_LAMBDA:
            loss = []
            beta = beta0

            # Initialize neural networks
            dnn = Net_nonlinear(
                self.z_dim, self.neuron1, self.neuron2,
                self.dropout1, self.dropout2, 1
            )
            best_dnn = Net_nonlinear(
                self.z_dim, self.neuron1, self.neuron2,
                self.dropout1, self.dropout2, 1
            )
            best_beta = torch.zeros_like(beta)

            # Alternating optimization
            for i in range(self.scad_dnn_epoch):

                # Step 1: Train DNN with fixed beta
                current_linear_predict = fixed_x_predict + torch.matmul(
                    std_order_linear_data_M, beta
                )

                fit_nonlinear_confounding(
                    std_order_nonlinear_data_Z,
                    current_linear_predict,
                    order_data_y,
                    dnn,
                    self.weight_decay,
                    self.learning_rate,
                    self.dnn_epoch
                )

                # Step 2: Update beta with fixed DNN
                dnn.eval()
                nonlinear_predict_Z = dnn.forward(std_order_nonlinear_data_Z).data
                offset = fixed_x_predict + nonlinear_predict_Z
                dnn.train()

                beta = fit_scad_cox(
                    std_order_linear_data_M,
                    order_data_y[:, 1],
                    beta,
                    offset,
                    l,
                    self.m_dim
                )


                # Compute total loss
                current_linear_predict = fixed_x_predict + torch.matmul(
                    std_order_linear_data_M, beta
                )
                total_predict = current_linear_predict + nonlinear_predict_Z
                current_loss = coxph_loss_scad_like(total_predict, order_data_y)
                loss.append(current_loss)

                if i == 0:
                    pass
                else:
                    loss_change = loss[-2] - current_loss if len(loss) > 1 else 0

                # Track best model or rollback if loss increases
                if loss[-1] == min(loss):
                    best_dnn.load_state_dict(dnn.state_dict())
                    best_beta = beta.clone()
                else:
                    # Rollback: restore both beta and DNN to best iteration
                    # This maintains consistency between (beta, g(Z)) pair
                    beta = best_beta.clone()
                    dnn.load_state_dict(best_dnn.state_dict())

            # Store results for this lambda
            self.LOSS[lam_id] = min(loss).item()
            self.BETA[:, lam_id] = best_beta.numpy().squeeze()
            self.MODEL.append(best_dnn)

            linear_effect = torch.matmul(std_order_linear_data_M, beta).mean().item()
            lam_id = lam_id + 1

        # Rescale coefficients to original M scale
        # Centering M only shifts eta by a constant, which cancels in Cox
        # likelihood, so only scale needs to be undone
        scale_np = self.scale.cpu().numpy().reshape(-1)
        degenerate = scale_np < 1e-6  # Near-constant mediators

        # Avoid dividing by tiny scale (would explode coefficients)
        safe_scale = np.where(degenerate, 1.0, scale_np)
        self.BETA_ORIG = self.BETA / safe_scale[:, None]
        self.BETA_ORIG[degenerate, :] = 0.0

        # Handle empty support set (no mediators selected)
        # This is correct for negative controls, not an error
        if scale_np.size:
            pass
        else:
            pass

        return self

    def _select_best_model(self):
        """
        Select best model via Extended EBIC.
        """
        s = (self.BETA != 0).sum(axis=0)  # Non-zero counts for each lambda
        gamma_ebic = 0.5  # Tune between 0 (BIC) and 1 (strict eBIC)

        output_ebic = (
            2 * self.LOSS +
            np.log(self.sample_size) * s +
            2 * gamma_ebic * s * np.log(self.m_dim)
        )

        best_idx = np.argmin(output_ebic)
        return best_idx, self.scad_LAMBDA[best_idx]

    def predict(self, data_x):
        """
        Predict risk scores for new data.
        """
        best_model_id, best_lambda = self._select_best_model()

        # Extract X, M, Z components
        data_x_X = torch.tensor(
            data_x.iloc[:, :self.x_dim].to_numpy(),
            dtype=torch.double
        )
        data_x_M = torch.tensor(
            data_x.iloc[:, self.x_dim:self.x_dim + self.m_dim].to_numpy(),
            dtype=torch.double
        )
        data_x_Z = torch.tensor(
            data_x.iloc[:, self.x_dim + self.m_dim:].to_numpy(),
            dtype=torch.double
        )

        # Standardize using training set statistics
        std_data_M = (data_x_M - self.center) / self.scale
        std_data_Z = (data_x_Z - self.center_Z) / self.scale_Z

        # Compute fixed X effect
        gamma_tensor = torch.tensor(self.gamma, dtype=torch.double).view(-1, 1)
        fixed_x_predict = torch.matmul(data_x_X, gamma_tensor)

        # Compute linear mediator effect
        best_beta = torch.tensor(
            self.BETA[:self.m_dim, [best_model_id]],
            dtype=torch.double
        )
        linear_predict = torch.matmul(std_data_M, best_beta)

        # Compute nonlinear confounding effect
        best_dnn = self.MODEL[best_model_id]
        best_dnn.eval()
        nonlinear_predict = best_dnn.forward(std_data_Z).data
        best_dnn.train()

        # Total risk score
        risk_scores = fixed_x_predict + linear_predict + nonlinear_predict

        return risk_scores.data.numpy().squeeze()

    def score(self, data_x, y):
        """
        Compute concordance index (C-index) on test data.
        """
        y_pred = self.predict(data_x)
        y_event = y.iloc[:, 1].values.astype(bool)
        y_time = y.iloc[:, 0].values.astype('float64')
        c = concordance_index_censored(y_event, y_time, y_pred)[0]
        return c

    def estimate_nonlinear(self, data_x):
        """
        Estimate nonlinear confounding function g(Z) for new data.
        """
        best_model_id, _ = self._select_best_model()

        data_x_Z = torch.tensor(
            data_x.iloc[:, self.x_dim + self.m_dim:].to_numpy(),
            dtype=torch.double
        )
        std_data_Z = (data_x_Z - self.center_Z) / self.scale_Z

        best_dnn = self.MODEL[best_model_id]
        best_dnn.eval()
        nonlinear_predict = best_dnn.forward(std_data_Z).data
        best_dnn.train()

        return nonlinear_predict.data.numpy()

    def estimate_linear(self, data_x):
        """
        Estimate linear mediator effect beta'M for new data.
        """
        best_model_id, _ = self._select_best_model()

        data_x_M = torch.tensor(
            data_x.iloc[:, self.x_dim:self.x_dim + self.m_dim].to_numpy(),
            dtype=torch.double
        )
        std_data_M = (data_x_M - self.center) / self.scale

        best_beta = torch.tensor(
            self.BETA[:self.m_dim, [best_model_id]],
            dtype=torch.double
        )
        linear_predict = torch.matmul(std_data_M, best_beta)

        return linear_predict.data.numpy()
