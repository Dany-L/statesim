clear all, close all
% analysis of trained hybrid model with doubly hyperdomininant multiplier
% compare l2 gain to static diagonal multiplier

cal_block_matrix_file_name = './data/HybridCRnn-10-8000000-False-MOSEK.mat';
config_file_name = './data/config-HybridCRnn-10-8000000-False-MOSEK.json';
norm_file_name = './data/HybridCRnn-10-8000000-False-MOSEK.json';

% normalization of rnn
% cal_block_matrix_file_name = './data/HybridCRnn-64-3000000-True-MOSEK.mat';
% config_file_name = './data/config-HybridCRnn-64-3000000-True-MOSEK.json';
% norm_file_name = './data/HybridCRnn-64-3000000-True-MOSEK.json';


% load trained matrices
sys_matrices = load(cal_block_matrix_file_name);
par = sys_matrices.predictor_parameter;

% load normalization data
norm_data = jsondecode(fileread(norm_file_name));

% load configuration
config = jsondecode(fileread(config_file_name));
% load linearized system
A_lin = config.parameters.A_lin; B_lin=config.parameters.B_lin; C_lin = config.parameters.C_lin;

% size of caligraphic matices for P_tilde
nx = size(config.parameters.A_lin, 1); ny = nx; nu=nx;
nwp = size(config.parameters.B_lin, 2);
nzp = size(config.parameters.C_lin, 1);
nwu = config.parameters.nwu; nzu = nwu;

% correct input size for normalization
% nwp_e = nwp + nx + nwp;
nwp_e = nwp;

%% stability analysis
% KLMN parameter
omega_tilde = double([par.K, par.L1, par.L2, par.L3;
    par.M1, par.N11, par.N12, par.N13;
    par.M2, par.N21, par.N22, zeros(nwu,nwu)]);

% coupling matrices flat
L_x_flat = double(par.L_x_flat);
L_y_flat = double(par.L_y_flat);

% multiplier
Lambda = diag(double(par.lam));

%%

% coupling matrices
L_x = utils.lower_triangular_from_vector(L_x_flat, nx);
X = L_x * L_x';
L_y = utils.lower_triangular_from_vector(L_y_flat, nx);
Y = L_y * L_y';

U = eye(nx);
V = eye(nx) - Y * X;

% convex matrices for lmi
P_21_1 = [A_lin * Y, A_lin, B_lin, zeros(nx,nwu);
    zeros(nx,nx), X * A_lin, X * B_lin, zeros(nx,nwu);
    C_lin * Y, C_lin, zeros(nzp,nwp), zeros(nzp,nwu);
    zeros(nzu, nx+nx+nwp+nwu)];
P_21_2 = [zeros(nx,nx), A_lin, zeros(nx,nzu);
    eye(nx), zeros(nx,nx), zeros(nx,nzu);
    zeros(nzp,nx), C_lin, zeros(nzp,nzu);
    zeros(nzu,nx), zeros(nzu,nx), eye(nzu)];
P_21_4 = [zeros(nx,nx), eye(nx), zeros(nx,nwp), zeros(nx,nwu);
    zeros(nwp,nx), zeros(nwp,nx), eye(nwp), zeros(nwp,nwu);
    eye(ny), zeros(ny,nx), zeros(ny,nwp), zeros(ny,nwu);
    zeros(nzu,nx), zeros(nzu,nx), zeros(nzu,nwp), eye(nwu)];
P_21 = P_21_1 + P_21_2 * omega_tilde * P_21_4;

% ga = 8;
ga = sdpvar(1,1);

X_cap = [Y, eye(nx);
    eye(nx), X];

P_11 = -blkdiag(X_cap, ga * eye(nwp),Lambda);
P_22 = -blkdiag(X_cap,eye(nwp), Lambda);
P = [P_11, P_21';
    P_21, P_22];

% min_ew_of_P = min(real(eig(-P)))

%% check if KLMN constraints are even satisfied
eps = 1e-5;
lmi = [];
lmi = lmi + (P <= -eps * eye(size(P,1)));
lmi = lmi + (ga >= 0);
optimize(lmi, ga, sdpsettings('solver','mosek', 'verbose', 0))
% checkset(lmi)
sqrt(double(ga))

%% nonlinearity
alpha=0;beta=1;
relu = @(z) max(0, z);
Delta_tilde =@(z) (2/(beta-alpha))*(relu(z)-(alpha+beta)/2*z);

%% verfiy linear bounds
z = -5:0.1:5;
% figure()
% plot(z, Delta_tilde(z), 'LineWidth',4); grid on, hold on
% plot(z,z, '--', 'LineWidth',1.5), plot(z,-z, '--', 'LineWidth',1.5) 
% legend('$\tilde{\Delta}(z)$','$z$', '$-z$', 'fontsize', 16, 'interpreter', 'latex')
% xlabel('$z$','interpreter', 'latex', 'FontSize',16)


%% convert KLMN to original parameters
T_l = [U, X * A_lin, zeros(nx,nwu);
    zeros(nu,nx), eye(nu), zeros(nu,nwu);
    zeros(nzu,nx),zeros(nzu,nu), Lambda];
T_r = [zeros(nx,nx), zeros(nx,nwp), V',zeros(nx,nwu);
    zeros(nwp,nx), eye(nwp), zeros(nwp,nx), zeros(nwp,nwu);
    eye(nx), zeros(nx,nwp), Y,zeros(nu,nwu);
    zeros(nzu,nx), zeros(nzu,nwp), zeros(nzu,ny), eye(nwu)];
T_s = [zeros(nx,nx+nwp), X*A_lin*Y, zeros(nx,nwu);
    zeros(nu,nx+nwp+ny+nwu);
    zeros(nzu,nx+nwp+ny+nwu)];

% original parameters
omega = T_l^(-1) * (omega_tilde - T_s) * (T_r)^(-1);
% omega = T_l \ (omega_tilde - T_s) / T_r;

A_tilde = omega(1:nx,1:nx);
B1_tilde = omega(1:nx, nx+1:nx+nwp);
B2_tilde = omega(1:nx, nx+nwp+1:nx+nwp+nu);
B3_tilde = omega(1:nx, nx+nwp+nu+1:end);

C1_tilde = omega(nx+1:nx+ny,1:nx);
D11_tilde = omega(nx+1:nx+ny, nx+1:nx+nwp);
D12_tilde = omega(nx+1:nx+ny, nx+nwp+1:nx+nwp+nu);
D13_tilde = omega(nx+1:nx+ny, nx+nwp+nu+1:end);

C2 = omega(nx+ny+1:end,1:nx);
D21 = omega(nx+ny+1:end, nx+1:nx+nwp);
D22 = omega(nx+ny+1:end, nx+nwp+1:nx+nwp+nu);

B1_hat = B1_tilde;
B2_hat = B2_tilde;
C1_hat = C1_tilde;
D11_hat = D11_tilde;
D12_hat = D12_tilde;
D13_hat = D13_tilde;
D21_hat = D21;
D22_hat = D22;

S_s = [A_lin, zeros(nx,nx), B_lin, zeros(nx,nwu);
    zeros(nx,nx+nx+nwp_e+nwu);
    C_lin, zeros(nzp,nx), zeros(nzp,nwp_e), zeros(nzp,nwu);
    zeros(nzu, nx+nx+nwp_e+nwu)];
S_l = [zeros(nx,nx), A_lin, zeros(nx,nzu);
    eye(nx), zeros(nx,nx), zeros(nx, nzu);
    zeros(nzp,nx), C_lin, zeros(nzp,nzu);
    zeros(nzu,nx), zeros(nzu,nx), eye(nzu)];
S_r = [zeros(nx,nx), eye(nx), zeros(nx,nwp_e), zeros(nx, nwu);
    zeros(nwp_e,nx), zeros(nwp_e,nx), eye(nwp_e), zeros(nwp_e, nwu);
    eye(nx), zeros(nx,nx), zeros(nx,nwp_e), zeros(nx,nwu);
    zeros(nwu, nx), zeros(nwu, nx), zeros(nwu,nwp_e), eye(nwu)];

omega_hat = [A_tilde, B1_hat, B2_hat, B3_tilde;
    C1_hat, D11_hat, D12_hat, D13_hat;
    C2, D21_hat, D22_hat, zeros(nwu,nwu)];


% B1_hat = B1_tilde * diag(1./norm_data.control_std);
% B2_hat = B2_tilde * diag(1./norm_data.state_std);
% C1_hat =  diag(norm_data.state_std) * C1_tilde;
% D11_hat =  diag(norm_data.state_std) * D11_tilde * diag(1./norm_data.control_std);
% D12_hat =  diag(norm_data.state_std) * D12_tilde * diag(1./norm_data.state_std);
% D13_hat = diag(norm_data.state_std) * D13_tilde;
% D21_hat = D21 * diag(1./norm_data.control_std);
% D22_hat = D22 * diag(1./norm_data.state_std);
% 
% S_s = [A_lin, zeros(nx,nx), [B_lin,zeros(nx,nx+nwp)], zeros(nx,nwu);
%     zeros(nx,nx+nx+nwp_e+nwu);
%     C_lin, zeros(nzp,nx), zeros(nzp,nwp_e), zeros(nzp,nwu);
%     zeros(nzu, nx+nx+nwp_e+nwu)];
% S_l = [zeros(nx,nx), A_lin, zeros(nx,nzu);
%     eye(nx), zeros(nx,nx), zeros(nx, nzu);
%     zeros(nzp,nx), C_lin, zeros(nzp,nzu);
%     zeros(nzu,nx), zeros(nzu,nx), eye(nzu)];
% S_r = [zeros(nx,nx), eye(nx), zeros(nx,nwp_e), zeros(nx, nwu);
%     zeros(nwp_e,nx), zeros(nwp_e,nx), eye(nwp_e), zeros(nwp_e, nwu);
%     eye(nx), zeros(nx,nx), zeros(nx,nwp_e), zeros(nx,nwu);
%     zeros(nwu, nx), zeros(nwu, nx), zeros(nwu,nwp_e), eye(nwu)];
% 
% omega_hat = [A_tilde, [B1_hat, -B1_hat, -B2_hat], B2_hat, B3_tilde;
%     C1_hat, [D11_hat,-D11_hat,eye(nx)-D12_hat], D12_hat, D13_hat;
%     C2, [D21_hat, -D21_hat, -D22_hat], D22_hat, zeros(nwu,nwu)];


P_tilde = S_s + S_l * omega_hat * S_r;
P_bold = S_s + S_l * sys_matrices.omega * S_r;

% extract caligraphic matrices for lure system
A_cal = P_tilde(1:nx*2, 1:nx*2);
B1_cal = P_tilde(1:nx*2, nx*2+1:nx*2+nwp_e);
B2_cal = P_tilde(1:nx*2, nx*2+1+nwp_e:end);

C1_cal = P_tilde(nx*2+1: nx*2+nzp, 1:nx*2);
D11_cal = P_tilde(nx*2+1:nx*2+nzp, nx*2+1:nx*2+nwp_e);
D12_cal = P_tilde(nx*2+1:nx*2+nzp, nx*2+1+nwp_e:end);

C2_cal = P_tilde(nx*2+nzp+1:end, 1:nx*2);
D21_cal = P_tilde(nx*2+nzp+1:end, nx*2+1:nx*2+nwp_e);
D22_cal = P_tilde(nx*2+nzp+1:end, nx*2+1+nwp_e:end);

A_cal = P_bold(1:nx*2, 1:nx*2);
B1_cal = P_bold(1:nx*2, nx*2+1:nx*2+nwp_e);
B2_cal = P_bold(1:nx*2, nx*2+1+nwp_e:end);

C1_cal = P_bold(nx*2+1: nx*2+nzp, 1:nx*2);
D11_cal = P_bold(nx*2+1:nx*2+nzp, nx*2+1:nx*2+nwp_e);
D12_cal = P_bold(nx*2+1:nx*2+nzp, nx*2+1+nwp_e:end);

C2_cal = P_bold(nx*2+nzp+1:end, 1:nx*2);
D21_cal = P_bold(nx*2+nzp+1:end, nx*2+1:nx*2+nwp_e);
D22_cal = P_bold(nx*2+nzp+1:end, nx*2+1+nwp_e:end);

%% evaluate condition on nonlinearity
a=0;b=1;
n = 2;
P_r = [b*eye(n),-1*eye(n);-a*eye(n),1*eye(n)];
M = [1,-0.5;-0.5,1];
P = P_r'*[zeros(n,n), M; M,zeros(n,n)]*P_r;

p = zeros(length(z),1);
for idx = 1:length(z)
    p(idx) = [z(idx), z(idx), relu(z(idx)), relu(z(idx))] * P * [z(idx);z(idx);relu(z(idx));relu(z(idx))];
end
figure()
plot(z,p,'LineWidth',3), legend('$(\star)^T P $col$(w^k,z^k)$','interpreter','latex','fontsize',18)
xlabel('$k$', 'interpreter','latex','fontsize',18), grid on
%% stability analysis of Lure system
a=-1;b=1;
P_r = [b*eye(nwu) -eye(nwu);-a*eye(nwu) eye(nwu)];
lambda = sdpvar(nwu,1);

L = diag(lambda);

multiplier_constraint = [];
for idx = 1:1:nwu-1
    L = L - diag(sdpvar(nwu-idx,1), idx);
    L = L - diag(sdpvar(nwu-idx,1), -idx);
    multiplier_constraint = multiplier_constraint + (diag(L,idx) <= 0);
    multiplier_constraint = multiplier_constraint +(diag(L,-idx) <= 0);
end


P = P_r' * [zeros(nwu,nwu), L; L, zeros(nwu,nwu)] * P_r;

%%
L1 = [A_cal, B1_cal, B2_cal;
    eye(2*nx), zeros(2*nx, nwp_e), zeros(2*nx, nwu)];
L2 = [C1_cal, D11_cal, D12_cal;
    zeros(nwp_e, nx*2), eye(nwp_e), zeros(nwp_e, nwu)];
L3 = [C2_cal, D21_cal, D22_cal;
    zeros(nzu, nx*2), zeros(nzu, nwp_e), eye(nwu)];

% X_cal = [Y, V;eye(nx), zeros(nx,nx)]^(-1)* [eye(nx), zeros(nx,nx);X,U];
X_cal = sdpvar(2*nx,2*nx);
ga = sdpvar(1,1);
% ga = config.parameters.gamma;

eps=1e-5;

lmis = [];
lmi = L1' * [X_cal, zeros(2*nx,2*nx); zeros(2*nx,2*nx), -X_cal] * L1 + ...
    L2' * [eye(nwp_e), zeros(nwp_e,nzp); zeros(nzp,nwp_e), -ga*eye(nzp)] * L2 + ...
    L3' * P * L3;
max(real(eig(lmi)))
lmis = lmis + (lmi <= -eps * eye(size(L1,2)));
lmis = lmis + (X_cal >= 0);
% lmis = lmis + (Lambda >= 0);
lmis = lmis + (ones(1,nwu)* L >= eps);
lmis = lmis + (L*ones(nwu,1) >= eps);
lmis = lmis + multiplier_constraint;

optimize(lmis, ga, sdpsettings('solver','MOSEK','verbose', 0))
checkset(lmis)
sqrt(double(ga))









