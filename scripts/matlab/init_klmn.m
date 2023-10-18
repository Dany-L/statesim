clear all, close all

% - init klmn parameters such that hybrid model behaves as linear model
% e.g. no controller
% - project parameters on convex set
% - compare initial parameter simulation with projected one

cal_block_matrix_file_name = './data/HybridCRnn-10-8000000-no-barrier.mat';
config_file_name = './data/config-HybridCRnn-10-8000000-2.json';
data_config_file_name = './data/coupled_msd_initial_state-0_u-15_K-100_T-1200_config.json';

% load trained matrices
sys_matrices = load(cal_block_matrix_file_name);
par = sys_matrices.predictor_parameter;

% load configuration
config = jsondecode(fileread(config_file_name));
% load data configuration
data_config = jsondecode(fileread(data_config_file_name));

% load linearized system
A_lin = config.parameters.A_lin; B_lin=config.parameters.B_lin; C_lin = config.parameters.C_lin;

% size of caligraphic matices
nx = size(config.parameters.A_lin, 1); ny = nx; nu=nx;
nwp = size(config.parameters.B_lin, 2);
nzp = size(config.parameters.C_lin, 1);
nwu = config.parameters.nwu; nzu = nwu;

%% transformation from controller matrices to interconnected system
S_s = [A_lin, zeros(nx,nx), B_lin, zeros(nx,nwu);
    zeros(nx,nx+nx+nwp+nwu);
    C_lin, zeros(nzp,nx), zeros(nzp,nwp), zeros(nzp,nwu);
    zeros(nzu, nx+nx+nwp+nwu)];
S_l = [zeros(nx,nx), A_lin, zeros(nx,nzu);
    eye(nx), zeros(nx,nx), zeros(nx, nzu);
    zeros(nzp,nx), C_lin, zeros(nzp,nzu);
    zeros(nzu,nx), zeros(nzu,nx), eye(nzu)];
S_r = [zeros(nx,nx), eye(nx), zeros(nx,nwp), zeros(nx, nwu);
    zeros(nwp,nx), zeros(nwp,nx), eye(nwp), zeros(nwp, nwu);
    eye(nx), zeros(nx,nx), zeros(nx,nwp), zeros(nx,nwu);
    zeros(nwu, nx), zeros(nwu, nx), zeros(nwu,nwp), eye(nwu)];

P_bold = S_s + S_l * sys_matrices.omega * S_r;
omega_shape = size(sys_matrices.omega);

%% simulation
T = 1000;
t = 0:config.parameters.time_delta:T';
N = length(t);

input_config = data_config.input_generator;
u_max = input_config.u_max; u_min = input_config.u_min;
% u_max = 0.1; u_min = -u_max;
T_min = input_config.interval_min; T_max = input_config.interval_max;

% input signal
u = zeros(N,1);
k_start = 1;k_end = 1;
while k_end < N
    k_end = k_start + randi([T_min, T_max],1);
    amplitude = u_min + (u_max - u_min).*rand(1,1);
    u(k_start:k_end)=amplitude;
    k_start=k_end;
end
u = u(1:N);
wp_ts = timeseries(u,t);

%% linearized system
linearized_sys = dss(A_lin, B_lin, C_lin, 0,eye(nx),config.parameters.time_delta);
zp_lin = lsim(linearized_sys,u,t);

%% find initial parameters
% ga = config.parameters.gamma;
fprintf('H_inf of linear approximation: %f\n',hinfnorm(linearized_sys));
% ga = 1000;
eps = 2;
ga = 100;

klmn_0 = zeros(nx+nu+nzu, nx+nwp+ny+nwu);

Y_init = sdpvar(nx,nx,'diag');
X_init = sdpvar(nx,nx,'diag');
optimize([Y_init, eye(nx);eye(nx), X_init]>=1e-3*eye(2*nx),(norm(X_init)+norm(Y_init)))
X_0 = double(X_init); 
Y_0 = double(Y_init);
% X_0 = 10 + 1.*randn(nx);


% X_0 = 1/10*eye(nx);
% Y_0 = eye(nx);
% Y_0 = 10 + 2.*randn(nx);
Lambda_0 = diag(ones(nwu,1));
% klmn_0(1:nx, nx+nwp+1:nx+nwp+ny) = X_0*A_lin*Y_0;

%%

U_0 = X_0;
V_0 = X_0^(-1)-Y_0;
assert (norm((eye(nx)-Y_0*X_0) - U_0*V_0)<1e-3)
% V*U' == I - Y_0*X_0 must be satisfied
T_l_0 = [U_0, X_0*A_lin, zeros(nx,nzu);
    zeros(nu,nx), eye(nu), zeros(nu,nwu);
    zeros(nzu, nx+nu), Lambda_0];
T_r_0 = [zeros(nx, nx+nwp), V_0', zeros(nx,nwu);
    zeros(nzp, nx), eye(nzp), zeros(nzp, nx+nwu);
    eye(nx), zeros(nx,nwp), Y_0, zeros(nx,nwu);
    zeros(nzu,nx+nwp+ny), eye(nwu)];
T_s_0 = [zeros(nx,nx+nwp), X_0*A_lin*Y_0, zeros(nx,nwu);
    zeros(nu, nx+nwp+ny+nwu);
    zeros(nzu, nx+nwp+ny+nwu)];
omega_0 = T_l_0^(-1) * (klmn_0 - T_s_0) * T_r_0^(-1);

cal = S_s + S_l*omega_0*S_r;
A_cal = cal(1:nx*2, 1:nx*2);
B1_cal = cal(1:nx*2, nx*2+1:nx*2+nwp);
B2_cal = cal(1:nx*2, nx*2+1+nwp:end);

C1_cal = cal(nx*2+1: nx*2+nzp, 1:nx*2);
D11_cal = cal(nx*2+1:nx*2+nzp, nx*2+1:nx*2+nwp);
D12_cal = cal(nx*2+1:nx*2+nzp, nx*2+1+nwp:end);

C2_cal = cal(nx*2+nzp+1:end, 1:nx*2);
D21_cal = cal(nx*2+nzp+1:end, nx*2+1:nx*2+nwp);
D22_cal = cal(nx*2+nzp+1:end, nx*2+1+nwp:end);

L1 = [A_cal, B1_cal, B2_cal;
    eye(2*nx), zeros(2*nx, nwp), zeros(2*nx, nwu)];
L2 = [C1_cal, D11_cal, D12_cal;
    zeros(nwp, nx*2), eye(nwp), zeros(nwp, nwu)];
L3 = [C2_cal, D21_cal, D22_cal;
    zeros(nzu, nx*2), zeros(nzu, nwp), eye(nwu)];
Y_cal = [Y_0, eye(nx);V_0', zeros(nx,nx)];
Z_cal = [eye(nx), zeros(nx,nx); X_0, U_0];
X_cal = (Y_cal')^(-1)*Z_cal;

X_cal = sdpvar(nx*2,nx*2);
ga = sdpvar(1,1);

P_0 = L1'*[X_cal, zeros(nx*2,nx*2);zeros(nx*2,nx*2), -X_cal]*L1 ...
    + L2'*[eye(nzp), 0;0, -ga*eye(nwp)]*L2 ...
    + L3'*[Lambda_0, zeros(nzu,nzu);zeros(nzu,nzu), -Lambda_0]*L3;

lmi = [];
lmi = lmi + [P_0 <= 0];
lmi = lmi + [X_cal >= 0];
optimize(lmi, ga, sdpsettings('Verbose', false))
fprintf('gamma: %f\n', sqrt(double(ga)))

fprintf('max real eig of P_0: %f \n',max(real(eig(double(P_0)))));
%%
% static

ga = double(ga);
% ga = 100;
P21_1_0 = [A_lin*Y_0, A_lin, B_lin, zeros(nx,nwu);
    zeros(nu,nx), X_0*A_lin, X_0*B_lin, zeros(nu,nwu);
    C_lin*Y_0, C_lin, zeros(nzp,nwp), zeros(nzp,nwu);
    zeros(nzu,nx+nwp+ny+nwu)];
P21_2_0 = [zeros(nx,nx), A_lin, zeros(nx,nwu);
    eye(nx), zeros(nx,nx+nwu);
    zeros(nzp,nx), C_lin, zeros(nzp,nwu);
    zeros(nzu,nx+nx), eye(nwu)];
P21_3_0 = [zeros(nx,nx), eye(nx), zeros(nx, nwp+nwu);
    zeros(nwp,nx+nx), eye(nwp), zeros(nwp,nwu);
    eye(ny), zeros(ny, nx+nwp+nwu);
    zeros(nwu,nx+nx+nwp), eye(nwu)];
% P21_0 = P21_1_0 + P21_2_0* (T_l_0*omega_0*T_r_0+T_s_0) * P21_3_0;
P21_0 = P21_1_0 + P21_2_0* klmn_0 * P21_3_0;
% P21_0 = P21_1_0;

YcalXcalYcal = [Y_0, eye(nx);eye(nx), X_0];
P11_0 = - blkdiag(YcalXcalYcal,ga^2*eye(nwp),Lambda_0);
P22_0 = - blkdiag(YcalXcalYcal,eye(nwp),Lambda_0);

P_0 = [P11_0,P21_0';P21_0, P22_0];
% check if initial parameter set is feasible
fprintf('max real eigenvalue of P_0: %f\n', max(real(eig(P_0))))

% extract matrices
A_tilde = omega_0(1:nx, 1:nx);
B1_tilde = omega_0(1:nx, nx+1:nx+nwp);
B2_tilde = omega_0(1:nx, nx+nwp+1:nx+nwp+ny);
B3_tilde = omega_0(1:nx, nx+nwp+ny+1:end);
C1_tilde = omega_0(nx+1:nx+nu, 1:nx);
D11_tilde = omega_0(nx+1:nx+nu, nx+1:nx+nwp);
D12_tilde = omega_0(nx+1:nx+nu, nx+nwp+1:nx+nwp+ny);
D13_tilde = omega_0(nx+1:nx+nu, nx+nwp+ny+1:end);
C2_c = omega_0(nx+nu+1:end, 1:nx);
D21_c = omega_0(nx+nu+1:end, nx+1:nx+nwp);
D22_c = omega_0(nx+nu+1:end, nx+nwp+1:nx+nwp+ny);
D23_c = zeros(nzu,nwu);
% D23 = omega_0(nx+nu+1:end, nx+nwp+ny+1:end);

% revert loop transformation to have nonlinearity in sector [alpha, beta]
alpha = config.parameters.alpha;
beta = config.parameters.beta;
J = 2 / (alpha - beta);
L = (alpha + beta) / 2;
B3_c = J * B3_tilde;
A_c = A_tilde - L * B3_c * C2_c;
B1_c = B1_tilde - L * B3_c * D21_c;
B2_c = B2_tilde - L * B3_c * D22_c;

D13_c = J * D13_tilde;
C1_c = C1_tilde - L * D13_c * C2_c;
D11_c = D11_tilde - L * D13_c * D21_c;
D12_c = D12_tilde - L * D13_c * D22_c;

omega_0 = [A_c, B1_c, B2_c, B3_c;C1_c, D11_c, D12_c, D13_c;C2_c, D21_c, D22_c, D23_c];

% make simulation with initial parameter set
% this should behave as the linear model

% linear system
lin_sys = dss(A_lin,[B_lin, A_lin], [C_lin; eye(nx)], [zeros(nzp,nwp), C_lin;zeros(ny,nwp), zeros(ny,nu)],eye(nx),config.parameters.time_delta);

% controller
con = dss(A_c,[B1_c B2_c B3_c], [C1_c;C2_c], [D11_c, D12_c, D13_c; D21_c, D22_c, D23_c], eye(nx), config.parameters.time_delta);

% simulate system in simulink
hyb_out_0 = sim('hybrid.slx');

%%
Y = sdpvar(nx,nx);
X = sdpvar(nx,nx);

% Lambda = diag(sdpvar(nwu,1));
La = diag(sdpvar(nwu,1));
ga = 100;
ga = 50;
% Static ZF
% La = sdpvar(nwu, nwu, 'full');
% multiplier_constraint = [ones(nwu, 1)' * La >= 0; La * ones(nwu, 1) >= 0];
% for i = 1 : nwu
%    for j = 1 : nwu
%        if i ~= j
%            multiplier_constraint = [multiplier_constraint; La(i, j) <= 0];
%        end
%    end
% end


klmn = sdpvar(nx+nu+nzu,nx+nwp+ny+nwu);

P21_1 = [A_lin*Y, A_lin, B_lin, zeros(nx,nwu);
    zeros(nu,nx), X*A_lin, X*B_lin, zeros(nu,nwu);
    C_lin*Y, C_lin, zeros(nzp,nwp), zeros(nzp,nwu);
    zeros(nzu,nx+nwp+ny+nwu)];
P21_2 = [zeros(nx,nx), A_lin, zeros(nx,nwu);
    eye(nx), zeros(nx,nx+nwu);
    zeros(nzp,nx), C_lin, zeros(nzp,nwu);
    zeros(nzu,nx+nx), eye(nwu)];
P21_3 = [zeros(nx,nx), eye(nx), zeros(nx, nwp+nwu);
    zeros(nwp,nx+nx), eye(nwp), zeros(nwp,nwu);
    eye(ny), zeros(ny, nx+nwp+nwu);
    zeros(nwu,nx+nx+nwp), eye(nwu)];
P21 = P21_1 + P21_2* klmn * P21_3;
P11 = -[Y, eye(nx), zeros(nx,nwp+nwu);
    eye(nx), X, zeros(nx, nwp+nwu);
    zeros(nzp, nx+nx), ga^2*eye(nwp), zeros(nzp,nwu);
    zeros(nzu, nx+nx+nwp), La];
P22 = -[Y, eye(nx), zeros(nx,nwp+nwu);
    eye(nx), X, zeros(nx, nwp+nwu);
    zeros(nzp, nx+nx), eye(nwp), zeros(nzp,nwu);
    zeros(nzu, nx+nx+nwp), La];

lmis = [];
lmis = lmis + [[P11,P21';P21, P22]<= -1e-4*eye((nx+nx+nzp+nzu)*2)];
lmis = lmis + [[Y, eye(nx);eye(nx), X]>= 0];
% lmis = lmis + multiplier_constraint;

% optimize(lmis, norm(klmn_0 - klmn)+norm(X_0 - X)+norm(Y_0-Y)+norm(Lambda_0-La))
% rand_L3 = normrnd(0,1/nx,nx,nwu);
rand_L3 = normrnd(0,10,nx,nwu);
rand_M2 = normrnd(0,10,nzu,nx);
rand_N21 = normrnd(0,1,nzu,nwp);
rand_N22 = normrnd(0,1, nzu,ny);
% optimize(lmis, norm(rand_L3 - L3)+norm(rand_M2-M2)+norm(rand_N21-N21)+norm(rand_N22-N22))
sdpset = sdpsettings('verbose',0);

optimize(lmis, norm(klmn-klmn_0), sdpset);
% optimize(lmis, norm(klmn),sdpsettings('verbose', false));
% checkset(lmis)
% verify P is negative semi-definite
% fprintf('Distance ||klmn_0 - klmn|| + ||X_0 - X|| + ||Y_0 - Y|| + ||Lambda_0 - Lambda||: %f\n',norm(klmn_0-double(klmn))+norm(X_0 - double(X))+norm(Y_0-double(Y))+norm(Lambda_0-double(La)))
fprintf('Distance ||klmn_0 - klmn||: %f\n',norm(klmn_0-double(klmn)))
fprintf('Max real eigenvalue of P: %f\n', max(real(eig(double([P11, P21';P21, P22])))))

X = double(X); Y = double(Y);
U = X;
V = X^(-1)-Y;
La = double(La);
assert (norm(V*U'-eye(nx)+Y*X) < 1e-5)
% V*U' == I - Y_0*X_0 must be satisfied

%%
% T_l = [U, X*A_lin, zeros(nx,nzu);
%     zeros(nu,nx), eye(nu), zeros(nu,nwu);
%     zeros(nzu, nx+nu), La];
% T_r = [zeros(nx, nx+nwp), V', zeros(nx,nwu);
%     zeros(nzp, nx), eye(nzp), zeros(nzp, nx+nwu);
%     eye(nx), zeros(nx,nwp), Y, zeros(nx,nwu);
%     zeros(nzu,nx+nwp+ny), eye(nwu)];
% T_s = [zeros(nx,nx+nwp), X*A_lin*Y, zeros(nx,nwu);
%     zeros(nu, nx+nwp+ny+nwu);
%     zeros(nzu, nx+nwp+ny+nwu)];
% omega = T_l^(-1) * (double(klmn) - T_s) * T_r^(-1);

omega = sys_matrices.omega;

A_tilde = omega(1:nx, 1:nx);
B1_tilde = omega(1:nx, nx+1:nx+nwp);
B2_tilde = omega(1:nx, nx+nwp+1:nx+nwp+ny);
B3_tilde = omega(1:nx, nx+nwp+ny+1:end);
C1_tilde = omega(nx+1:nx+nu, 1:nx);
D11_tilde = omega(nx+1:nx+nu, nx+1:nx+nwp);
D12_tilde = omega(nx+1:nx+nu, nx+nwp+1:nx+nwp+ny);
D13_tilde = omega(nx+1:nx+nu, nx+nwp+ny+1:end);
C2_c = omega(nx+nu+1:end, 1:nx);
D21_c = omega(nx+nu+1:end, nx+1:nx+nwp);
D22_c = omega(nx+nu+1:end, nx+nwp+1:nx+nwp+ny);
D23_c = zeros(nzu,nwu);
% D23 = omega(nx+nu+1:end, nx+nwp+ny+1:end);

% revert loop transformation to have nonlinearity in sector [alpha, beta]
alpha = config.parameters.alpha;
beta = config.parameters.beta;
J = 2 / (alpha - beta);
L = (alpha + beta) / 2;
B3_c = J * B3_tilde;
A_c = A_tilde - L * B3_c * C2_c;
B1_c = B1_tilde - L * B3_c * D21_c;
B2_c = B2_tilde - L * B3_c * D22_c;

D13_c = J * D13_tilde;
C1_c = C1_tilde - L * D13_c * C2_c;
D11_c = D11_tilde - L * D13_c * D21_c;
D12_c = D12_tilde - L * D13_c * D22_c;

fprintf('size of B3_c: %f\n', norm(B3_c))
fprintf('size of C2_c: %f\n', norm(C2_c))
fprintf('size of D21_c: %f\n', norm(D21_c))
fprintf('size of D22_c: %f\n', norm(D22_c))

omega = [A_c, B1_c, B2_c, B3_c;C1_c, D11_c, D12_c, D13_c;C2_c, D21_c, D22_c, D23_c];

% linear system
lin_sys = dss(A_lin,[B_lin, A_lin], [C_lin; eye(nx)], [zeros(nzp,nwp), C_lin;zeros(ny,nwp), zeros(ny,nu)],eye(nx),config.parameters.time_delta);

% controller
con = dss(A_c,[B1_c B2_c B3_c], [C1_c;C2_c], [D11_c, D12_c, D13_c; D21_c, D22_c, D23_c], eye(nx), config.parameters.time_delta);

% simulate system in simulink
hyb_out = sim('hybrid.slx');


%% true system
% simulate true system
sys_par = data_config.system;
x0 = data_config.simulator.initial_state;
u_ts = wp_ts;
true_out = sim('coupled_msd_4.slx');

%% plot
plot(hyb_out.zp_ts, 'linewidth', 2),hold on,grid on,
plot(hyb_out_0.zp_ts, 'linewidth', 2)
plot(t,zp_lin,'--', 'linewidth', 2)
plot(true_out.y_true, '--', 'linewidth', 2)
plot(t,u, 'linewidth', 2)
legend('proj', 'zero init','lin','true', 'input', 'interpreter', 'latex', 'fontsize',16)

