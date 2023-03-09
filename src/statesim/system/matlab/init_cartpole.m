clear all, close all


x_bar = [0, 0, pi, 0]';
x0 = [0, 0, pi+0.1, 0]';
T_end = 10;

% cartpole parameters
g=9.81;
m_c= 1.0;
m_p = 0.1;
length = 0.5;
mu_c= 0.0;
mu_p = 0.01;

% continuous linear system
A = [0 1 0 0; ...
    0 0 -g.*m_p./(4*m_c/3 + m_p/3) mu_p./(length.*(-4*m_c/3 - m_p/3)); ...
    0 0 0 1; ...
    0 0 g.*(-m_c - m_p)./(length.*(4*m_c/3 + m_p/3)) mu_p.*(m_c + m_p)./(length.^2.*m_p.*(-4*m_c/3 - m_p/3))];
B = [0; 4./(3*(4*m_c/3 + m_p/3)); 0; 1./(length.*(4*m_c/3 + m_p/3))];
C = [1 0 0 0;0 0 1 0];
D = 0;
nx = size(A,1);
nu = size(B,2);
ny = size(C,1);

% discrete linear system
step_size = 0.01;
A_d = A*step_size + eye(nx);
B_d = B*step_size;
C_d = C;
D_d = D;



