%% Stable output-feedback controller construction
% LMI lecture 5 slide 24
% 1. Solve synthesis inequalities to determine X, Y and K, L, M, N.
% 2. Determine non-singular U, V with V UT = I - Y X.
% 3. Analysis inequalities on slide 18 are satisfied

A = [0, 1, 0;
    0, 0, 1;
    -1, -0.5, -0.1];
B = [0;0;1];
C = eye(3);

nx = size(A,1);nwu=size(B,2);ny=nx;nwu=5;nzu=nwu;

K = sdpvar(nx,nx);
L1 = sdpvar(nx,nwu);
L2 = sdpvar(nx,ny);
M1 = sdpvar(nzu,nx);
N11 = sdpvar(nzu,nwu);
N12 = sdpvar(nzu,ny);
M2 = sdpvar(nu,nx);
N21 = sdpvar(nu,nwu);

Y = sdpvar(nx,nx);
X = sdpvar(nx,nx);

lambda = sdpvar(nwu,1);
Lambda = diag(lambda);

A_cap = [A*Y, A;0, X*A]+[zeros(nx,nx), zeros(nx,nzu), B;eye(nx), zeros(nx,nzu), zeros(nx,nwp)]

