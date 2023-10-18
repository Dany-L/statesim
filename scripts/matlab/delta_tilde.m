clear all, close all
alpha = -0.0; beta=1.0;
relu = @(x) max(x,0);

Delta_tilde = @(x) 2/(beta-alpha) * (relu(x)- (alpha+beta)/2*x);

x = -5:0.1:5;

plot(x,Delta_tilde(x)), hold on, grid on
plot(x, relu(x))
legend('$\tilde{\Delta}(x)$', 'ReLU$(x)$', 'interpreter', 'latex')
xlabel('$x$', 'interpreter', 'latex')
