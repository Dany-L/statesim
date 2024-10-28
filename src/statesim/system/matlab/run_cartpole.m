init_cartpole

data_folder_name = '../../../../tests/data';
mdl_name = 'cartpole.slx';
load_system(mdl_name);
out = sim(mdl_name);

t_eval = 0:step_size:T_end-step_size;

y_c_lin = resample(out.y.y_c_lin, t_eval);
y_d_lin = resample(out.y.y_d_lin, t_eval);
y_non_lin = resample(out.y.y_non_lin, t_eval);
u = resample(out.u, t_eval);

column_names = {'t','y','u'};


writetable(table(...
    t_eval(:), ...
    squeeze(y_c_lin.Data), ...
    squeeze(u.Data), ...
    'VariableNames', column_names), ...
    sprintf('%s/%s_cartpole_linear_continous.csv', data_folder_name, datestr(now, 'yyyy_mm_dd-HH_MM_SS')));
writetable(table(...
    t_eval(:), ...
    squeeze(y_d_lin.Data), ...
    squeeze(u.Data), ...
    'VariableNames', column_names), ...
    sprintf('%s/%s_cartpole_linear_discrete.csv', data_folder_name, datestr(now, 'yyyy_mm_dd-HH_MM_SS')));
writetable(table(...
    t_eval(:), ...
    squeeze(y_non_lin.Data), ...
    squeeze(u.Data), ...
    'VariableNames', column_names), ...
    sprintf('%s/%s_cartpole_nonlinear_continuous.csv', data_folder_name, datestr(now, 'yyyy_mm_dd-HH_MM_SS')));