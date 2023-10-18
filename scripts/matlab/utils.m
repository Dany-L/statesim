
classdef utils

    methods(Static)

        function L = lower_triangular_from_vector(L_flat, nx)
            % construct lower triangular matrix from vector.
            L = zeros(nx,nx);flat_idx = 0;
            for diag_idx = -0:-1:-nx+1
                diag_len = diag_idx + nx;
                L = L + diag(L_flat(flat_idx+1: flat_idx + diag_len),diag_idx);
                flat_idx = flat_idx + diag_len;
            end
        end
    end
end
