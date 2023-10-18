function F = Gamma(d)
    if d <= -1
        F = d+0.75;
    elseif -1 < d && d < 1
        F = 0.25*d;
    else
        F = d-0.75;
    end
end
