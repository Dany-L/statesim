

function f = Gamma(d)
    if -d < -1
        f = d+0.75;
    elseif (-1 < d) && (d < 1)
        f = 0.25*d;
    else
        f = d-0.75;
    end
end


