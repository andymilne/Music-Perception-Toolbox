function tf = densityHasKernelCov(dens)
%DENSITYHASKERNELCOV  True if dens carries a matrix-valued kernel covariance.

    tf = false;
    if ~isstruct(dens) || ~isfield(dens, 'kernelCov')
        return;
    end
    kc = dens.kernelCov;
    if iscell(kc)
        tf = any(~cellfun(@isempty, kc));
    else
        tf = ~isempty(kc);
    end
end
