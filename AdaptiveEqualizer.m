function [y, w, e] = adaptive_equalizer_QAM(x, a, P)
% ADAPTIVE_EQUALIZER_QAM Adaptive equalization at 2SPS for QAM signals
%   INPUTS:
%   x           := Array of signals (column vectors) to process at 2SPS (complex)
%   a           := Transmit sequence (complex QAM symbols)
%   P           := Equalizer parameters sub-structure
%
%   OUTPUTS:
%   y           := Array of processed signals (column vectors) at 2SPS (complex)
%   w           := Equalizer taps [Nrx*Nd*Ntaps x Ntx*Nd] (complex)
%   e           := Equalizer error (complex)

%% Retrieve parameters
Ntaps = P.Ntaps;                % Taps of the adaptive equalizer
nSpS = P.nSpS;                  % Number of samples per symbol
mus = P.mus .* [1; 0.5];        % Adaptation coefficients
methods = P.methods;           % Equalizer algorithms: 'lms', 'lms_dd'
eqmode = P.eqmode;              % Equalizer type: 'FFE'
Ks = P.Ks;                      % Number of symbols to run the equalizer
C = P.C;                        % QAM constellation (complex)

%% Check input parameters
validateattributes(x, {'double'}, {'2d'}, '', 'x', 1);
validateattributes(a, {'numeric'}, {'2d'}, '', 'a', 2);
validateattributes(Ntaps(1), {'numeric'}, {'scalar', 'positive', 'integer', 'even'}, 'Parameters', 'Ntaps', 3);
validateattributes(nSpS, {'numeric'}, {'nonnegative'}, 'Parameters', 'nSpS', 3);
validateattributes(methods, {'cell'}, {'column'}, 'Parameters', 'methods', 3);
validateattributes(eqmode, {'char'}, {'scalartext'}, 'Parameters', 'eqmode', 3);
validateattributes(mus, {'numeric'}, {'nonnegative', 'size', [size(methods, 1) 1]}, 'Parameters', 'mus', 3);
validateattributes(Ks, {'numeric'}, {'nonnegative', 'size', [size(methods, 1) 1]}, 'Parameters', 'Ks', 3);
validateattributes(C, {'numeric'}, {'column'}, '', 'C');

%% Precalculate parameters
Lx = size(x, 1);                % Number of samples to process
Nrx = size(x, 2);               % Number of received signals
Ntx = size(a, 2);               % Number of transmitted signals

%% Initialize
y = zeros(Lx, Ntx, 'like', 1i); % Output signal (complex)
e = zeros(Lx, Ntx, 'like', 1i); % Error signal (complex)
Lpk = floor(Ntaps / 2);         % Peak position in equalizer taps
vec = -Lpk:1:Lpk - 1;           % Preallocate data vector
fin = 0;                        % Initialize end index
w = zeros(Nrx * Ntaps, Ntx, 'like', 1i); % Equalizer taps (complex)

%% Run
switch eqmode
    case 'FFE'
        [y, w, e] = eq_FFE(x, Lx, Nrx, a, Ntx, mus, Ntaps, nSpS, Ks, methods, C);
    otherwise
        error(['Equalizer method ', methods{1}, ' not supported.']);
end

%% Cut head and tail
y = y(Ntaps(1):end - Ntaps(1) - 1, :);
end

%%  FFE for QAM
function [y, w, e] = eq_FFE(x, Lx, Nrx, a, Ntx, mus, Ntaps, nSpS, Ks, methods, C)
    %% Initialize
    y = zeros(Lx, Ntx, 'like', 1i); % Output signal (complex)
    e = zeros(Lx, Ntx, 'like', 1i); % Error signal (complex)
    Lpk = floor(Ntaps / 2);
    vec = -Lpk:1:Lpk - 1;
    fin = 0;
    w = zeros(Nrx * Ntaps, Ntx, 'like', 1i); % Equalizer taps (complex)

    %% Run
    for m = 1:size(methods, 1)
        iniz = max(Ntaps, fin + 1);
        fin = min(Lx - Ntaps - 1, Ks(m));
        mu = mus(m);

        for nn = iniz:fin
            %% Apply equalizer
            u = reshape(x(nn + vec, :), Nrx * Ntaps, 1); % Get signal (complex)
            y(nn, :) = u' * w; % Apply equalizer (complex)

            %% Calculate error every nSpS samples
            if rem(nn, nSpS) == 0
                if strcmp(methods{m}, 'lms')
                    i_a = mod(nn / nSpS - 1, size(a, 1)) + 1; % Index of training symbol
                    adec = a(i_a, :); % Use training symbol (complex)
                elseif strcmp(methods{m}, 'lms_dd')
                    adec = qam_hard_decision(y(nn, :), C); % QAM hard decision
                end

                e(nn, :) = y(nn, :) - adec; % Calculate equalizer error (complex)
                w = w - mu * u * e(nn, :)'; % Update equalizer taps (complex)
            end
        end
    end
end

%% QAM Hard Decision
function d = qam_hard_decision(x, C)
    [~, idx] = min(abs(C - x).^2, [], 1); % Find closest constellation point
    d = C(idx); % Return hard-decided symbol (complex)
end
