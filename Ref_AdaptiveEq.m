function [y,w,e] = adaptive_equalizer_PAM(x,a,P)
%ADAPTIVE_EQUALIZER_PAM     Adaptive equalization at 2SPS
%   This function applies a time-domain adaptive equalizer to the received
%   signal at 2SPS. The LMS-based real-valued equalizer can either FFE or
%   DFE. In case of data-aided transmission,
%   an aligned training sequence is required .
%
%   INPUTS:
%   x           :=  Array of signals (column vectors) to process at 2SPS
%   a           :=  Transmit sequence
%   Parameters  :=  Equalizer parameters sub-structure
%
%   OUTPUTS
%   y           :=  Array of processed signals (column vectors) at 2SPS
%   w           :=  Equalizer taps [Nrx*Nd*Ntaps x Ntx*Nd]
%   e           :=  Equalizer error
%% Retrieve parameters
Ntaps = P.Ntaps;                                                % taps of the adaptive equalizer
nSpS = P.nSpS;                                                  % number of samples pes symbol
mus = P.mus.*[1;0.5];                                                    % adaptation coefficents
methods = P.methods;                                            % equalizer algorithms 'lms','lms_dd'
eqmode = P.eqmode;                                              % equalizer type FFE DFE MLSE
Ks = P.Ks;                                                      % number of symbols to run the equalizer
C = P.C;                                                        % constellation

%% Check input parameters
validateattributes(x,{'double'},{'2d'},'','x',1);
validateattributes(a,{'numeric'},{'2d'},'','a',2);
validateattributes(Ntaps(1),{'numeric'},{'scalar','positive','integer','even'},'Parameters','Ntaps',3);
validateattributes(nSpS,{'numeric'},{'nonnegative'},'Parameters','nSpS',3);
validateattributes(methods,{'cell'},{'column'},'Parameters','methods',3);
validateattributes(eqmode,{'char'},{'scalartext'},'Parameters','eqmode',3);
validateattributes(mus,{'numeric'},{'nonnegative','size',[size(methods,1) 1]},'Parameters','mus',3);
validateattributes(Ks,{'numeric'},{'nonnegative','size',[size(methods,1) 1]},'Parameters','Ks',3);
validateattributes(C,{'numeric'},{'column'},'','C');

%% Precalculate parameters
Lx = size(x,1);                                                             % number of samples to process
Nrx = size(x,2);                                                            % number of received signals
Ntx = size(a,2);                                                            % number of transmitted signals

%% Find correct equalizer and run it
switch eqmode
    case 'FFE'
        [y,w,e] = eq_FFE(x,Lx,Nrx,a,Ntx,mus,Ntaps,nSpS,Ks,methods,C);
    otherwise
        error(['Equalizer method ',methods{nn},' not supported.']);
end

end

%% Start of helper functions
%% Real-valued Data-Aided/ Decision-Directed FFE
function [y,w,e] = eq_FFE(x,Lx,Nrx,a,Ntx,mus,Ntaps,nSpS,Ks,methods,C)

    %% Initialize
    y = zeros(Lx,Ntx);                                                  % output signal
    e = zeros(Lx,Ntx);                                                  % error signal
    Lpk = floor(Ntaps/2);                                               % peak position in equalizer taps
    vec = -Lpk:1:Lpk-1;                                                 % preallocate data vector
    fin = 0;                                                            % initialize end index
    w = zeros(Nrx*Ntaps,Ntx);                                           % equalizer taps: LMS has to be initialized to zero

    %% Run
    for m = 1:size(methods,1)
        iniz = max(Ntaps,fin+1);                                        % starting point of equalizer
        fin = min(Lx-Ntaps-1,Ks(m));                                    % ending point of equalizer
        mu = mus(m);                                                    % adaptive coefficient
        for nn = iniz:fin
            %% Apply equalizer
            u = reshape(x(nn+vec,:),Nrx*Ntaps,1);                       % get signal
            y(nn,:)= u'*w;                                              % apply equalizer
            %% Calculate error every nSpS samples
            if rem(nn,nSpS)==0
                if strcmp(methods{m},'lms')
                    i_a = mod(nn/nSpS-1,size(a,1))+1;                    % index of training symbol
                    adec(1,:)= a(i_a,:);
                elseif strcmp(methods{m},'lms_dd')
                    adec(1,:) = pam_hard_decision(y(nn,:),C);
                end
                e(nn,:) = y(nn,:) - adec(1,:);                                       % calculate equalizer error
                w = w-mu*u*e(nn,:);                                                 % apply equalizer
            end
        end
    end
    %% End of adaptive equalizer
    y = y(Ntaps:end-Ntaps-1,:);                                                 % cut head and tail
end


%% End of adaptive equalizer
y = y(Ntaps(1):end-Ntaps(1)-1,:);                                                 % cut head and tail
w = [wf;wb];
end

%% PAM hard decision
% function d = pam_hard_decision(x,C)
% [~,dp] = min(abs(C-x).^2);
% d = C(dp);
% end

