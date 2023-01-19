

function ffnet = make_ffnet(Nlayers, Nneurons, hasBiasNeuron)

ffnet.Nlayers       = Nlayers;
ffnet.Nneurons      = Nneurons;
ffnet.hasBiasNeuron = hasBiasNeuron;
hasBiasNeuron(Nneurons) = false; % output layer has no bias neuron
for l=1:Nlayers
   ffnet.O{l}  = ones(Nneurons(l)+hasBiasNeuron(l),1);
   if l>1
      ffnet.I{l}    = zeros(Nneurons(l),1); % output without activation funct.
      ffnet.dEdI{l} = zeros(Nneurons(l),1); % place holder for dE/dI
   else
      ffnet.I{l} = [];
      ffnet.dEdI{l} = [];
   end
end

% activation functions for each non-input, non-output layer
for l=2:Nlayers-1
   ffnet.factiv{l}  = @(x) max([0,x]);  % Rectified linear unig ReLU
   ffnet.dfactiv{l} = @(x) double(x>0); % dfactiv/dx
   %ffnet.factiv{l}  = @(x) tanh(x);
   %ffnet.dfactiv{l} = @(x) 1/cosh(x)^2;
end

% linear activation function for the output layer
%ffnet.factiv{Nlayers} = @(x) x;  % linear
%ffnet.dfactiv{Nlayers} = @(x) 1; % dfactiv/dx
ffnet.factiv{Nlayers}  = @(x) tanh(x);
ffnet.dfactiv{Nlayers} = @(x) 1/cosh(x)^2;

%TODO: softmax activation funciton for the output layer

% create and initialize weights (dense layers)
for l=1:Nlayers-1
   M = length(ffnet.O{l});
   N = ffnet.Nneurons(l+1);
   sig = sqrt(2/(M+N));
   %sig = 0.05;
   ffnet.w{l} = randn(M,N)*sig;    % initialization of weights between layer l and l+1
   ffnet.dEdw{l}    = zeros(M,N);  % place holder for gradients
   ffnet.dwold{l} = zeros(M,N);    % place holder for last weight-change
end

% funcion for computing the objective score
ffnet.E  = @(xout,xgoal) mean((xout-xgoal).^2);
ffnet.dE = @(xout,xgoal) 2*(xout-xgoal)/length(xout); % gradient of E
