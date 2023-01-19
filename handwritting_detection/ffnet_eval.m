

function ffnet = ffnet_eval(ffnet, inputLayer)

if length(inputLayer) ~= ffnet.Nneurons(1)
   error('Number of inputs does not match the network\n')
end

ffnet.O{1}(1:ffnet.Nneurons(1)) = inputLayer;

for l = 2:ffnet.Nlayers
   %for n=1:ffnet.Nneurons(l)
   %   % compute the value of Neuron n in layer l
   %   x = 0;
   %   for k=1:length(ffnet.NeuronValues{l-1}) % includes bias neuron
   %      x = x+ffnet.w{l-1}(k,n)*ffnet.NeuronValues{l-1}(k); %TODO: replace by dot-product
   %   end
   %   ffnet.I{l}(n) = x;
   %   ffnet.NeuronValues{l}(n) = ffnet.factiv{l}(x);
   %end
   ffnet.I{l}(1:ffnet.Nneurons(l)) = ffnet.w{l-1}.' * ffnet.O{l-1};
   ffnet.O{l}(1:ffnet.Nneurons(l)) = arrayfun(ffnet.factiv{l},ffnet.I{l}(1:ffnet.Nneurons(l)));
end
