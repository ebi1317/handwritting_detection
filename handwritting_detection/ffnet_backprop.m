

function net = ffnet_backprop(net, goal, rho, alpha)

% compute gradients
for l = net.Nlayers-1:-1:1
    % compute dE/dI in layer l+1
    if l+1 == net.Nlayers
       dEdO = net.dE(net.O{l+1},goal); 
       net.dEdI{l+1} = dEdO.*arrayfun(net.dfactiv{l+1}, net.I{l+1});
    else
       %for j=1:net.Nneurons(l+1)
       %   dEdI=0;
       %   for k=1:net.Nneurons(l+2)
       %      dEdI = dEdI + net.dEdI{l+2}(k) * net.w{l+1}(j,k)*net.dfactiv{l+1}(net.I{l+1}(j));
       %   end
       %   net.dEdI{l+1}(j) = dEdI;
       %end

       df = arrayfun(net.dfactiv{l+1}, net.I{l+1}(1:net.Nneurons(l+1)));
       net.dEdI{l+1} = df.* (net.w{l+1}(1:net.Nneurons(l+1),1:net.Nneurons(l+2))*net.dEdI{l+2}(1:net.Nneurons(l+2)));
    end
    
    % compute the current gradients
    %for i=1:length(net.O{l})
    %   for j=1:net.Nneurons(l+1)
    %       net.dEdw{l}(i,j) = net.dEdI{l+1}(j)*net.O{l}(i);
    %   end
    %end
    net.dEdw{l} = net.O{l} * net.dEdI{l+1}(1:net.Nneurons(l+1)).';
end

% update weights according to w -> -rho*grad E + alpha*dw_old
for l=1:net.Nlayers-1
   %for i=1:length(net.Nneurons(l))
   %   for j=1:net.Nneurons(l+1)
   %       dw = -rho * net.dEdw{l}(i,j) + alpha*net.dwold{l}(i,j);
   %       net.dwold{l}(i,j) = dw;
   %       net.w{l}(i,j) = net.w{l}(i,j) + dw;
   %   end
   %end

   net.dwold{l} = -rho* net.dEdw{l} + alpha*net.dwold{l};
   net.w{l}     = net.w{l} + net.dwold{l};
end

end
