
net = make_ffnet(3, [196,300,3], [true, true, false]);

goals = eye(3); goals(goals==0) = -1;

load('small_digits.mat');

% supervised learning: stochastic gradient descent

rho=0.005; alpha=0.1;
for e=1:100 %epochs
   fprintf('------------ EPOCH %d, rho=%f-----------\n',e,rho);
   for i=1:4000
      % random digit
      d = floor(rand*3)+1;
      % random picture with digit d
      [Nim,~] = size(train{d});
      im = floor(rand*Nim)+1;
      % evaluate the current network with this image
      net = ffnet_eval(net, train{d}(im,:));
      if i<10
         E = net.E(net.O{end},goals(:,d));
         fprintf('%4d) training with digit %d, sample %5d. Achieved score: %0.15f\n',i,d-1,im,E);
      end
      net = ffnet_backprop(net,goals(:,d),rho,alpha);
   end
   rho = rho*0.95;
end
