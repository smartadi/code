clear all;
close all;
clc;

lambda = - 0 - 5i;

t = 0:0.1:1000;
x0=1;
phi = 0.1;
x = x0*exp(lambda*t);

figure()
plot(t,x)

% Generate basis
n = 10
V = 0.5*ones(n,n) - rand(n)
rank(V)

V = V./vecnorm(V,2,1);

V(:,1)'*V(:,2)


% Gram Schmidt
Q = zeros(n,n);
R = zeros(n,n);
for j=1:n
    v=V(:,j);
    for i=1:j-1
        R(i,j) = Q(:,j)'*V(:,j);
        v = v-R(i,j)*Q(:,i);
    end
    R(j,j) = norm(v);
    Q(:,j) = v/R(j,j);
end

% EigV of random symmetric matrix
A = rand(n)
A = A.*A';
[E D] = eig(A)

norm(E(:,1))
E(:,1)'*E(:,2)


%% Distribute eigenvalues (Margianlly stable)
%a = rand(n/2,1,"like",1i);
a = 0*(rand(n/2,1)-1) + (rand(n/2,1)-.5)*1i;

% stable
% a = 0.001*(rand(n/2,1)-1) + (rand(n/2,1)-.5)*1i;

b = conj(a);
D = diag([a;b]);

%% New system
B = E*D*E';

x0 = rand(10,1);
x=[];
for i = t
 
    x = [x,expm(B*i)*x0];
end
figure()
plot(t,x)
title('latent dynamics')

%% Fast and Slow
eps = 32;
EPS = [eye(n/2),zeros(n/2,n/2);
    zeros(n/2,n/2),1/eps*eye(n/2)];
B = EPS*E*D*E';

%x0 = rand(10,1);
C = 1*(1-2*rand(n,n));
x=[];
for i = t
 
    x = [x,expm(B*i)*x0];
    
end
y = C*x;
figure()
plot(t,x)
title('latent dynamics with fast and slow dynamics')

figure()
subplot(2,1,1)
plot(t,x(1:n/2,:))
title('fast dynamics')
subplot(2,1,2)
plot(t,x(n/2+1:end,:))
title('slow dynamics')


figure()
plot(t,y)
title('output with fast and slow dynamics')


%% Mix and Match
NP = [rand(n/2,n/2),zeros(n/2,n/2)];
WF = [zeros(n/2,n/2),ones(n/2,n/2)];

NP = [eye(n/2),zeros(n/2,n/2)];
WF = [zeros(n/2,n/2),eye(n/2)];

yNP = NP*x;% %%
% 
% Im = rand(n/2,n/2);
% Im_WF = Im*yWF(:,1)*yWF(:,1)';
% close all
% 
% for i = 1:length(t)
% %for i = 1:1000
%     
%     Im_WF = real(Im*yWF(:,i)*yWF(:,i)');
% 
% figure(1)
% image(Im_WF,'CDataMapping','scaled');hold off;
% colorbar
% caxis([-1 1]);
% 
% F(i) = getframe(gcf);
% drawnow
% end
%   writerObj = VideoWriter('myVideo2.avi');
%   writerObj.FrameRate = 30;
% 
% 
%   open(writerObj);
% % write the frames to the video
% for i=1:length(F)
%     % convert the image to a frame
%     frame = F(i) ;    
%     writeVideo(writerObj, frame);
% end
% % close the writer object
% close(writerObj);

yWF = WF*x;

figure()
subplot(2,1,1)
plot(t,yWF)
title('WF')

subplot(2,1,2)
plot(t,yNP)
title('NP')

%% Image generation
% 
% Im = rand(n/2,n/2);
% Im_WF = Im*yWF(:,1)*yWF(:,1)';
% close all
% 
% for i = 1:length(t)
% %for i = 1:1000
%     
%     Im_WF = real(Im*yWF(:,i)*yWF(:,i)');
% 
% figure(1)
% image(Im_WF,'CDataMapping','scaled');hold off;
% colorbar
% caxis([-1 1]);
% 
% F(i) = getframe(gcf);
% drawnow
% end
%   writerObj = VideoWriter('myVideo2.avi');
%   writerObj.FrameRate = 30;
% 
% 
%   open(writerObj);
% % write the frames to the video
% for i=1:length(F)
%     % convert the image to a frame
%     frame = F(i) ;    
%     writeVideo(writerObj, frame);
% end
% % close the writer object
% close(writerObj);
