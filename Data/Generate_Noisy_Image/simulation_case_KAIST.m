%clear all;
clc;	

%% Simulated experiment 0
Omsi       = img;          
noiselevel = 0.15*ones(1,28); 

noisy_img      = Omsi;
[M,N,p]   = size(Omsi); 
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
save('../KAIST_Noisy_data/Case0_15.mat', 'noisy_img', 'img');

% niose level 35
Omsi       = img;          
noiselevel = 0.35*ones(1,28); 

noisy_img      = Omsi;
[M,N,p]   = size(Omsi); 
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
save('../KAIST_Noisy_data/Case0_35.mat', 'noisy_img', 'img');

% niose level 55
Omsi       = img;          
noiselevel = 0.55*ones(1,28); 

noisy_img      = Omsi;
[M,N,p]   = size(Omsi); 
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
save('../KAIST_Noisy_data/Case0_55.mat', 'noisy_img', 'img');

%% Simulated experiment 1
Omsi       = img;
ratio      = 0.1*ones(1,28);           
noiselevel = 0.1*ones(1,28); 

noisy_img      = Omsi;
[M,N,p]   = size(Omsi); 
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
% S&P noise
for i = 1:p
     noisy_img(:,:,i)=imnoise(noisy_img(:,:,i),'salt & pepper',ratio(i));
end
save('../KAIST_Noisy_data/Case1.mat', 'noisy_img', 'img');

%% Simulated experiment 2
Omsi       = img;
load Simu_ratio                       
load Simu_noiselevel                 
ratio  = ratio ;
noiselevel = noiselevel ;
noisy_img      = Omsi;
[M,N,p]   = size(Omsi);
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
% S&P noise
for i = 1:p
     noisy_img(:,:,i)=imnoise(noisy_img(:,:,i),'salt & pepper',ratio(i));
end
save('../KAIST_Noisy_data/Case2.mat', 'noisy_img', 'img');

%% Simulated experiment 3
Omsi       = img;
load Simu_ratio                       
load Simu_noiselevel                 
ratio  = ratio ;
noiselevel = noiselevel ;

noisy_img      = Omsi;
[M,N,p]   = size(Omsi);
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
% S&P noise
for i = 1:p
     noisy_img(:,:,i)=imnoise(noisy_img(:,:,i),'salt & pepper',ratio(i));
end
% stripe line
for band=18:28
    num = 19+randperm(21,1);
    loc = ceil(N*rand(1,num));
    t = rand(1,length(loc))*0.5-0.25;
    noisy_img(:,loc,band) = bsxfun(@minus,noisy_img(:,loc,band),t);
end
save('../KAIST_Noisy_data/Case3.mat', 'noisy_img', 'img');


%% Simulated experiment 4
Omsi       = img;
load Simu_ratio                       
load Simu_noiselevel                 
ratio  = ratio ;
noiselevel = noiselevel ;

noisy_img      = Omsi;
[M,N,p]   = size(Omsi);
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
% S&P noise
for i = 1:p
     noisy_img(:,:,i)=imnoise(noisy_img(:,:,i),'salt & pepper',ratio(i));
end
% Deadline
for i=10:20
    indp=randperm(10,1)+2;
    ind=randperm(N-1,indp);
    an=funrand(2,length(ind));
    % searching the location of an which value is 1,2,3
    loc1=find(an==1);loc2=find(an==2);loc3=find(an==3);
    noisy_img(:,ind(loc1),i)=0; 
    noisy_img(:,ind(loc2):ind(loc2)+1,i)=0;
    noisy_img(:,ind(loc3)-1:ind(loc3)+1,i)=0;
end  
save('../KAIST_Noisy_data/Case4.mat', 'noisy_img', 'img');

%% Simulated experiment 5
Omsi       = img;
ratio      = 0.1*ones(1,28);           
noiselevel = 0.1*ones(1,28); 

noisy_img      = Omsi;
[M,N,p]   = size(Omsi); 
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
% S&P noise
for i = 1:p
     noisy_img(:,:,i)=imnoise(noisy_img(:,:,i),'salt & pepper',ratio(i));
end
% stripe line
for band=18:28
    num = 19+randperm(21,1);
    loc = ceil(N*rand(1,num));
    t = rand(1,length(loc))*0.5-0.25;
    noisy_img(:,loc,band) = bsxfun(@minus,noisy_img(:,loc,band),t);
end
% Deadline
for i=10:20
    indp=randperm(10,1)+2;
    ind=randperm(N-1,indp);
    an=funrand(2,length(ind));
    % searching the location of an which value is 1,2,3
    loc1=find(an==1);loc2=find(an==2);loc3=find(an==3);
    noisy_img(:,ind(loc1),i)=0; 
    noisy_img(:,ind(loc2):ind(loc2)+1,i)=0;
    noisy_img(:,ind(loc3)-1:ind(loc3)+1,i)=0;
end  
save('../KAIST_Noisy_data/Case5.mat', 'noisy_img', 'img');

%% Simulated experiment 6
Omsi       = img;
load Simu_ratio                       
load Simu_noiselevel
ratio  = ratio ;
noiselevel = noiselevel ;

noisy_img      = Omsi;
[M,N,p]   = size(Omsi); 
% Gaussian noise
for i = 1:p
     noisy_img(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
% S&P noise
for i = 1:p
     noisy_img(:,:,i)=imnoise(noisy_img(:,:,i),'salt & pepper',ratio(i));
end
% stripe line
for band=18:28
    num = 19+randperm(21,1);
    loc = ceil(N*rand(1,num));
    t = rand(1,length(loc))*0.5-0.25;
    noisy_img(:,loc,band) = bsxfun(@minus,noisy_img(:,loc,band),t);
end
% Deadline
for i=10:20
    indp=randperm(10,1)+2;
    ind=randperm(N-1,indp);
    an=funrand(2,length(ind));
    % searching the location of an which value is 1,2,3
    loc1=find(an==1);loc2=find(an==2);loc3=find(an==3);
    noisy_img(:,ind(loc1),i)=0; 
    noisy_img(:,ind(loc2):ind(loc2)+1,i)=0;
    noisy_img(:,ind(loc3)-1:ind(loc3)+1,i)=0;
end  
save('../KAIST_Noisy_data/Case6.mat', 'noisy_img', 'img');
