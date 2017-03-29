function run_demoVelodyne (base_dir,calib_dir)
% KITTI RAW DATA DEVELOPMENT KIT
% 
% Demonstrates projection of the velodyne points into the image plane
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)
% calib_dir ... absolute path to directory that contains calibration files

% clear and close everything
close all; dbstop error; clc;
disp('======= KITTI DevKit Demo =======');

% options (modify this to select your sequence)

base_dir  = '/home/rdx/personal/kitti/2011_09_26/2011_09_26_drive_0001_sync';

calib_dir = '/home/rdx/personal/kitti/2011_09_26';

cam       = 2; % 0-based index
frame     = 51; % 0-based index

% load calibration
calib = loadCalibrationCamToCam(fullfile(calib_dir,'calib_cam_to_cam.txt'));
Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir,'calib_velo_to_cam.txt'));

% compute projection matrix velodyne->image plane
R_cam_to_rect = eye(4);
R_cam_to_rect(1:3,1:3) = calib.R_rect{1};
P_velo_to_img = calib.P_rect{cam+1}*R_cam_to_rect*Tr_velo_to_cam;

% load and display image
img = imread(sprintf('%s/image_%02d/data/%010d.png',base_dir,cam,frame));
fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
imshow(img); hold on;
img2=imread('umm_road_000001.png')
% load velodyne points
fid = fopen(sprintf('%s/velodyne_points/data/%010d.bin',base_dir,frame),'rb');
velo = fread(fid,[4 inf],'single')';
velo = velo(1:5:end,:); % remove every 5th point for display speed
fclose(fid);

% remove all points behind image plane (approximation
idx = velo(:,1)<5;
velo(idx,:) = [];
A=[]
% project to image plane (exclude luminance)
velo_img = project(velo(:,1:3),P_velo_to_img);
idx2=velo_img(:,1)<0;
% plot points
cols = jet;
for i=1:size(velo_img,1)
  col_idx = round(64*5/velo(i,1));
  plot(velo_img(i,1),velo_img(i,2),'o','LineWidth',4,'MarkerSize',1,'Color',cols(col_idx,:));
end
for i=1:size(velo_img,1)
  col_i=int32(velo_img(i,1));
  row_i=int32(velo_img(i,2));
  if col_i<1242 && col_i>0 && row_i<375 && row_i>0
        aaa=img2(row_i,col_i,:);
        aa=reshape(aaa,[1,3]);
        if isequal(aa,[255,0,255])
                C=[row_i,col_i,velo(i,1)];
                A = vertcat(A,C);
        end
  end
end

B=single(A);
ptclound=pointCloud(B)
[model, inlier, outlier] = pcfitplane(ptclound,10.0);

