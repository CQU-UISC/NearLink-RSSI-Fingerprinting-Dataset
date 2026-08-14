%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wifi Indoor Localization
% Minhtu 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Nearest_Neighbour
% Euclidean Distance
% D = sqrt((ss-ss1)^2+(ss-ss1)^2+...)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SRL- KNN Using Mean Database 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter - 1 Unit = 40 inches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = 3; % The number of Nearest neighbours
Num_Mac = 24; % Number of APs per vector in database
% Cluster_Distance = 15; % dB

% Load Database 
myFolder = 'C:\Users\Administrator\Desktop\1PAPER\2_KNN\'; % Database Folder
Input = importdata([myFolder  'train_wifi_sle_ble_dataset.csv']);
Database = Input;
Length_Database = length(Database);

% Test
InputTest = importdata([myFolder  'test_wifi_sle_ble_dataset.csv']);
TestPoint = InputTest; 
NumTestingPoint = length(TestPoint); % Number of Test points

% History Buffer Option
IdealFlag = 0; % 1: Using Ideal History
                % 0: Using Real History
       
IdealHistory_Array = TestPoint(:,1:2); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Init
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Starting Position
Starting_Position = TestPoint(1,1:2);

% Assumption
v_user_max = 2; % m/s
t_request = 1; % Period of request time is 5 seconds
distance_max = v_user_max*t_request; % meters

% Estimated Location Buffer
Estimated_Position        = zeros(NumTestingPoint,2);
Distance_Error_Meters     = zeros(1,NumTestingPoint);

% History Buffer
% Num_Buff_His = 2;
History = Starting_Position;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop to find the location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic; % Start timing
for CountPoint = 1:NumTestingPoint
    CountPoint
    % Get RSSI 
    Current_RSSI = TestPoint(CountPoint,3:end);    

    Core_Weight_Point = History; % Most Recent Previous Point
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 1: Clustering (Choose Possible Zone)
    % Input: 
    % - Database: X Y Z MAC1 Mean_RSSI1 MAC2 Mean_RSSI2 MAC3 Mean_RSSI3  
    % - Current_RSSI
    % Output: Cluster includes the vectors which have RSSI values are nearest to Current_RSSI  
    % - Cluster_Array:  X Y Z MAC1 Mean_RSSI1 MAC2 Mean_RSSI2 MAC3 Mean_RSSI3  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %     NumCol_Cluster = 2 + Num_Mac;
% % % %     Size_Database = size(Database);
% % % %     Length_Database = Size_Database(1);
% % % %     NumRow_Cluster = round(Length_Database/2);
% % % %     Cluster_Vector = zeros(NumRow_Cluster,NumCol_Cluster);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Get The Set of Possible Zone from the Original Database 
% % % %     Pos_Array = NearDistanceNeighbour(Core_Weight_Point, Database, Length_Database, Cluster_Min_Distance, Cluster_Max_Distance);
% % % %     Length_Cluster = length(Pos_Array);
% % % %     for ii=1:Length_Cluster
% % % %         Cluster_Vector(ii,:) = Database(Pos_Array(ii),:);
% % % %     end
% % % %      Cluster_Vector = Cluster_Vector(1:Length_Cluster,:);
     
     Length_Cluster = Length_Database;
     Cluster_Vector = Database;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 2: Locate Position - Using Gaussian Weight
    % Input : 
    % - Current_RSSI : Array of Mean RSSI for each APs 
    % - Cluster_Array:  X Y Z MAC1 Mean_RSSI1 MAC2 Mean_RSSI2 ...   
    % - Num_Mac (Number of APs in each vector)
    % - Value of K (1,2,3)
    % Output: 
    % X Y Z
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % KNN without Prediction
    EstPosition  = Radar_KNN(Current_RSSI,Cluster_Vector,Length_Cluster, Num_Mac,K);
   
    Estimated_Position(CountPoint,:) =   EstPosition;            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update History
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if IdealFlag == 0 % Real History
        History = EstPosition; % Update recent point
    else
        History = IdealHistory_Array(CountPoint,:); % Update recent point
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Distance Error
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    True_Position = [TestPoint(CountPoint,1) TestPoint(CountPoint,2)];
    Distance_Error = (EstPosition(1)-True_Position(1))^2+(EstPosition(2)-True_Position(2))^2;
    Distance_Error_Meters(CountPoint) = sqrt(Distance_Error);
end
Average_Error = sum(Distance_Error_Meters)/length(Distance_Error_Meters)
Run_Time = toc; % End timing
% Variance Calculation 
Var_Dis_Error = 0;
for jj=1:length(Distance_Error_Meters)
    Var_Dis_Error = Var_Dis_Error + (Distance_Error_Meters(jj)-Average_Error)^2;
end
Var_Dis_Error = Var_Dis_Error/length(Distance_Error_Meters);
Std_Dis_Error = sqrt(Var_Dis_Error)

% MSE and RMSE Calculation
MSE = sum(Distance_Error_Meters.^2)/length(Distance_Error_Meters);
RMSE = sqrt(MSE);

% Determine the next file index 'x'
files = dir(fullfile(myFolder, 'result_*.csv'));
existing_indices = zeros(1, length(files));
for i = 1:length(files)
    name = files(i).name;
    num_str = regexp(name, 'result_(\d+)\.csv', 'tokens');
    if ~isempty(num_str)
        existing_indices(i) = str2double(num_str{1}{1});
    end
end
if isempty(existing_indices)
    file_index = 1;
else
    file_index = max(existing_indices) + 1;
end

% Save Metrics to CSV
metrics = table(Average_Error, Var_Dis_Error, Std_Dis_Error, MSE, RMSE, Run_Time, ...
    'VariableNames', { 'Average_Error_meters', 'Variance_meters2', ...
    'Std_Deviation_meters', 'MSE_meters2', 'RMSE_meters', 'Run_Time_seconds'});
metrics_filename = fullfile(myFolder, sprintf('result_%d.csv', file_index));
writetable(metrics, metrics_filename);
fprintf('Metrics saved to %s\n', metrics_filename);

% Display Results
fprintf('Average Error (Mean Euclidean Distance): %.4f meters\n', Average_Error);
fprintf('Variance of Error: %.4f meters^2\n', Var_Dis_Error);
fprintf('Standard Deviation of Error: %.4f meters\n', Std_Dis_Error);
fprintf('Mean Squared Error (MSE): %.4f meters^2\n', MSE);
fprintf('Root Mean Squared Error (RMSE): %.4f meters\n', RMSE);
fprintf('Algorithm Run Time: %.4f seconds\n', Run_Time);

% % CDF 
% Theshold_Array = 0:0.5:max(Distance_Error_Meters)+1;
% CDF_Array = zeros(1,length(Theshold_Array));
% 
% for ii= 1:length(Theshold_Array)
%     Count_CDF = 0;
%     for jj=1:length(Distance_Error_Meters)
%         if Distance_Error_Meters(jj) <= Theshold_Array(ii)
%             Count_CDF = Count_CDF + 1;
%         end
%     end
%     CDF_Array(ii) = Count_CDF/length(Distance_Error_Meters);
% end
% figure,plot(Theshold_Array,CDF_Array);
% title('KNN With Mean Fingerprint and Noise');
% xlabel('Distance Error (meter)');
% ylabel('CDF'); 

% ECDF Calculation and Plot
[sorted_errors, ~] = sort(Distance_Error_Meters); % Sort the errors
ecdf_values = (1:length(sorted_errors)) / length(sorted_errors); % ECDF values
figure;
stairs(sorted_errors, ecdf_values, 'LineWidth', 1.5); % Plot ECDF as a step function
title('ECDF of Positioning Errors');
xlabel('Distance Error (meter)');
ylabel('ECDF');
grid on;

% Save ECDF data to CSV
ecdf_data = [sorted_errors', ecdf_values'];
ecdf_filename = fullfile(myFolder, sprintf('ECDF_Data_%d.csv', file_index));
csvwrite(ecdf_filename, ecdf_data);
fprintf('ECDF data saved to %s\n', ecdf_filename);

% % Draw Ground Truth
% figure,plot(Database(:,1),Database(:,2),'b*');
% hold on;
% plot(IdealHistory_Array(:,1),IdealHistory_Array(:,2),'r-');
% hold on;
% plot(Estimated_Position(:,1),Estimated_Position(:,2),'k-');
% legend('RP','Ground Truth','RADAR Predicted Trajectory');
% error to CSV
error_filename = fullfile(myFolder, sprintf('Error_Data_%d.csv', file_index));
csvwrite(error_filename, Distance_Error_Meters');
fprintf('Error data saved to %s\n', error_filename);

% box
figure;
boxplot(Distance_Error_Meters');
title('Boxplot of Positioning Errors');
xlabel('KNN Algorithm');
ylabel('Distance Error (meters)');
grid on;
