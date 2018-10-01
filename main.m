% The following program is a image recognition application for 
%  car plate detection. 


%{
General Ideas of Logic:

1. Read in the original image
2. Use multiple image processing techniques to calculate the area of plate
3. Elimiate and clean unnecessary small parts and remember the x y values
    for plate only
4. Go back to original photo and seperate each number or character 
5. Compare the character to images in our database
6. Apply algorithm and calculation to find the image with highest
    similarities of features (KNN or other machine learning techniques can be
    used here)
7. Control the accuracy_score to be more than 97%

%}  

% Special Notices:
% For privacy sensitivity concern in Japan and North America, 
% the training data and testing data are achieved using python crawler
% from Chinese Search Engine "BaiDu". Please contact Gu Ruozhen directly if
% you have any questions for privacy issue. For simplicity, I only put
% several images instead of the giant database (on my server) for this
% assignment submission


%----------------- start of program --------------------------------------

% clear the screen and close all before starting
clear;
close all;
clc;
% -------- whether to detect the fist kanji --------------------------------
recognize_kanji = 0;
% --------------------------------------------------------------------------
% Read in the target image for detection
% Target plate is the image we will process later
% use img to get the image
img = imread('./Plate Photo/car2.jpg');
% img now is a matrix of int(pixels)

% make subplot larger

% disp(img) %uncomment to see details data
figure('Position', [100, 100, 1200, 1000], 'name', 'Black & White');

subplot(4,2,1)
% display to double check
imshow(img,'InitialMagnification','fit');
title('1.Caputured Image to Recognize','FontSize',22);
% adjust image size

% Apply rgb method to turn the image to whilte&black only
% Doing this will make it easiser to define the target plate area
img_bw = rgb2gray(img);
subplot(4,2,2)
imshow(img_bw);
title('2.RGB To Turn Img To B&W','FontSize',22)
subplot(4,1,3)
% Common step to examine the distribution of image : histogram
imhist(img_bw);
title('3.Display histgram to see the distirbution/variance of image','FontSize',22);
subplot(4,1,4)
% Apply histogram equalization in case the graph is not evenly distributed
img_bw2 = histeq(img_bw);
imhist(img_bw2);
title('4.Applied Histogram Equalization (May not be necessary if graph is already very well evenly distributed)','FontSize',16)

% ---------- user roberts method to find edge of image --------------------

method = 'roberts';
threshold = 0.15; % values in 0.15 - 0.2 should all be fine
img_edge = edge(img_bw, method, threshold, 'both');
figure('Position', [100, 100, 1000, 900],'name','Display & Erode Edges');
subplot(2,2,1);
imshow(img_edge);
title('5.Finds edges by approximating the gradient magnitude of image','FontSize',13);

% ----------- Erode Technique to delete most unnecessary edges ------------

% Applied imerode function with se = [1:1:1]
img_erode = imerode(img_edge, [1;1;1]);
subplot(2,2,2);
imshow(img_erode);
title('6. Erode Image to leave only the edge information for center characters','FontSize',13);

% ------------ morphological dilation and erosion operations --------------

% use strel function to create a create a rect structuring element

% use rectangle shape to preserve the natural shape of plate
se = strel('rectangle', [30, 30]);
% Perform a morphological close operation on the image
img_strel = imclose(img_erode, se);
% display
subplot(2,2,3)
imshow(img_strel);
title('7. Plot flat rectangular structuring','FontSize',16);

% ------------- Remove small elements -------------------------------------

% As visualized in the figure2, we need to remove some small elements like
% "Audi" logo to leave only the plate number area left.
% A quick research will give us a useful function bwareaopen
img_plate = bwareaopen(img_strel, 2200);
subplot(2,2,4)
% display
imshow(img_plate);
title('8.Remove small elements','FontSize',16);

% -------- ------- Calculate the area of target numbers -------------------
[row, col, z] = size(img_plate);

% use double function and now the plate area is value 1 and the rests are 0
img_double = double(img_plate);

% disp(img_double) %Uncomment to verify the fact

Plate_Y = zeros(row, 1);
for i = 1:row
    for j = 1:col
        if(img_double(i, j)) % if == 1, then it is plate area
            Plate_Y(i, 1) = Plate_Y(i, 1) + 1;   
        end
    end
end
% Now along y axis, as long as the integer is not 0, it should be the area
% for plate. 
% ------------------
% Calculate y axis top:
yaxistop = 1;
while true
    % as we know rbg for blue is around 2 hundred. So if
    % the integer is too small, we do not want it(not plate area).
    % That is why i set cond > 30

    if (Plate_Y(yaxistop) < 30) && (yaxistop < row)
        yaxistop = yaxistop + 1;
    else 
        break;
    end
end

% calculate "y axis bottom"
yaxisbottom = row;
while true
    % same logic as above function but just start from bottom of list
    if (Plate_Y(yaxisbottom) < 30) && (yaxisbottom > yaxistop)
        yaxisbottom = yaxisbottom - 1;
    else 
        break;
    end
end
% ------------------
% Calculate x axis:
% same logic but reverse orders; will obmit some comments here

Plate_X = zeros(1, col);

for j = 1:col
    for i = 1:row
        if(img_double(i, j)) 
            Plate_X(1, j) = Plate_X(1, j) + 1;
        end
    end
end

% calculate "x axis left"
x_left = 1;
while true
    if (Plate_X(1, x_left) < 30) && (x_left < col)
        x_left = x_left + 1;
    else 
        break;
    end 
end
% calculate "x axis right"
x_right = col;
while true
    if (Plate_X(1, x_right) < 30) && (x_right > x_left)
        x_right = x_right - 1;
    else 
        break;
    end
end

% ------- so far we know x and y axis for the plate rectangle -------------
% -------------------------------------------------------------------------
% time to see the plate on original img using the axis we find above
disp(yaxistop)
disp(yaxisbottom)
img_plate = img(yaxistop:yaxisbottom, x_left:x_right, : );
figure('name', 'Plate Area');
imshow(img_plate);
title('9. Pinpoint Plate Area on Original Graph','FontSize',30)
truesize([900,900])

% Save current work and read again to do the next step: seperation 
imwrite(img_plate, './Plate_to_seperate.jpg')
img_plate = imread('./Plate_to_seperate.jpg');

%------------------------ Seperate the Numbers ----------------------------
img_p_bw = rgb2gray(img_plate);    % RGBÕº?Ò??ª??»Õº?Ò
figure('name','Examine distribution of numbers only')
subplot(3, 2, 1);
imshow(img_p_bw);
title('white and black');
subplot(3, 2, 2);
imhist(img_p_bw);
title('Hist: Numbers distribution on plate');
% Apply histogram equalization in case the graph is not evenly distributed
img_p_bw2 = histeq(img_p_bw);
subplot(3,2,3);
imshow(img_p_bw2);
title('Improved');
subplot(3,2,4);
imhist(img_p_bw2);
title('Improved');

%-------- filter out unnecessary information ------------------------------
img_p_bw3 = imbinarize(img_p_bw2, 0.76);
subplot(3,2,5);
imshow(img_p_bw3);
title('Binarize the img');

% use midium filter method 
img_p_bw4 = medfilt2(img_p_bw3);
% display again to compare
subplot(3,2,6);
imshow(img_p_bw4);
title('Filter out unnecessaery info');

%------- split the numbers to individual section --------------------------
[m, n] = size(img_p_bw4);
% define axis starting values
top = 1;
bottom = m;
left = 1;
right = n;

% start from top and find first row containing information
while (sum(img_p_bw4(top, :)) == 0) && (top <= m)
    top = top + 1;
end

% start from bottom and find row containing information

while (sum(img_p_bw4(bottom, :)) == 0) && (bottom >= 1)
    bottom = bottom -1;
end

% find left side of fist digit
while (sum(img_p_bw4(:, left)) == 0) && (left <= n)
    left = left + 1;
end

% note that the seperation area between two numbers will result in 
%  sum(img_o_bw4(:, left) as 0 and will terminate the loop. Thus this right
%  value will be the rightmost edge of one digit 
while (sum(img_p_bw4(:, right)) == 0) && (right >= 1)
    right = right - 1;
end

% we assume all digits are evenly distributed, so their lengths are roughly
% the same. This requires camera's position to be at the center
width = right - left;
height = bottom - top;

% finally we split the image based on above height and width
split_img = imcrop(img_p_bw4, [left top width height]);


[m, n] = size(split_img);
s = sum(split_img);    % all rows are summed to one value, columns numbers're the same
counter = 1;
start_p = 1;
end_p = 1;

% --------------- Sanity Check --------------------------------------------

% find continuous chunks, if not continuous, then we need to re seperate
while counter ~= n
    % define the start point and end point
    while s(counter) == 0 % no pixel
        counter = counter + 1;
    end
    % no we find the starting point, save value to start_p
    % start_p is the starting value we care
    start_p = counter;
    while s(counter) ~= 0 && counter <= n - 1
        counter = counter + 1;
    end
    % we find the finishing point and save value to end_p
    end_p = counter + 1;
    len_poi = end_p - start_p;
    if len_poi > round(n / 6.5) % because there is around 7 chars in total on plate
        % We adjust the size if goes into this cond
        [v1, v2] = min(sum(split_img(:, [start_p + 5 : end_p - 5])));
        split_img(:, start_p + v2 + 5) = 0;
    end
end

% ---------------- Seperation ---------------------------------------------

% Seperate the seven characters
loop = 0;
find_result = [];
while loop == 0 % we use loop because we would like to do sanity check
    [m, n] = size(split_img);
    left = 1;
    width = 0;
    while sum(split_img(:, width+1)) ~= 0
        width = width + 1;
    end
    % find the seperation posiiton for the first character
    if width < 10 % sanity check
        % if reaches here, that means there are some small parts
        %   that are not perfectly cleaned. set their value to 0
        split_img(:, [1:width]) = 0;
        % Here we need to re split the image...
    else
        % Cut it off!
        im_tmp = my_imsplit(imcrop(split_img, [1,1,width,m]));
        % New size without first char
        [m, n] = size(im_tmp);
        all = sum(sum(im_tmp));
        two_thirds=sum(sum(im_tmp([round(m/3):2*round(m/3)],:)));
        if two_thirds / all > 0.25 % larger than a quarter
            loop = 1; % find it, stop looping
            find_result = im_tmp;
        end
        split_img(:, [1:width]) = 0;
        split_img = my_imsplit(split_img);
    end
end

figure('name', 'Crop the first char');
subplot(2,4,1)
imshow(split_img);

% --------------------  Seperate each letter by calculation ---------------
[word2,split_img]=getword(split_img);
subplot(2,4,2), imshow(split_img);
[word3,split_img]=getword(split_img);
subplot(2,4,3), imshow(split_img);
[word4,split_img]=getword(split_img);
subplot(2,4,4), imshow(split_img);
[word5,split_img]=getword(split_img);
subplot(2,3,4), imshow(split_img);
[word6,split_img]=getword(split_img);
subplot(2,3,5), imshow(split_img);
[word7,split_img]=getword(split_img);
subplot(2,3,6), imshow(split_img);

% --------------------- Plot the seperated image --------------------------

figure('name','Compare Letter & Num')
subplot(5,7,1);
imshow(find_result);
title('1');
subplot(5,7,2);
imshow(word2);
title('2');
subplot(5,7,3);
imshow(word3);
title('3');
subplot(5,7,4);
imshow(word4);
title('4');
subplot(5,7,5);
imshow(word5);
title('5');
subplot(5,7,6);
imshow(word6);
title('6');
subplot(5,7,7);
imshow(word7);
title('7');

% ------- Resize and Then Plot --------------------------------------------
word1=imresize(find_result,[40 20]);
word2=imresize(word2,[40 20]);
word3=imresize(word3,[40 20]);
word4=imresize(word4,[40 20]);
word5=imresize(word5,[40 20]);
word6=imresize(word6,[40 20]);
word7=imresize(word7,[40 20]);

subplot(5,7,15);
imshow(find_result);
title('11');
subplot(5,7,16);
imshow(word2);
title('22');
subplot(5,7,17);
imshow(word3);
title('33');
subplot(5,7,18);
imshow(word4);
title('44');
subplot(5,7,19);
imshow(word5);
title('55');
subplot(5,7,20);
imshow(word6);
title('66');
subplot(5,7,21);
imshow(word7);
title('77');

% ----- write to folder ---------------------------------------------------
imwrite(word1,'1.jpg'); 
imwrite(word2,'2.jpg');
imwrite(word3,'3.jpg');
imwrite(word4,'4.jpg');
imwrite(word5,'5.jpg');
imwrite(word6,'6.jpg');
imwrite(word7,'7.jpg');
% -------------------------------------------------------------------------
% -------------- Find img in database anc compare -------------------------

code=char(['0':'9' 'A':'Z' '沪鲁苏贵陕']);

% 1-10 is number; 11 -36 is letters; 37-41 is chinese characters
% -------------------------------------------------------------------------

% Warning: depending on your computer, chinese character might be read as ?
%  so here we will ignore detecting the first chinese character in case you
%  get an error; However, detecting chinese character is no different from
%  detecting number or letters. Change it back by setting variable "recognize_kanji" to 1

% -------------------------------------------------------------------------
 subBw2 = zeros(40, 20);
 num = 1; 
 
 for i = 1:7
    if recognize_kanji == 0
        if i == 1
            continue;
        end
    end
    i_str = int2str(i);    
    word = imread([i_str,'.jpg']); 
    segBw2 = imresize(word, [40,20], 'nearest');  %resize
    segBw2 = im2bw(segBw2, 0.5);
    
    if recognize_kanji == 1
        if i == 1
            KMin = 37
            KMax = 44
        end
    end
    if i == 2  % detect second letters A-Z
        kMin = 11;
        kMax = 36;
    elseif i >= 3  5 % detect following letters and number
        kMin = 1;
        kMax = 36;
    end
    
    l = 1;
    for k = kMin : kMax
        disp(code(k))
        fname = strcat('Sample Char/',code(k),'.jpg'); 
        samBw2 = imread(fname); 
        samBw2 = im2bw(samBw2, 0.5);  
        
        for i1 = 1:40
            for j1 = 1:20
                subBw2(i1, j1) = segBw2(i1, j1) - samBw2(i1 ,j1);
            end
        end
        
        Dmax = 0;
        for i2 = 1:40
            for j2 = 1:20
                if subBw2(i2, j2) ~= 0
                    Dmax = Dmax + 1;
                end
            end
        end
        error(l) = Dmax;
        l = l + 1;
    end
    
    errorMin = min(error);
    findc = find(error == errorMin);

       
    Code(num * 2 - 1) = code(findc(1) + kMin - 1);
    Code(num * 2) = ' ';
    num = num + 1;
 end
 
 disp(Code);
 msgbox(Code,'Plate is');





