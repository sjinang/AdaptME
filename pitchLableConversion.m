allpvLable=recursiveFileList('D:\MIR1000release\PitchLabel', 'pv');
targetPath = 'D:\Dataset_for_MIREX\PitchLabel\';
for songID = 1:length(allpvLable)
    fprintf('Converting...%d/%d\n',songID,length(allpvLable));
    pv = load(allpvLable(songID).path);
   
    %interpolation
    newpv = zeros(2*length(pv),1);
    newpv(1:2:end) = pv;
   for i=2:2:length(newpv)-1
       if (newpv(i-1) == 0) & (newpv(i+1) ~= 0) %�e���O0�᭱���O0
          newpv(i) = 0;
       elseif (newpv(i-1) ~= 0) & (newpv(i+1) == 0) %�e�����O0�᭱�O0
          newpv(i) = newpv(i-1); 
       else
           newpv(i) = (newpv(i-1)+newpv(i+1))/2;
       end
   end
    newpv(end) = newpv(end-1);
    Freq = pitch2freq(newpv);
    TimeStamp= 0.01*[1:length(Freq)]';
    finalform = [TimeStamp Freq];
    myasciiWrite(finalform,[targetPath allpvLable(songID).name])
end