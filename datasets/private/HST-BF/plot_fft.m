function plot_fft(t,mixedsig000,P,nfft,k,idx,xl1)
ffts=zeros(P,nfft);



  for i=1:P
      figure
      subplot(1,2,1)
      plot(t(1:length(mixedsig000(i,:))),mixedsig000(i,:)); 
      xlabel('Time(s)'); ylabel('Amplitude(m/s²)'); set(gca,'FontName','Times New Roman','FontSize',14); 

      ffts(i,:)=fft(mixedsig000(i,:),nfft)*2/nfft;
      subplot(1,2,2)
      plot(k,abs(ffts(i,idx+1))); 
      xlim([0 xl1]); xlabel('Frequency(Hz)'); ylabel('Amplitude(m/s²)'); set(gca,'FontName','Times New Roman','FontSize',14); 
      set (gcf,'Position',[400,1000-300*i,800,200]);
  end 