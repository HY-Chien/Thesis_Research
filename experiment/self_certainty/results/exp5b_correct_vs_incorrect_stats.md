# Exp5b — SC / PPL by Correctness

Higher SC = more confident. Lower PPL = more confident.
AUC = P(confident score ranks correct answer above incorrect one). 0.5 = chance, 1.0 = perfect.

## Overall

### ALL   (n=500, acc=36.0%)
  SC     correct: n=180  mean= 13.603  median= 13.679  std= 2.423
         incorrect: n=320  mean= 11.847  median= 11.734  std= 2.600
         gap(c-i)=+1.755   AUC=0.6843
  PPL    correct: n=180  mean=  1.235  median=  1.185  std= 0.189
         incorrect: n=320  mean=  1.567  median=  1.427  std= 0.590
         gap(c-i)=-0.333   AUC=0.7535

## Sliced by answer length (n_tokens)

### short 1-4   (n=164, acc=38.4%)
  SC     correct: n= 63  mean= 12.490  median= 12.139  std= 2.396
         incorrect: n=101  mean= 10.306  median= 10.435  std= 2.132
         gap(c-i)=+2.184   AUC=0.7504
  PPL    correct: n= 63  mean=  1.270  median=  1.218  std= 0.214
         incorrect: n=101  mean=  1.725  median=  1.514  std= 0.742
         gap(c-i)=-0.455   AUC=0.7998

### med   5-15   (n=307, acc=37.8%)
  SC     correct: n=116  mean= 14.203  median= 14.552  std= 2.236
         incorrect: n=191  mean= 12.590  median= 12.571  std= 2.483
         gap(c-i)=+1.613   AUC=0.6799
  PPL    correct: n=116  mean=  1.215  median=  1.157  std= 0.173
         incorrect: n=191  mean=  1.486  median=  1.368  std= 0.499
         gap(c-i)=-0.271   AUC=0.7371

### long  16+   (n=29, acc=3.4%)
  SC     correct: n=  1  mean= 14.001  median= 14.001  std= 0.000
         incorrect: n= 28  mean= 12.338  median= 11.812  std= 2.574
         gap(c-i)=+1.663   AUC=0.7143
  PPL    correct: n=  1  mean=  1.248  median=  1.248  std= 0.000
         incorrect: n= 28  mean=  1.547  median=  1.480  std= 0.417
         gap(c-i)=-0.300   AUC=0.7143
