# Exp7b — CoT SC / PPL by Correctness  (NQ)

Hypothesis check: under CoT, does SC beat PPL, and does the gap grow with length?

## Overall

### ALL   (n=500, acc=40.6%, mean_n_tok=246)
  SC     correct: n=203  mean= 16.552  median= 16.735  std= 1.321
         incorrect: n=297  mean= 15.714  median= 16.334  std= 2.441
         gap(c-i)=+0.838   AUC=0.5812
  PPL    correct: n=203  mean=  1.291  median=  1.276  std= 0.094
         incorrect: n=297  mean=  1.301  median=  1.282  std= 0.110
         gap(c-i)=-0.010   AUC=0.5160

## Sliced by output length (CoT token count)

### short  <50   (n=8, acc=0.0%, mean_n_tok=18)
  SC     correct: n=  0  mean=    nan  median=    nan  std=   nan
         incorrect: n=  8  mean= 11.424  median= 11.449  std= 0.413
         gap(c-i)=+nan   AUC=nan
  PPL    correct: n=  0  mean=    nan  median=    nan  std=   nan
         incorrect: n=  8  mean=  1.513  median=  1.503  std= 0.135
         gap(c-i)=+nan   AUC=nan

### med   50-150   (n=119, acc=64.7%, mean_n_tok=118)
  SC     correct: n= 77  mean= 16.301  median= 16.324  std= 1.224
         incorrect: n= 42  mean= 16.062  median= 16.392  std= 1.485
         gap(c-i)=+0.238   AUC=0.5331
  PPL    correct: n= 77  mean=  1.300  median=  1.277  std= 0.092
         incorrect: n= 42  mean=  1.333  median=  1.308  std= 0.137
         gap(c-i)=-0.033   AUC=0.5686

### long  151-300   (n=237, acc=44.7%, mean_n_tok=211)
  SC     correct: n=106  mean= 16.640  median= 16.826  std= 1.285
         incorrect: n=131  mean= 16.219  median= 16.351  std= 1.483
         gap(c-i)=+0.421   AUC=0.5910
  PPL    correct: n=106  mean=  1.295  median=  1.284  std= 0.091
         incorrect: n=131  mean=  1.321  median=  1.311  std= 0.101
         gap(c-i)=-0.026   AUC=0.5746

### xlong 301+   (n=136, acc=14.7%, mean_n_tok=430)
  SC     correct: n= 20  mean= 17.048  median= 17.477  std= 1.694
         incorrect: n=116  mean= 15.312  median= 16.599  std= 3.205
         gap(c-i)=+1.736   AUC=0.6409
  PPL    correct: n= 20  mean=  1.236  median=  1.220  std= 0.104
         incorrect: n=116  mean=  1.253  median=  1.247  std= 0.076
         gap(c-i)=-0.017   AUC=0.5961
