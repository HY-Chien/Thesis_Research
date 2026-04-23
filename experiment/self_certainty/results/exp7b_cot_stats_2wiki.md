# Exp7b — CoT SC / PPL by Correctness  (2WIKI)

Hypothesis check: under CoT, does SC beat PPL, and does the gap grow with length?

## Overall

### ALL   (n=500, acc=26.8%, mean_n_tok=175)
  SC     correct: n=134  mean= 15.895  median= 15.815  std= 1.249
         incorrect: n=366  mean= 14.253  median= 14.660  std= 2.109
         gap(c-i)=+1.643   AUC=0.7349
  PPL    correct: n=134  mean=  1.288  median=  1.268  std= 0.101
         incorrect: n=366  mean=  1.368  median=  1.358  std= 0.146
         gap(c-i)=-0.080   AUC=0.6681

## Sliced by output length (CoT token count)

### short  <50   (n=57, acc=0.0%, mean_n_tok=21)
  SC     correct: n=  0  mean=    nan  median=    nan  std=   nan
         incorrect: n= 57  mean= 11.607  median= 11.319  std= 0.972
         gap(c-i)=+nan   AUC=nan
  PPL    correct: n=  0  mean=    nan  median=    nan  std=   nan
         incorrect: n= 57  mean=  1.465  median=  1.447  std= 0.146
         gap(c-i)=+nan   AUC=nan

### med   50-150   (n=219, acc=31.1%, mean_n_tok=112)
  SC     correct: n= 68  mean= 15.916  median= 15.832  std= 1.100
         incorrect: n=151  mean= 14.939  median= 14.952  std= 1.508
         gap(c-i)=+0.977   AUC=0.7034
  PPL    correct: n= 68  mean=  1.268  median=  1.266  std= 0.086
         incorrect: n=151  mean=  1.372  median=  1.355  std= 0.154
         gap(c-i)=-0.105   AUC=0.7175

### long  151-300   (n=156, acc=37.2%, mean_n_tok=198)
  SC     correct: n= 58  mean= 15.839  median= 15.747  std= 1.432
         incorrect: n= 98  mean= 15.371  median= 15.341  std= 1.361
         gap(c-i)=+0.468   AUC=0.5966
  PPL    correct: n= 58  mean=  1.308  median=  1.276  std= 0.117
         incorrect: n= 98  mean=  1.377  median=  1.375  std= 0.106
         gap(c-i)=-0.069   AUC=0.6773

### xlong 301+   (n=68, acc=11.8%, mean_n_tok=451)
  SC     correct: n=  8  mean= 16.127  median= 15.987  std= 1.134
         incorrect: n= 60  mean= 13.213  median= 13.546  std= 2.577
         gap(c-i)=+2.914   AUC=0.8354
  PPL    correct: n=  8  mean=  1.321  median=  1.347  std= 0.064
         incorrect: n= 60  mean=  1.250  median=  1.224  std= 0.090
         gap(c-i)=+0.072   AUC=0.2437
