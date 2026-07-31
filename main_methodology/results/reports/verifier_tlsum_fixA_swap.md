== sanity: canonical signals reproduce stored tlsum_fixA run ==
hotpotqa: rederived EM=37.7 (stored 37.7) choice mismatches=0
2wiki: rederived EM=49.0 (stored 49.0) choice mismatches=0
musique: rederived EM=25.9 (stored 25.9) choice mismatches=0
popqa: rederived EM=53.2 (stored 53.2) choice mismatches=0

rate-matched: P(direct ent<0.3)=0.235 -> tau_d=0.0021; P(max post ent>0.8)=0.566 -> tau_p=0.0158

== configs (EM/F1 per ds, macro) ==
      config |         hotpotqa |            2wiki |          musique |            popqa |  macro EM/F1
   canonical | 37.7/49.3 + 0/- 0 | 49.0/55.8 + 0/- 0 | 25.9/34.3 + 0/- 0 | 53.2/55.5 + 0/- 0 | 41.45/48.72
   direct_tl | 37.6/49.3 + 6/- 7 | 48.5/55.5 + 5/-10 | 26.0/34.4 + 4/- 3 | 53.5/55.5 + 5/- 2 | 41.40/48.65
     post_tl | 37.2/48.7 + 2/- 7 | 48.5/55.3 + 0/- 5 | 25.5/33.8 + 1/- 5 | 53.6/55.6 + 5/- 1 | 41.20/48.36
        both | 37.3/48.8 + 7/-11 | 48.0/55.0 + 5/-15 | 26.0/34.3 + 5/- 4 | 53.8/55.6 + 7/- 1 | 41.27/48.44

both vs canonical: macro dEM -0.18 [95% CI -0.53,+0.20] p=0.386

== tau_d rate sweep (config=both, tau_p fixed rate-matched) ==
rate=0.05 tau_d=  0.0001 macro EM=41.50
rate=0.10 tau_d=  0.0003 macro EM=41.58
rate=0.15 tau_d=  0.0007 macro EM=41.55
rate=0.23 tau_d=  0.0021 macro EM=41.27 <- rate-matched
rate=0.25 tau_d=  0.0024 macro EM=41.25
rate=0.30 tau_d=  0.0040 macro EM=40.92
rate=0.40 tau_d=  0.0090 macro EM=40.30

== final adopted config (thesis headline) ==
config=both, tau_d=0.00050 (rate=0.12), tau_p=0.0158 (rate-matched)
hotpotqa 37.50/48.72 | 2wiki 48.70/55.63 | musique 26.60/34.74 | popqa 53.70/55.53 | macro 41.62/48.66
(entropy canonical for reference: 37.70/49.30 | 49.00/55.80 | 25.90/34.30 | 53.20/55.50 | 41.45/48.72)
