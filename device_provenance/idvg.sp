BSIMCMG NMOS Id-Vgs
.options abstol=1e-15
VD dd 0 0.05
VG gg 0 0
VS ss 0 0
VB bb 0 0
.include Modelcards/modelcard.nmos
N1 dd gg ss bb BSIMCMG_osdi_N
.control
pre_osdi ../../../lib/ngspice/BSIMCMG.osdi
dc VG 0 1.0 0.005
wrdata idvg.csv i(VS)
.endc
.end
