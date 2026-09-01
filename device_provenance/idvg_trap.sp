BSIMCMG NMOS Id-Vgs with 5mV trap Vth shift
.options abstol=1e-15
VD dd 0 0.05
VG gg 0 0
VS ss 0 0
VB bb 0 0
.include Modelcards/modelcard_trap.nmos
N1 dd gg ss bb BSIMCMG_osdi_N
.control
pre_osdi ../../../lib/ngspice/BSIMCMG.osdi
dc VG 0 1.0 0.005
wrdata idvg_trap.csv i(VS)
.endc
.end
