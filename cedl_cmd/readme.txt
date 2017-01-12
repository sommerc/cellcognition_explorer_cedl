* build docker
docker build -t sommerc/cedl-gui -f DockerfileGUI .

* run docker with xserver ip and shares mounted
winpty docker run -e DISPLAY=10.50.162.248:0 -v //c//://root//data -v //c//://root//C_win  sommerc/cedl-gui

* run nsis installer creation
c:\Program Files (x86)\NSIS\makensis.exe <nsis-file>
