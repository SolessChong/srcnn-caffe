"""
This script plots the loss value.
"""

import re
import matplotlib.pyplot as plt

txt = \
"""
I0830 23:02:29.417739   796 solver.cpp:195] Iteration 20004, loss = 0.578622
I0830 23:02:29.417790   796 solver.cpp:365] Iteration 20004, lr = 1e-05
I0830 23:02:46.353974   796 solver.cpp:232] Iteration 40004, Testing net (#0)
I0830 23:02:47.138726   796 solver.cpp:270] Test score #0: 0.161416
I0830 23:02:47.142474   796 solver.cpp:287] Snapshotting to SRCNN_iter_40008
I0830 23:02:47.142845   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_40008.solverstate
I0830 23:02:47.143815   796 solver.cpp:195] Iteration 40008, loss = 0.428221
I0830 23:02:47.143838   796 solver.cpp:365] Iteration 40008, lr = 1e-05
I0830 23:03:04.043588   796 solver.cpp:232] Iteration 60006, Testing net (#0)
I0830 23:03:04.819161   796 solver.cpp:270] Test score #0: 0.133597
I0830 23:03:04.825080   796 solver.cpp:287] Snapshotting to SRCNN_iter_60012
I0830 23:03:04.825417   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_60012.solverstate
I0830 23:03:04.826032   796 solver.cpp:195] Iteration 60012, loss = 0.361884
I0830 23:03:04.826072   796 solver.cpp:365] Iteration 60012, lr = 1e-05
I0830 23:03:21.758008   796 solver.cpp:232] Iteration 80008, Testing net (#0)
I0830 23:03:22.536635   796 solver.cpp:270] Test score #0: 0.121355
I0830 23:03:22.543200   796 solver.cpp:287] Snapshotting to SRCNN_iter_80016
I0830 23:03:22.543598   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_80016.solverstate
I0830 23:03:22.544186   796 solver.cpp:195] Iteration 80016, loss = 0.328423
I0830 23:03:22.544211   796 solver.cpp:365] Iteration 80016, lr = 1e-05
I0830 23:03:39.410823   796 solver.cpp:232] Iteration 100010, Testing net (#0)
I0830 23:03:40.187535   796 solver.cpp:270] Test score #0: 0.115613
I0830 23:03:40.196040   796 solver.cpp:287] Snapshotting to SRCNN_iter_100020
I0830 23:03:40.196393   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_100020.solverstate
I0830 23:03:40.196987   796 solver.cpp:195] Iteration 100020, loss = 0.309899
I0830 23:03:40.197016   796 solver.cpp:365] Iteration 100020, lr = 1e-05
I0830 23:03:57.118890   796 solver.cpp:232] Iteration 120012, Testing net (#0)
I0830 23:03:57.898970   796 solver.cpp:270] Test score #0: 0.111475
I0830 23:03:57.909205   796 solver.cpp:287] Snapshotting to SRCNN_iter_120024
I0830 23:03:57.909582   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_120024.solverstate
I0830 23:03:57.910202   796 solver.cpp:195] Iteration 120024, loss = 0.298778
I0830 23:03:57.910228   796 solver.cpp:365] Iteration 120024, lr = 1e-05
I0830 23:04:14.830847   796 solver.cpp:232] Iteration 140014, Testing net (#0)
I0830 23:04:15.615363   796 solver.cpp:270] Test score #0: 0.107606
I0830 23:04:15.626828   796 solver.cpp:287] Snapshotting to SRCNN_iter_140028
I0830 23:04:15.627166   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_140028.solverstate
I0830 23:04:15.627758   796 solver.cpp:195] Iteration 140028, loss = 0.290971
I0830 23:04:15.627780   796 solver.cpp:365] Iteration 140028, lr = 1e-05
I0830 23:04:32.654047   796 solver.cpp:232] Iteration 160016, Testing net (#0)
I0830 23:04:33.431738   796 solver.cpp:270] Test score #0: 0.104881
I0830 23:04:33.445960   796 solver.cpp:287] Snapshotting to SRCNN_iter_160032
I0830 23:04:33.446285   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_160032.solverstate
I0830 23:04:33.446825   796 solver.cpp:195] Iteration 160032, loss = 0.285247
I0830 23:04:33.446849   796 solver.cpp:365] Iteration 160032, lr = 1e-05
I0830 23:04:50.394443   796 solver.cpp:232] Iteration 180018, Testing net (#0)
I0830 23:04:51.176180   796 solver.cpp:270] Test score #0: 0.103053
I0830 23:04:51.191655   796 solver.cpp:287] Snapshotting to SRCNN_iter_180036
I0830 23:04:51.191973   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_180036.solverstate
I0830 23:04:51.192560   796 solver.cpp:195] Iteration 180036, loss = 0.280845
I0830 23:04:51.192582   796 solver.cpp:365] Iteration 180036, lr = 1e-05
I0830 23:05:08.135113   796 solver.cpp:232] Iteration 200020, Testing net (#0)
I0830 23:05:08.926317   796 solver.cpp:270] Test score #0: 0.101469
I0830 23:05:08.944000   796 solver.cpp:287] Snapshotting to SRCNN_iter_200040
I0830 23:05:08.944427   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_200040.solverstate
I0830 23:05:08.945114   796 solver.cpp:195] Iteration 200040, loss = 0.277326
I0830 23:05:08.945152   796 solver.cpp:365] Iteration 200040, lr = 1e-05
I0830 23:05:25.926376   796 solver.cpp:232] Iteration 220022, Testing net (#0)
I0830 23:05:26.706737   796 solver.cpp:270] Test score #0: 0.100722
I0830 23:05:26.725574   796 solver.cpp:287] Snapshotting to SRCNN_iter_220044
I0830 23:05:26.725883   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_220044.solverstate
I0830 23:05:26.726430   796 solver.cpp:195] Iteration 220044, loss = 0.274449
I0830 23:05:26.726454   796 solver.cpp:365] Iteration 220044, lr = 1e-05
I0830 23:05:43.644079   796 solver.cpp:232] Iteration 240024, Testing net (#0)
I0830 23:05:44.424046   796 solver.cpp:270] Test score #0: 0.0989064
I0830 23:05:44.444973   796 solver.cpp:287] Snapshotting to SRCNN_iter_240048
I0830 23:05:44.445240   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_240048.solverstate
I0830 23:05:44.445792   796 solver.cpp:195] Iteration 240048, loss = 0.271982
I0830 23:05:44.445816   796 solver.cpp:365] Iteration 240048, lr = 1e-05
I0830 23:06:01.358248   796 solver.cpp:232] Iteration 260026, Testing net (#0)
I0830 23:06:02.135788   796 solver.cpp:270] Test score #0: 0.0977695
I0830 23:06:02.158663   796 solver.cpp:287] Snapshotting to SRCNN_iter_260052
I0830 23:06:02.158993   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_260052.solverstate
I0830 23:06:02.159603   796 solver.cpp:195] Iteration 260052, loss = 0.269841
I0830 23:06:02.159629   796 solver.cpp:365] Iteration 260052, lr = 1e-05
I0830 23:06:19.036283   796 solver.cpp:232] Iteration 280028, Testing net (#0)
I0830 23:06:19.880151   796 solver.cpp:270] Test score #0: 0.0978085
I0830 23:06:19.904516   796 solver.cpp:287] Snapshotting to SRCNN_iter_280056
I0830 23:06:19.904867   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_280056.solverstate
I0830 23:06:19.905436   796 solver.cpp:195] Iteration 280056, loss = 0.267913
I0830 23:06:19.905462   796 solver.cpp:365] Iteration 280056, lr = 1e-05
I0830 23:06:36.781340   796 solver.cpp:232] Iteration 300030, Testing net (#0)
I0830 23:06:37.560071   796 solver.cpp:270] Test score #0: 0.0977241
I0830 23:06:37.585538   796 solver.cpp:287] Snapshotting to SRCNN_iter_300060
I0830 23:06:37.585863   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_300060.solverstate
I0830 23:06:37.586901   796 solver.cpp:195] Iteration 300060, loss = 0.266174
I0830 23:06:37.586925   796 solver.cpp:365] Iteration 300060, lr = 1e-05
I0830 23:06:54.512564   796 solver.cpp:232] Iteration 320032, Testing net (#0)
I0830 23:06:55.294209   796 solver.cpp:270] Test score #0: 0.0968464
I0830 23:06:55.322038   796 solver.cpp:287] Snapshotting to SRCNN_iter_320064
I0830 23:06:55.322463   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_320064.solverstate
I0830 23:06:55.323237   796 solver.cpp:195] Iteration 320064, loss = 0.264589
I0830 23:06:55.323262   796 solver.cpp:365] Iteration 320064, lr = 1e-05
I0830 23:07:12.194263   796 solver.cpp:232] Iteration 340034, Testing net (#0)
I0830 23:07:12.991510   796 solver.cpp:270] Test score #0: 0.094625
I0830 23:07:13.021288   796 solver.cpp:287] Snapshotting to SRCNN_iter_340068
I0830 23:07:13.021567   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_340068.solverstate
I0830 23:07:13.022100   796 solver.cpp:195] Iteration 340068, loss = 0.263111
I0830 23:07:13.022124   796 solver.cpp:365] Iteration 340068, lr = 1e-05
I0830 23:07:30.003763   796 solver.cpp:232] Iteration 360036, Testing net (#0)
I0830 23:07:30.779625   796 solver.cpp:270] Test score #0: 0.0940636
I0830 23:07:30.811125   796 solver.cpp:287] Snapshotting to SRCNN_iter_360072
I0830 23:07:30.811430   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_360072.solverstate
I0830 23:07:30.811996   796 solver.cpp:195] Iteration 360072, loss = 0.261755
I0830 23:07:30.812029   796 solver.cpp:365] Iteration 360072, lr = 1e-05
I0830 23:07:47.708026   796 solver.cpp:232] Iteration 380038, Testing net (#0)
I0830 23:07:48.483667   796 solver.cpp:270] Test score #0: 0.094565
I0830 23:07:48.517200   796 solver.cpp:287] Snapshotting to SRCNN_iter_380076
I0830 23:07:48.517474   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_380076.solverstate
I0830 23:07:48.517997   796 solver.cpp:195] Iteration 380076, loss = 0.260513
I0830 23:07:48.518021   796 solver.cpp:365] Iteration 380076, lr = 1e-05
I0830 23:08:05.448905   796 solver.cpp:232] Iteration 400040, Testing net (#0)
I0830 23:08:06.236060   796 solver.cpp:270] Test score #0: 0.0936951
I0830 23:08:06.270323   796 solver.cpp:287] Snapshotting to SRCNN_iter_400080
I0830 23:08:06.270596   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_400080.solverstate
I0830 23:08:06.271121   796 solver.cpp:195] Iteration 400080, loss = 0.259338
I0830 23:08:06.271145   796 solver.cpp:365] Iteration 400080, lr = 1e-05
I0830 23:08:23.333593   796 solver.cpp:232] Iteration 420042, Testing net (#0)
I0830 23:08:24.121047   796 solver.cpp:270] Test score #0: 0.0923165
I0830 23:08:24.157490   796 solver.cpp:287] Snapshotting to SRCNN_iter_420084
I0830 23:08:24.157830   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_420084.solverstate
I0830 23:08:24.158382   796 solver.cpp:195] Iteration 420084, loss = 0.258207
I0830 23:08:24.158409   796 solver.cpp:365] Iteration 420084, lr = 1e-05
"""

m = re.findall('Iteration (\d+), loss = ([\d.]+)', txt)
data = []
x = []
for i in range(len(m)):
    data.append(float(m[i][1]))
    x.append(float(m[i][0]))
    
plt.plot(x, data)
plt.show()