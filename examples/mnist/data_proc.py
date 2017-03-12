import os
import struct
import numpy as np
from array import array

def read_mnist():
    def mnist_file_name(filename):
        return os.path.join(os.path.dirname(__file__), 'datas', filename)

    # Read train set labels into numpy (shapes: 60000 x 1)
    with open(mnist_file_name('train-labels-idx1-ubyte'), 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        train_labels = np.array(array('B', f.read()), dtype=np.uint8)

    # Read train set images into numpy (shapes: 60000 x 784)
    with open(mnist_file_name('train-images-idx3-ubyte'), 'rb') as f:
        magic, n, row, col = struct.unpack(">IIII", f.read(16))
        sz = row * col
        train_images = np.array(array('B', f.read()), dtype=np.uint8).reshape((n, sz))

    # Read test set labels into numpy (shapes: 10000 x 1)
    with open(mnist_file_name('t10k-labels-idx1-ubyte'), 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        test_labels = np.array(array('B', f.read()), dtype=np.uint8)

    # Read test set images into numpy (shapes: 10000 x 784)
    with open(mnist_file_name('t10k-images-idx3-ubyte'), 'rb') as f:
        magic, n, row, col = struct.unpack(">IIII", f.read(16))
        sz = row * col
        test_images = np.array(array('B', f.read()), dtype=np.uint8).reshape((n, sz))

    return train_images, train_labels, test_images, test_labels


def show_one_example(one_x, one_y):
    from matplotlib import pyplot as plt
    print(one_y)
    img = one_x.reshape((28, 28))
    plt.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    plt.show()


def encode_col(col):
    ret = 0
    val = 1
    for i in range(6):
        if col[i*5:(i+1)*5].sum() > 200:
            ret += val
        val *= 2
    return ret


# Parameter of HMM
# Emission probability is come from statistic_model.py
N = 41
M = 64


A = np.full((N, N), 0, dtype=np.float64)
A[N-1] = np.ones((N)) * 1/N
for i in range(10):
    A[i, i] = 0.75
    A[i, 10+i] = 0.25
    A[10+i, 10+i] = 0.75
    A[10+i, 20+i] = 0.25
    A[20+i, 20+i] = 0.75
    A[20+i, 30+i] = 0.25
    A[30+i, 30+i] = 0.75
    A[30+i, N-1] = 0.25


B = (
        (
            0.80294734811,  4.82381032778e-05,  0.000675333445889,  0.0,  0.00414847688189,  0.0,  0.00113359542703,  0.0,  0.0369503871108,  0.0,  0.000144714309833,  0.0,  0.0430525071754,  0.0,  0.00513735799908,  0.0,  0.00622271532283,  0.0,  2.41190516389e-05,  0.0,  0.000241190516389,  0.0,  0.0,  0.0,  0.0613347483177,  2.41190516389e-05,  7.23571549167e-05,  0.0,  0.0364438870264,  0.0,  0.00139890499506,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.991270076705,  0.0,  0.000190702207908,  4.2378268424e-05,  0.000148323939484,  0.0,  8.47565368479e-05,  0.0,  0.000169513073696,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.00775522312158,  0.0,  0.00010594567106,  0.0,  2.1189134212e-05,  0.0,  0.0,  0.0,  0.000169513073696,  0.0,  0.0,  0.0,  0.0,  0.0,  2.1189134212e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.1189134212e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.797678991032,  0.000311705749772,  0.0256797583082,  0.00122284563372,  0.00577854505347,  0.0,  0.00565865822663,  0.0,  0.0381240109337,  9.59094614684e-05,  0.00184625713327,  2.39773653671e-05,  0.00309308013236,  0.0,  0.000311705749772,  0.0,  0.0330647868412,  7.19320961013e-05,  0.00222989497914,  9.59094614684e-05,  0.000647388864912,  0.0,  0.000647388864912,  0.0,  0.0783340526543,  9.59094614684e-05,  0.00234978180598,  7.19320961013e-05,  0.00208603078694,  0.0,  0.000455569941975,  0.0,  2.39773653671e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.787426893772,  0.000116503949484,  0.0458093529371,  0.000675722907007,  0.00442715008039,  0.0,  0.00645431880141,  2.33007898968e-05,  0.0182678192791,  0.0,  0.00158445371298,  0.000163105529277,  0.00079222685649,  0.0,  0.000302910268658,  0.0,  0.0810634480509,  0.000139804739381,  0.00824847962346,  6.99023696903e-05,  0.0007223244868,  0.0,  0.000955332385768,  2.33007898968e-05,  0.0385628072792,  0.0,  0.00246988372906,  9.32031595871e-05,  0.000466015797936,  0.0,  0.000326211058555,  0.0,  0.000163105529277,  0.0,  9.32031595871e-05,  0.0,  2.33007898968e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.000372812638348,  0.0,  0.000139804739381,  0.0,  0.0,  0.0,  2.33007898968e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.884506284541,  0.0,  0.00273878808627,  2.4453465056e-05,  0.0186824473028,  0.0,  0.00819191079376,  7.3360395168e-05,  0.0427691103829,  0.0,  9.7813860224e-05,  0.0,  0.0330855382208,  0.0,  0.00748276030714,  2.4453465056e-05,  0.000758057416736,  0.0,  0.0,  0.0,  9.7813860224e-05,  0.0,  0.0,  0.0,  0.00112485939258,  0.0,  0.0,  0.0,  0.000146720790336,  0.0,  7.3360395168e-05,  0.0,  7.3360395168e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.8906930112e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.871452288718,  0.0,  0.00216090863573,  0.0,  0.0108572482673,  0.0,  0.0049015732469,  5.27050886763e-05,  0.0353387619575,  0.0,  0.000237172899043,  2.63525443382e-05,  0.00339947821962,  0.0,  0.000869633963159,  0.0,  0.0417951353203,  0.0,  0.000368935620734,  0.0,  0.00184467810367,  0.0,  0.000263525443382,  0.0,  0.0244815136901,  0.0,  0.000158115266029,  0.0,  0.00123856958389,  0.0,  0.000368935620734,  0.0,  7.90576330145e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.63525443382e-05,  0.0,  0.0,  0.0,  2.63525443382e-05,  0.0,  0.0,  0.0,  5.27050886763e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.963646019408,  0.000144836576063,  0.000313812581471,  0.000144836576063,  0.00321054410274,  0.0,  0.00140008690195,  0.000386230869502,  0.013831893014,  0.0,  7.24182880317e-05,  0.0,  0.0102351180418,  0.0,  0.00202771206489,  0.000144836576063,  0.00043450972819,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.00306570752667,  0.0,  4.82788586878e-05,  2.41394293439e-05,  0.00086901945638,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.82439858625,  0.0,  0.0549082202713,  0.0,  0.0559115266218,  0.0,  0.0416828183787,  0.0,  0.00417284232129,  0.0,  0.000342036255843,  0.0,  0.00773001938205,  0.0,  0.00171018127922,  0.0,  0.00314673355376,  0.0,  0.00018241933645,  0.0,  0.000159616919393,  0.0,  2.28024170562e-05,  0.0,  0.000205221753506,  0.0,  0.0,  0.0,  0.000159616919393,  0.0,  0.0,  0.0,  0.00207501995211,  0.0,  0.00018241933645,  0.0,  0.000342036255843,  0.0,  0.000159616919393,  0.0,  0.0,  0.0,  0.0,  0.0,  4.56048341124e-05,  0.0,  0.0,  0.0,  0.00191540303272,  0.0,  0.00018241933645,  0.0,  0.000342036255843,  0.0,  2.28024170562e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.941743780062,  0.0,  0.0041995263325,  0.0,  0.00793515150035,  0.0,  0.00944893424811,  0.0,  0.00271015943551,  0.0,  4.88317015406e-05,  0.0,  0.00112312913543,  0.0,  0.000292990209244,  0.0,  0.0200209976317,  0.0,  0.000170910955392,  0.0,  0.000341821910784,  0.0,  0.000170910955392,  0.0,  0.0109383011451,  0.0,  0.000122079253852,  0.0,  0.000439485313866,  0.0,  0.000195326806163,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  9.76634030813e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.940326105228,  0.0,  0.000216122757726,  0.0,  0.0238215306294,  0.0,  0.00441850971352,  0.0,  0.00345796412362,  0.0,  0.0,  0.0,  0.0215162212136,  0.0,  0.00223326849651,  0.0,  0.0014168047451,  0.0,  0.0,  0.0,  9.60545589895e-05,  0.0,  0.0,  0.0,  0.000408231875705,  0.0,  0.0,  0.0,  0.000336190956463,  0.0,  7.20409192421e-05,  0.0,  0.000936531950148,  0.0,  0.0,  0.0,  2.40136397474e-05,  0.0,  0.0,  0.0,  4.80272794948e-05,  0.0,  0.0,  0.0,  2.40136397474e-05,  0.0,  0.0,  0.0,  0.000624354633432,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.40136397474e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.0246255517233,  0.0,  0.00135066689178,  2.41190516389e-05,  0.00238778611225,  2.41190516389e-05,  0.00248426231881,  0.0,  0.0123730734908,  2.41190516389e-05,  0.00352138153928,  9.64762065556e-05,  0.0289669810183,  4.82381032778e-05,  0.0204770748414,  0.000916523962278,  0.00537854851547,  0.00127830973686,  0.158631002629,  0.0175345505415,  0.0773497986059,  9.64762065556e-05,  0.192518270182,  0.00299076240322,  0.0446926026869,  0.000168833361472,  0.0236607896578,  0.00328019102289,  0.220399893876,  0.000120595258194,  0.15084054895,  0.0030390005065,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.82381032778e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  7.23571549167e-05,  0.0,  0.0,  0.0,  0.00021707146475,  0.0,  0.0,  0.0,  4.82381032778e-05,  0.0,  9.64762065556e-05,  0.0,  0.00021707146475,  0.0
        ),
        (
            0.485188795186,  0.00010594567106,  0.013794126372,  0.00245793956859,  0.00434377251346,  0.0,  0.0118023477561,  0.00233080476332,  0.0308937576811,  0.0,  0.00031783701318,  4.2378268424e-05,  0.0403229224054,  0.0,  0.0122896978429,  0.00305123532652,  0.109505445607,  0.000148323939484,  0.00468279866085,  0.000444971818451,  0.00258507437386,  0.0,  0.00241556130017,  0.000339026147392,  0.168940967072,  0.00010594567106,  0.00286053311862,  0.000275458744756,  0.0612154087384,  6.35674026359e-05,  0.0325041318812,  0.00620841632411,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.000148323939484,  0.0,  2.1189134212e-05,  0.0,  8.47565368479e-05,  0.0,  0.0,  0.0,  8.47565368479e-05,  0.0,  8.47565368479e-05,  0.0,  0.000127134805272,  0.0,  0.00021189134212,  0.0
        ),
        (
            0.0255119167506,  0.00117489090299,  0.0360140027814,  0.00381240109337,  0.00338080851676,  0.0,  0.00786457584041,  0.0,  0.0446698316789,  0.00683354912962,  0.0670407135664,  0.0117249316645,  0.0266388529228,  0.00244569126744,  0.0420562988539,  0.00481945043879,  0.0178631371985,  0.00122284563372,  0.0313863712655,  0.00340478588213,  0.00633002445691,  0.00251762336354,  0.0122044789719,  0.00258955545965,  0.16525200211,  0.0290845441903,  0.22171869755,  0.0734426701194,  0.031026710785,  0.00613820553398,  0.0840406656117,  0.0275020380761,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.39773653671e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  7.19320961013e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  2.39773653671e-05,  0.0,  0.0,  0.0,  0.00016784155757,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.0389123191276,  0.000419414218142,  0.0569471305077,  0.00198056714123,  0.00277279399772,  0.000116503949484,  0.0226716685696,  0.00109513712515,  0.00463685718946,  4.66015797936e-05,  0.0116969965282,  0.000349511848452,  0.00202716872102,  9.32031595871e-05,  0.0161008458187,  0.000489316587832,  0.116900062912,  0.00293589952699,  0.124402917259,  0.00899410490016,  0.032737609805,  0.00932031595871,  0.223081762472,  0.0416618123354,  0.0402404641517,  0.000535918167626,  0.0362327282895,  0.00274949320782,  0.0217629377636,  0.00177086003216,  0.142530931799,  0.0161474473985,  0.000139804739381,  0.0,  0.00125824265443,  0.0,  0.000233007898968,  0.0,  0.0014446489736,  0.0,  0.000233007898968,  0.0,  0.000908730805974,  0.0,  2.33007898968e-05,  0.0,  0.00102523475546,  0.0,  0.00065242211711,  0.0,  0.00209707109071,  0.0,  0.000163105529277,  0.0,  0.00295920031689,  0.0,  0.000116503949484,  0.0,  0.0014679497635,  0.0,  2.33007898968e-05,  0.0,  0.00489316587832,  0.0
        ),
        (
            0.0644348804226,  0.0,  0.00166283562381,  2.4453465056e-05,  0.0657798210006,  0.0,  0.0451900034235,  0.000195627720448,  0.141781190395,  0.000220081185504,  0.0418398787108,  0.00212745145987,  0.261285274123,  0.0,  0.180808920624,  0.00254316036582,  0.0024453465056,  0.0,  0.00012226732528,  0.0,  0.00931677018634,  0.0,  0.00677360982051,  0.0,  0.0268009977014,  2.4453465056e-05,  0.0156013107057,  0.000220081185504,  0.0613537438255,  0.0,  0.0557294468626,  0.000317895045728,  4.8906930112e-05,  0.0,  0.0,  0.0,  0.00012226732528,  0.0,  7.3360395168e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.000220081185504,  0.0,  0.000195627720448,  0.0,  0.000268988115616,  0.0,  0.0,  0.0,  0.00166283562381,  0.0,  0.00158947522864,  0.0,  0.000758057416736,  0.0,  0.00097813860224,  0.0,  0.00325231085245,  0.0,  0.00423044945469,  0.0
        ),
        (
            0.0517563970801,  0.0,  0.00324136295359,  5.27050886763e-05,  0.0189738319235,  0.0,  0.0288033309616,  0.000500698342425,  0.023769994993,  0.0,  0.00397923419506,  5.27050886763e-05,  0.0367090942631,  5.27050886763e-05,  0.0258518459957,  0.000263525443382,  0.0986112209134,  0.0,  0.00614014283079,  0.000263525443382,  0.100166021029,  0.00115951195088,  0.245131367434,  0.00756318022505,  0.053074024297,  7.90576330145e-05,  0.0164176351227,  0.000342583076396,  0.112525364324,  5.27050886763e-05,  0.141407752919,  0.00384747147337,  0.000316230532058,  0.0,  5.27050886763e-05,  0.0,  0.000447993253749,  0.0,  0.00158115266029,  0.0,  7.90576330145e-05,  0.0,  0.000685166152792,  0.0,  0.000632461064116,  0.0,  0.00118586449522,  0.0,  0.00102774922919,  0.0,  5.27050886763e-05,  0.0,  0.00123856958389,  0.0,  0.00463804780352,  0.0,  0.000158115266029,  0.0,  0.000922339051835,  0.0,  0.00137033230558,  0.0,  0.00482251561388,  0.0
        ),
        (
            0.156230386714,  0.000168976005407,  0.000651764592285,  0.000241394293439,  0.00704871336842,  2.41394293439e-05,  0.00277603437455,  0.00123111089654,  0.0618693574084,  0.00444165499928,  0.0122386906774,  0.00837638198233,  0.10008207406,  0.000820740597692,  0.0643557186308,  0.015304398204,  0.004562352146,  0.00238980350505,  0.00917298315068,  0.00470718872206,  0.0135663592913,  0.000313812581471,  0.0206150726597,  0.00164148119538,  0.0490271809974,  0.00789359339545,  0.0371264423309,  0.0283396900497,  0.188504803746,  0.00277603437455,  0.166006855598,  0.0274706705933,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.41394293439e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.0583741876639,  0.0,  0.179044578725,  0.0,  0.0960209782237,  0.0,  0.166799680766,  0.0,  0.00367118914605,  0.0,  0.0363242503705,  0.0,  0.0172614297115,  0.0,  0.0241705620796,  0.0,  0.0100330635047,  0.0,  0.0484779386615,  0.0,  0.0178086877209,  0.0,  0.0307376581918,  0.0,  0.00289590696614,  0.0,  0.0562079580436,  0.000114012085281,  0.00700034203626,  0.0,  0.0346368715084,  0.0,  0.00280469729791,  0.0,  0.0107627408505,  0.0,  0.00804925322084,  0.0,  0.0110591722723,  0.0,  2.28024170562e-05,  0.0,  0.000364838672899,  0.0,  0.000592862843461,  0.0,  0.000273629004674,  0.0,  0.00832288222552,  0.0,  0.0610192680424,  0.0,  0.0196100786683,  0.0,  0.0615437236347,  0.0,  0.00018241933645,  0.0,  0.0143655227454,  0.0,  0.00159616919393,  0.0,  0.00985064416828,  0.0
        ),
        (
            0.120174817492,  0.0,  0.0130136484606,  0.000195326806163,  0.0190199477501,  9.76634030813e-05,  0.0688038674708,  0.000415069463095,  0.00642136875259,  0.0,  0.00527382376639,  4.88317015406e-05,  0.0071538442757,  9.76634030813e-05,  0.024025197158,  0.000585980418488,  0.0449739971189,  0.0,  0.00292990209244,  2.44158507703e-05,  0.00881412212809,  0.000390653612325,  0.0392362721879,  0.00192885221086,  0.131039871084,  7.3247552311e-05,  0.0385526283663,  0.000585980418488,  0.0600141611934,  0.00102546573235,  0.381863906048,  0.00859437947115,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  7.3247552311e-05,  0.0,  4.88317015406e-05,  0.0,  0.000292990209244,  0.0,  0.000146495104622,  0.0,  0.000439485313866,  0.0,  0.000512732866177,  0.0,  2.44158507703e-05,  0.0,  0.000219742656933,  0.0,  0.00078130722465,  0.0,  0.00100104988158,  0.0,  0.00119637668775,  0.0,  0.00102546573235,  0.0,  0.00886295382963,  0.0
        ),
        (
            0.116730302812,  0.0,  0.00242537761449,  0.0,  0.0834954254016,  0.0,  0.0857767211776,  0.0,  0.00453857791225,  0.0,  0.134428355306,  0.0,  0.138438633144,  0.0,  0.168383641909,  2.40136397474e-05,  0.00823667843335,  0.0,  0.00122469562712,  0.0,  0.0104939605696,  2.40136397474e-05,  0.0292245995726,  0.000216122757726,  0.000768436471916,  0.0,  0.0212280575367,  4.80272794948e-05,  0.0206757438225,  0.0,  0.0649328818769,  2.40136397474e-05,  0.00194510481954,  0.0,  7.20409192421e-05,  0.0,  0.00225728213625,  0.0,  0.00302571860817,  0.0,  9.60545589895e-05,  0.0,  0.00175299570156,  0.0,  0.00338592320438,  0.0,  0.00309775952741,  0.0,  0.00477871430973,  0.0,  0.000360204596211,  0.0,  0.00854885575007,  0.0,  0.0198592800711,  0.0,  0.000192109117979,  0.0,  0.0128953245443,  0.0,  0.014168047451,  0.0,  0.032226304541,  0.0
        ),
        (
            0.00530619136056,  2.41190516389e-05,  0.00279780999011,  0.000144714309833,  0.00294252429994,  4.82381032778e-05,  0.0105400255662,  0.00195364318275,  0.00311135766142,  0.00115771447867,  0.0948361110441,  0.0140614071055,  0.0177516220062,  0.000192952413111,  0.136562070379,  0.0186922650201,  0.00487204843106,  0.00335254817781,  0.246279636285,  0.0414606497673,  0.00438966739828,  4.82381032778e-05,  0.0654832251996,  0.00328019102289,  0.00284604809339,  0.00125419068522,  0.145293167073,  0.0334048865199,  0.012156002026,  0.000120595258194,  0.11610911459,  0.00940643013917,  0.0,  0.0,  2.41190516389e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  7.23571549167e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.41190516389e-05,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.35104038649,  0.000699241428995,  0.101623087681,  0.0364029325762,  0.0281179810993,  0.0,  0.142984277662,  0.021189134212,  0.00305123532652,  2.1189134212e-05,  0.000360215281604,  0.000254269610544,  0.0667033944993,  4.2378268424e-05,  0.0536296986905,  0.0074373861084,  0.022290969191,  0.00010594567106,  0.00292410052125,  0.000974700173751,  0.000656863160571,  0.0,  0.00362334195025,  0.00127134805272,  0.0173327117854,  0.000148323939484,  0.00161037420011,  0.00031783701318,  0.0215281603594,  0.000127134805272,  0.0906471161588,  0.02258761707,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.1189134212e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.000275458744756,  0.0
        ),
        (
            0.0123962978948,  0.0,  0.00347671797823,  0.000647388864912,  0.00412410684314,  0.000455569941975,  0.0165443821033,  0.00549081666906,  0.0680717402772,  0.0128758452021,  0.0683594686616,  0.0259914640579,  0.0291564762864,  0.0102383350117,  0.174651129334,  0.0512875845202,  0.0403539059128,  9.59094614684e-05,  0.0153694912003,  0.00268546492111,  0.00328489905529,  0.00383637845873,  0.0665132115283,  0.0233299765022,  0.0352467270896,  0.00851196470532,  0.0616697837242,  0.0297319330552,  0.010909701242,  0.0072171869755,  0.141610319858,  0.0650505922409,  4.79547307342e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.00016784155757,  0.0,  0.0,  0.0,  4.79547307342e-05,  0.0,  2.39773653671e-05,  0.0,  7.19320961013e-05,  0.0,  4.79547307342e-05,  0.0,  2.39773653671e-05,  0.0,  0.0,  0.0,  7.19320961013e-05,  0.0,  0.0,  0.0,  7.19320961013e-05,  0.0,  2.39773653671e-05,  0.0,  0.000215796288304,  0.0
        ),
        (
            0.0557121886432,  9.32031595871e-05,  0.0238134072745,  0.00156115292308,  0.00209707109071,  2.33007898968e-05,  0.0342987627281,  0.00316890742596,  0.0281939557751,  0.0,  0.0151688142228,  0.000862129226181,  0.0197124682527,  0.000302910268658,  0.0836498357294,  0.0138173684088,  0.00228347740988,  0.0,  0.000978633175665,  0.000116503949484,  0.0226483677797,  0.00342521611483,  0.173311275252,  0.0324580003262,  0.0553393760048,  4.66015797936e-05,  0.0245124309714,  0.000838828436284,  0.0491413658923,  0.00160775450288,  0.302840366288,  0.0407297807396,  0.0,  0.0,  0.0,  0.0,  4.66015797936e-05,  0.0,  0.000466015797936,  0.0,  9.32031595871e-05,  0.0,  0.000116503949484,  0.0,  4.66015797936e-05,  0.0,  0.000908730805974,  0.0,  6.99023696903e-05,  0.0,  2.33007898968e-05,  0.0,  0.000279609478761,  0.0,  0.000978633175665,  0.0,  0.000372812638348,  0.0,  0.000302910268658,  0.0,  0.000233007898968,  0.0,  0.00330871216534,  0.0
        ),
        (
            0.0473908152785,  0.000171174255392,  0.0252359759378,  0.00215190492493,  0.0592996527608,  4.8906930112e-05,  0.0782510881792,  0.00136939404314,  0.0984496503154,  0.0006113366264,  0.0279014036289,  0.00401036826918,  0.188609575977,  0.000146720790336,  0.131706362792,  0.00530640191715,  0.00535530884726,  2.4453465056e-05,  0.000782510881792,  2.4453465056e-05,  0.00503741380154,  0.0,  0.00276324155133,  7.3360395168e-05,  0.0723822565658,  0.000195627720448,  0.0123734533183,  0.000537976231232,  0.106910549225,  0.00012226732528,  0.10035702059,  0.00523304152198,  2.4453465056e-05,  0.0,  4.8906930112e-05,  0.0,  4.8906930112e-05,  0.0,  2.4453465056e-05,  0.0,  9.7813860224e-05,  0.0,  0.0,  0.0,  0.000146720790336,  0.0,  9.7813860224e-05,  0.0,  0.00036680197584,  0.0,  4.8906930112e-05,  0.0,  0.000464615836064,  0.0,  0.00012226732528,  0.0,  0.00256761383088,  0.0,  0.000758057416736,  0.0,  0.00699369100602,  0.0,  0.00535530884726,  0.0
        ),
        (
            0.0136506179672,  0.00105410177353,  0.161804622236,  0.00419005454977,  0.0293567343927,  5.27050886763e-05,  0.0302790734445,  7.90576330145e-05,  0.00758953276939,  0.000922339051835,  0.0558146889082,  0.00297783751021,  0.0246132764118,  0.00150209502727,  0.0641684454634,  0.00303054259889,  0.000368935620734,  0.000237172899043,  0.0096713837721,  0.00042164070941,  0.0139404959549,  0.00419005454977,  0.159907239044,  0.0148364824624,  0.0145466044747,  0.00250349171212,  0.144491000606,  0.00603473265344,  0.0239017577147,  0.00300419005455,  0.177299918307,  0.0149155400954,  0.0,  0.0,  5.27050886763e-05,  0.0,  0.0,  0.0,  0.00131762721691,  0.0,  2.63525443382e-05,  0.0,  0.000685166152792,  0.0,  2.63525443382e-05,  0.0,  0.000368935620734,  0.0,  0.0,  0.0,  0.000395288165072,  0.0,  7.90576330145e-05,  0.0,  0.00189738319235,  0.0,  5.27050886763e-05,  0.0,  0.0020027933697,  0.0,  5.27050886763e-05,  0.0,  0.00168656283764,  0.0
        ),
        (
            0.0282672717617,  0.00987302660165,  0.00697629508038,  0.00745908366726,  0.0035726355429,  0.000362091440158,  0.000748322309661,  0.000386230869502,  0.0359677497224,  0.0169941582581,  0.0155940713562,  0.0161975570898,  0.167696615652,  0.0386230869502,  0.035026311978,  0.0363298411626,  0.00212426978226,  0.0017621783421,  0.00253464008111,  0.00147250518998,  0.0319123255926,  0.0137111958673,  0.0195529377686,  0.0138801718727,  0.0366677931734,  0.0282431323324,  0.0495823878724,  0.0481098826824,  0.138174093564,  0.0385989475209,  0.0791049099599,  0.0744942789552,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.0747463231102,  0.0,  0.0324022346369,  0.0,  0.0363242503705,  0.0,  0.189260061567,  0.0,  0.0143199179113,  0.0,  0.0591494698438,  2.28024170562e-05,  0.0435982214115,  0.0,  0.253813704253,  0.0,  0.000912096682248,  0.0,  0.00816326530612,  0.0,  0.00492532208414,  0.0,  0.00503933416942,  0.0,  0.00141374985748,  0.0,  0.0626154372363,  2.28024170562e-05,  0.02029415118,  0.0,  0.0932618857599,  0.0,  0.000387641089956,  0.0,  0.00116292326987,  0.0,  0.000729677345799,  0.0,  0.00148215710865,  0.0,  2.28024170562e-05,  0.0,  0.000114012085281,  0.0,  0.000159616919393,  0.0,  0.00116292326987,  0.0,  0.000592862843461,  0.0,  0.0121308858739,  0.0,  0.00367118914605,  0.0,  0.00875612814958,  0.0,  0.000912096682248,  0.0,  0.0180823167256,  0.0,  0.00761600729677,  0.0,  0.0427317295633,  0.0
        ),
        (
            0.030617476866,  0.000146495104622,  0.0474399980467,  0.00349146666016,  0.0267597724443,  0.00148936689699,  0.217716141319,  0.011060380399,  0.00385770442171,  0.000122079253852,  0.0119149351759,  0.000292990209244,  0.00595746758796,  0.000415069463095,  0.0658495495276,  0.00412627878018,  0.00249041677857,  2.44158507703e-05,  0.00358913006324,  0.000268574358474,  0.00659227970799,  0.00146495104622,  0.0641648558244,  0.00737358693264,  0.0151866591791,  0.000292990209244,  0.0432160558635,  0.0019776839124,  0.0229264838733,  0.00280782283859,  0.360793026833,  0.0280538125351,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.000146495104622,  0.0,  2.44158507703e-05,  0.0,  0.000195326806163,  0.0,  0.0,  0.0,  0.000341821910784,  0.0,  7.3247552311e-05,  0.0,  4.88317015406e-05,  0.0,  7.3247552311e-05,  0.0,  0.000390653612325,  0.0,  7.3247552311e-05,  0.0,  0.000830138926191,  0.0,  0.000170910955392,  0.0,  0.00515174451254,  0.0
        ),
        (
            0.102274091684,  0.0,  0.020723771102,  0.0,  0.0420238695579,  0.0,  0.165261868741,  2.40136397474e-05,  0.00283360949019,  0.0,  0.046202242874,  2.40136397474e-05,  0.0216843166919,  0.0,  0.237927142617,  7.20409192421e-05,  0.00528300074442,  0.0,  0.00115265470787,  0.0,  0.00204115937853,  2.40136397474e-05,  0.0147443748049,  0.000288163676969,  0.00605143721634,  0.0,  0.0332348774104,  9.60545589895e-05,  0.0219724803688,  0.0,  0.171673510554,  0.000360204596211,  0.000696395552674,  0.0,  9.60545589895e-05,  0.0,  0.000312177316716,  0.0,  0.000864491030906,  0.0,  2.40136397474e-05,  0.0,  0.0004562591552,  0.0,  0.000336190956463,  0.0,  0.000984559229642,  0.0,  0.00285762312994,  0.0,  0.000408231875705,  0.0,  0.00144081838484,  0.0,  0.00696395552674,  0.0,  0.00237735033499,  0.0,  0.0113584516005,  0.0,  0.0121268880724,  0.0,  0.0627236270201,  0.0
        ),
        (
            0.660910253009,  0.000144714309833,  0.00918935867442,  0.000554738187694,  0.0372156966788,  0.0,  0.0515906514556,  0.00183304792456,  0.0102505969465,  0.0,  0.00586092954825,  4.82381032778e-05,  0.090036419768,  0.0,  0.101034707315,  0.00231542895733,  0.000144714309833,  0.0,  0.000458261981139,  0.0,  0.000964762065556,  0.0,  0.000699452497528,  0.0,  0.000844166807361,  0.0,  0.000964762065556,  0.0,  0.0141337642604,  0.0,  0.0107812160826,  2.41190516389e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.97796330042,  0.000381404415816,  0.0135610458957,  0.00493706827139,  0.00031783701318,  0.0,  0.000254269610544,  2.1189134212e-05,  0.000847565368479,  0.0,  0.0,  0.0,  0.00010594567106,  0.0,  2.1189134212e-05,  0.0,  0.000974700173751,  0.0,  2.1189134212e-05,  2.1189134212e-05,  2.1189134212e-05,  0.0,  0.0,  0.0,  0.000466160952663,  0.0,  8.47565368479e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.657267539443,  7.19320961013e-05,  0.0126360715485,  0.00136670982592,  0.0105020860308,  4.79547307342e-05,  0.015873015873,  0.0016064834796,  0.141610319858,  0.0,  0.00309308013236,  0.000431592576608,  0.00908742147413,  0.0,  0.00824821368628,  0.000479547307342,  0.0744017647341,  0.0,  0.00402819738167,  0.000119886826835,  0.00110295880689,  0.0,  0.00450774468901,  0.000287728384405,  0.0462763151585,  0.0,  0.00187023449863,  0.00016784155757,  0.00141466455666,  0.0,  0.00330887642066,  0.000119886826835,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  7.19320961013e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.887573688748,  6.99023696903e-05,  0.0263764941632,  0.000768926066594,  0.00137474660391,  0.0,  0.00922711279912,  6.99023696903e-05,  0.0330172192837,  0.0,  0.00246988372906,  9.32031595871e-05,  0.00815527646387,  0.0,  0.00293589952699,  0.000139804739381,  0.00177086003216,  0.0,  6.99023696903e-05,  2.33007898968e-05,  6.99023696903e-05,  0.0,  6.99023696903e-05,  0.0,  0.0198988745718,  0.0,  0.0007223244868,  0.0,  0.00424074376121,  0.0,  0.0007223244868,  6.99023696903e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  6.99023696903e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.798454541008,  0.00112485939258,  0.0722599892405,  0.00496405340637,  0.0236709541742,  2.4453465056e-05,  0.0309336332958,  0.000684697021568,  0.0288795422311,  0.000317895045728,  0.00638235437962,  0.0012226732528,  0.00919450286106,  4.8906930112e-05,  0.00635790091456,  0.000562429696288,  0.00604000586883,  0.0,  0.000268988115616,  7.3360395168e-05,  0.00036680197584,  0.0,  0.000464615836064,  0.0,  0.00315449699222,  0.0,  4.8906930112e-05,  2.4453465056e-05,  0.0022252653201,  0.0,  0.00185846334426,  9.7813860224e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.00012226732528,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  9.7813860224e-05,  0.0,  0.0,  0.0,  2.4453465056e-05,  0.0,  4.8906930112e-05,  0.0
        ),
        (
            0.603552322977,  0.00445357999315,  0.29006245553,  0.00930244815137,  0.0176562047066,  0.0,  0.0156007062482,  0.0,  0.0149418926397,  0.000474345798087,  0.00643002081851,  0.000658813608454,  0.00574485466572,  0.000158115266029,  0.00223996626874,  0.000316230532058,  0.00115951195088,  0.0,  0.000869633963159,  7.90576330145e-05,  0.000658813608454,  0.0,  0.000158115266029,  0.0,  0.0130708619917,  0.000158115266029,  0.00532321395631,  0.000395288165072,  0.00463804780352,  7.90576330145e-05,  0.0014493899386,  5.27050886763e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.63525443382e-05,  0.0,  0.0,  0.0,  0.000105410177353,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.000184467810367,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.793680297398,  0.0107179066287,  0.00485202529812,  0.004562352146,  0.0179838748612,  0.000362091440158,  0.00147250518998,  0.000144836576063,  0.0460097523295,  0.00127938975523,  0.00103799546179,  0.000844880027036,  0.100009655772,  0.00188287548882,  0.00251050065176,  0.000651764592285,  0.000289673152127,  2.41394293439e-05,  2.41394293439e-05,  0.0,  0.000362091440158,  0.0,  7.24182880317e-05,  0.0,  0.004562352146,  0.000265533722783,  0.000313812581471,  4.82788586878e-05,  0.00545551103172,  7.24182880317e-05,  0.000458649157534,  4.82788586878e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.891050051305,  0.0,  0.0285486261544,  0.0,  0.0201801390947,  0.0,  0.0380572340668,  2.28024170562e-05,  0.0122220955421,  0.0,  0.000615665260518,  0.0,  0.00524455592293,  0.0,  0.00191540303272,  0.0,  0.000410443507012,  0.0,  4.56048341124e-05,  0.0,  0.000159616919393,  0.0,  6.84072511686e-05,  0.0,  0.000319233838787,  0.0,  0.0,  0.0,  0.000250826587618,  0.0,  6.84072511686e-05,  0.0,  0.00018241933645,  0.0,  0.0,  0.0,  2.28024170562e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.000296431421731,  0.0,  0.0,  0.0,  2.28024170562e-05,  0.0,  0.0,  0.0,  9.12096682248e-05,  0.0,  0.0,  0.0,  0.000159616919393,  0.0,  4.56048341124e-05,  0.0
        ),
        (
            0.770149180848,  0.000488317015406,  0.0991283541275,  0.00234392167395,  0.0311302097322,  4.88317015406e-05,  0.0635300437044,  0.000659227970799,  0.00710501257416,  0.0,  0.00100104988158,  4.88317015406e-05,  0.00244158507703,  0.0,  0.0019776839124,  2.44158507703e-05,  0.00302756549552,  0.0,  0.000561564567717,  0.0,  0.000488317015406,  0.0,  0.000390653612325,  0.0,  0.00920477574041,  0.0,  0.00158703030007,  0.0,  0.00278340698782,  0.0,  0.00183118880777,  2.44158507703e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.44158507703e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        ),
        (
            0.930240376534,  0.0,  0.0139038974137,  0.0,  0.0162332204692,  0.0,  0.0199313209903,  0.0,  0.00108061378863,  0.0,  0.000120068198737,  0.0,  0.00268952765171,  0.0,  0.00160891386307,  0.0,  0.0055231371419,  0.0,  7.20409192421e-05,  0.0,  0.000408231875705,  0.0,  0.000192109117979,  0.0,  0.00264150037221,  0.0,  2.40136397474e-05,  0.0,  0.00153687294383,  0.0,  0.000264150037221,  0.0,  0.0004562591552,  0.0,  0.0,  0.0,  2.40136397474e-05,  0.0,  0.0,  0.0,  0.0,  0.0,  2.40136397474e-05,  0.0,  4.80272794948e-05,  0.0,  0.0,  0.0,  0.00177700934131,  0.0,  2.40136397474e-05,  0.0,  0.000168095478232,  0.0,  0.0,  0.0,  0.000432245515453,  0.0,  0.0,  0.0,  0.000408231875705,  0.0,  0.000168095478232,  0.0
        ),
        (1/M, ) * M
    )


pi = [0] * N
pi[N-1] = 1
