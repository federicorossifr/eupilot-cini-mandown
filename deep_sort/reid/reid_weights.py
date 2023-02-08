
import gdown
from os.path import exists as file_exists

__model_types = ['mlfn', 'hacnn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0', 'osnet_ain_x1_0']

__trained_urls = {

    'mlfn_market1501.pt':
    'https://drive.google.com/uc?id=1wXcvhA_b1kpDfrt9s2Pma-MHxtj9pmvS',
    'mlfn_dukemtmcreid.pt':
    'https://drive.google.com/uc?id=1rExgrTNb0VCIcOnXfMsbwSUW1h2L1Bum',
    'mlfn_msmt17.pt':
    'https://drive.google.com/uc?id=18JzsZlJb3Wm7irCbZbZ07TN4IFKvR6p-',

    'hacnn_market1501.pt':
    'https://drive.google.com/uc?id=1LRKIQduThwGxMDQMiVkTScBwR7WidmYF',
    'hacnn_dukemtmcreid.pt':
    'https://drive.google.com/uc?id=1zNm6tP4ozFUCUQ7Sv1Z98EAJWXJEhtYH',
    'hacnn_msmt17.pt':
    'https://drive.google.com/uc?id=1MsKRtPM5WJ3_Tk2xC0aGOO7pM3VaFDNZ',

    'osnet_x1_0_market1501.pt':
    'https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA',
    'osnet_x1_0_dukemtmcreid.pt':
    'https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq',
    'osnet_x1_0_msmt17.pt':
    'https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M',

    'osnet_x0_75_market1501.pt':
    'https://drive.google.com/uc?id=1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer',
    'osnet_x0_75_dukemtmcreid.pt':
    'https://drive.google.com/uc?id=1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or',
    'osnet_x0_75_msmt17.pt':
    'https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc',

    'osnet_x0_5_market1501.pt':
    'https://drive.google.com/uc?id=1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT',
    'osnet_x0_5_dukemtmcreid.pt':
    'https://drive.google.com/uc?id=1KoUVqmiST175hnkALg9XuTi1oYpqcyTu',
    'osnet_x0_5_msmt17.pt':
    'https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv',

    'osnet_x0_25_market1501.pt':
    'https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj',
    'osnet_x0_25_dukemtmcreid.pt':
    'https://drive.google.com/uc?id=1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l',
    'osnet_x0_25_msmt17.pt':
    'https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF',

    # Others:
    'osnet_x1_0_msmt17.pt':
    'https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x',
    'osnet_x0_75_msmt17.pt':
    'https://drive.google.com/uc?id=1fhjSS_7SUGCioIf2SWXaRGPqIY9j7-uw',
    'osnet_x0_5_msmt17.pt':
    'https://drive.google.com/uc?id=1DHgmb6XV4fwG3n-CnCM0zdL9nMsZ9_RF',
    'osnet_x0_25_msmt17.pt':
    'https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e',
    'osnet_ibn_x1_0_msmt17.pt':
    'https://drive.google.com/uc?id=1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ',
    'osnet_ain_x1_0_msmt17.pt':
    'https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal',
}

model_name = 'osnet_ain_x1_0_msmt17.pt'
model_url = 'https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal'
gdown.download(model_url, str(model_name), quiet = False)
