#!/usr/bin/env python3
"""positional_catalogue.py
James Gardner 2019
generates catalogue of positional matched NVSS and TGSS,
must run in directory containing CATALOG.FIT
(from https://heasarc.gsfc.nasa.gov/W3Browse/all/nvss.html)
and TGSSADR1_7sigma_catalog.tsv
(from http://tgssadr.strw.leidenuniv.nl/doku.php)
"""

import numpy as np
from astropy.io import fits
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

# 2' separation on sky
SEPARATION_LIMIT = 2*1/60

def generate_matches():
    """generates matches.py, which contains all
    close enough pairs from nvss, tgss
    NB: takes at least 30 mins to complete"""
    # pull ra, dec pairs from both surveys
    tgss = np.genfromtxt(
        fname="TGSSADR1_7sigma_catalog.tsv",
        delimiter='\t',
        skip_header=1,
        usecols=(1,3))

    with fits.open("CATALOG.FIT") as hdulist:
        data = hdulist[1].data
        nvss = np.column_stack((data['RA(2000)'],data['DEC(2000)']))

    tgss = tgss[tgss[:,1].argsort()]
    nvss = nvss[nvss[:,1].argsort()]

    # bin nvss
    bin_size = 0.1
    nvss_dec_min = round(nvss[:,1].min(),1)
    nvss_dec_max = round(nvss[:,1].max(),1)
    mark = nvss_dec_min - bin_size
    chunkis = []
    count = 0
    for i,dec in enumerate(nvss[:,1]):    
        if mark < dec < mark + bin_size:
            chunkis.append(i)
            mark += bin_size
    bins = [x/10 for x in range(int((nvss_dec_min-bin_size)*10),
                                int((nvss_dec_max+bin_size+0.01)*10))]
    cos_dec = np.array([np.cos(dec*np.pi/180) for dec in bins])

    matches = []
    tqdmbar = tqdm(total=len(tgss))
    for i1,p1 in enumerate(tgss):
        if p1[1] < nvss_dec_min - 0.1:
            tqdmbar.update(1)
            continue
        elif p1[1] > nvss_dec_max + 0.1:
            break

        which_bin = bins.index(np.floor(p1[1]*10)/10)
        nslice = chunkis[which_bin-1],chunkis[which_bin+2]

        for i2,p2 in enumerate(nvss[nslice[0]:nslice[1]]):
            if (abs((p1[0]-p2[0])*cos_dec[which_bin]) < SEPARATION_LIMIT
                    and abs(p1[1]-p2[1]) < SEPARATION_LIMIT):
                matches.append((p1,p2))
        tqdmbar.postfix = 'matches = {}'.format(len(matches))
        tqdmbar.update(1)
    matches = np.array(matches)
    np.save('matches.npy', matches)

def geodesic_dist(p1,p2):
    """arguments are two points on the unit sphere,
    with ra and dec given in radians;
    returns their geodesic distance, see:
    https://en.wikipedia.org/wiki/Great-circle_distance#Formulae"""
    ra1,dec1,ra2,dec2 = p1[0],p1[1],p2[0],p2[1]
    decdiff = (dec1-dec2)/2
    radiff  = (ra1-ra2)/2
    better_circle = 2*np.arcsin(np.sqrt(np.sin(decdiff)**2
                    + np.cos(dec1)*np.cos(dec2) * np.sin(radiff)**2))
    r = 1
    return better_circle*r

def degdist(p1,p2):
    """calls geodesic_dist on two points,
    with ra and dec given in degrees;
    returns their separation in degrees"""
    return 180/np.pi*geodesic_dist([x*np.pi/180 for x in p1],
                                   [x*np.pi/180 for x in p2])

def deci_deg_to_deg_min_sec(deci_deg):
    """https://stackoverflow.com/questions/2579535\
    /convert-dd-decimal-degrees-to-dms-degrees-minutes-seconds-in-python"""
    is_positive = (deci_deg >= 0)
    deci_deg = abs(deci_deg)
    # divmod returns quotient and remainder
    minutes,seconds = divmod(deci_deg*3600,60)
    degrees,minutes = divmod(minutes,60)
    degrees = degrees if is_positive else -degrees
    return (degrees,minutes,seconds)

def deci_deg_to_hr_min_sec(deci_deg):
    """assume deci_deg +ve"""
    deci_hours = deci_deg/15.
    schminutes,schmeconds = divmod(deci_hours*3600,60)
    hours,schminutes = divmod(schminutes,60)   
    return (hours,schminutes,schmeconds)

def iau_designation(ra,dec):
    """generate NVSS names as per:
    https://heasarc.gsfc.nasa.gov/W3Browse/all/nvss.html
    There are four cases where there are pairs of sources which are so close together that their names would be identical according to this schema (see below), and the HEASARC has added suffixes of 'a' (for the source with the smaller RA) and 'b' (for the source with the larger RA) in such cases in order to differentate them.
    It was easier just to hard-code this in,
    should really check if designation alreadys exists and compare
    """
    hr,schmin,schmec = deci_deg_to_hr_min_sec(ra)
    rhh = str(int(hr)).zfill(2)
    rmm = str(int(schmin)).zfill(2)
    rss = str(int(schmec - schmec%1)).zfill(2)

    deg,minu,sec = deci_deg_to_deg_min_sec(dec)
    sgn = '+' if deg>=0 else '-'
    ddd = str(int(abs(deg))).zfill(2)
    dmm = str(int(minu)).zfill(2)
    dss = str(int(sec - sec%1)).zfill(2)

    designation = ''.join(('NVSS J',rhh,rmm,rss,sgn,ddd,dmm,dss))

    close_pairs = {'NVSS J093731-102001':144.382,
                   'NVSS J133156-121336':202.987,
                   'NVSS J160612+000027':241.553,
                   'NVSS J215552+380029':328.968}
    if designation in close_pairs:
        if ra < close_pairs[designation]:
            designation = ''.join((designation,'a'))
        else:
            designation = ''.join((designation,'b'))          

    return designation

def generate_catalogue():
    """filters matches.npy by proper separation,
    constructs catalogue containing pair and separation,
    then adds uniqueness flag, names, fluxes, and spectral index;
    catalogue columns are: (T:TGSS, N:NVSS)
    Tname,Tra,Tdec,Nname,Nra,Ndec,sepdist,nonuniqueflag,Tflux,Nflux,alpha"""

    matches = np.load('matches.npy')
    catalogue = []
    for m in tqdm(matches):
        p1,p2 = m
        d = degdist(p1,p2)
        if d < SEPARATION_LIMIT:
            catalogue.append(('',p1[0],p1[1],'',p2[0],p2[1],d,0,0,0,0))
    catalogue = np.array(catalogue,dtype=object)

    for i,m in enumerate(tqdm(catalogue)):
        p1ra,p1dec,p2ra,p2dec = m[1],m[2],m[4],m[5]
        # checking only the nearest points for non-uniqueness
        nearby = 10
        if nearby < i < len(catalogue)-nearby:
            rest_wo = np.concatenate((catalogue[i-nearby:i],catalogue[i+1:i+1+nearby]))
        elif i < nearby:
            rest_wo = np.concatenate((catalogue[:i],catalogue[i+1:i+1+nearby]))
        elif len(catalogue)-nearby < i:
            rest_wo = np.concatenate((catalogue[i-nearby:i],catalogue[i+1:]))

        tgss_wo = rest_wo[:,(1,2)]
        nvss_wo = rest_wo[:,(4,5)]

        if (np.any((tgss_wo[:]==[p1ra,p1dec]).all(1)) or
            np.any((nvss_wo[:]==[p2ra,p2dec]).all(1))):
            catalogue[i][7] = 1

    tgss_labels = np.genfromtxt('TGSSADR1_7sigma_catalog.tsv',
                                delimiter='\t', skip_header=1, usecols=0, dtype=str)
    ftgss = np.genfromtxt('TGSSADR1_7sigma_catalog.tsv',
                          delimiter='\t', skip_header=1, usecols=(1,3,7))
    tgss_labels = tgss_labels[ftgss[:,1].argsort()]
    ftgss = ftgss[ftgss[:,1].argsort()]
    
    with fits.open("CATALOG.FIT") as hdulist:
        data = hdulist[1].data
        fnvss = np.column_stack((data['RA(2000)'],data['DEC(2000)'],data['PEAK INT']))
    fnvss = fnvss[fnvss[:,1].argsort()]    
    nvss_labels = np.array([iau_designation(p[0],p[1]) for p in tqdm(fnvss)])    

    freq_nvss, freq_tgss = 1.4e9,150e6
    # tgss in mJy/beam, nvss in Jy/beam; current not beam adjusted
    tgssbeam = 25#''
    ftgss[:,2] = ftgss[:,2]*1e-3
    nvssbeam = 45#''
    fnvss[:,2] = fnvss[:,2]

    for i,m in enumerate(tqdm(catalogue)):
        tdec,ndec = m[2],m[5]
        # searchsorted gives index value if equal to or next if not
        ti = np.searchsorted(ftgss[:,1],tdec)
        ni = np.searchsorted(fnvss[:,1],ndec)
        t_name,n_name = tgss_labels[ti],nvss_labels[ni]
        s_tgss,s_nvss = ftgss[ti][2],fnvss[ni][2]
        alpha = np.log(s_tgss/s_nvss)/np.log(freq_nvss/freq_tgss)

        catalogue[i][0] = t_name
        catalogue[i][3] = n_name
        catalogue[i][8] = s_tgss
        catalogue[i][9] = s_nvss
        catalogue[i][10] = alpha
    catalogue_fmt = ('%s','%1.5f','%1.5f','%s','%1.14f','%1.14f',
                     '%1.18f','%i','%1.5f','%1.5f','%1.5f')
    np.savetxt('catalogue.csv', catalogue, delimiter=',',fmt = catalogue_fmt)    

def generate_histograms():
    """generates histograms of angular separation and spectral index of matches,
    saves as hist_angle.pdf and hist_alpha.pdf"""
    plot_catalogue = np.loadtxt('catalogue.csv', delimiter=',', usecols=(6,8,10))
    seps   = plot_catalogue[:,0]*3600
    s_tgss = plot_catalogue[:,1]*1e3
    alpha  = plot_catalogue[:,2]
    allalpha = alpha,alpha[s_tgss>50],alpha[s_tgss>100],alpha[s_tgss>150],alpha[s_tgss>200]

    plt.figure(figsize=(14,7))
    plt.rcParams.update({'font.size': 18})
    plt.hist(seps, bins=100,color = "darkmagenta", ec="orchid")
    plt.xlabel("angular separation, '' (arcsec)")
    plt.ylabel('counts')
    plt.title('distribution of angular separation of matches')
    plt.savefig('hist_angle.pdf',bbox_inches='tight')

    fig,axis = plt.subplots(figsize=(14,7))
    plt.rcParams.update({'font.size': 18})
    plt.hist(allalpha, bins=200, histtype='step', stacked=False, fill=False,\
             label=['no cut','S_tgss>50mJy','S_tgss>100mJy','S_tgss>150mJy','S_tgss>200mJy'],\
             color=['black','limegreen','orange','magenta','darkblue'])
    plt.legend(loc='upper right')
    handles, labels = axis.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels))
    plt.xlim(0,2.5)
    plt.xlabel("observed spectral index")
    plt.ylabel('counts')
    plt.title('distribution of observed spectral index of matches')
    plt.savefig('hist_alpha.pdf',bbox_inches='tight')

def main(gen_matches_necessary=False):
    if gen_matches_necessary:
        generate_matches()
    generate_catalogue()
    generate_histograms()

if __name__ == "__main__":
    main()
