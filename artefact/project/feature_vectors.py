#!/usr/bin/env python3
"""James Gardner 2019
reads in TGSS and NVSS datapoints in a 5° patch of sky
and computes positional matches within 5',
saves feature vectors (individuals and pairs) as .csv
"""

import pandas as pd
import numpy as np
from astropy.io import fits
from tqdm import tqdm_notebook as tqdm

PATCH_SIZE = 5
SEPARATION_LIMIT = 5*1/60

# these five functions could be imported from positional_catalogue.py
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
    There are four cases where there are pairs of sources which are
    so close together that their names would be identical according
    to this schema (see below), and the HEASARC has added suffixes
    of 'a' (for the source with smaller RA) and 'b' (for the source
    with the larger RA) in such cases in order to differentate them.
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

def main():
    """this should really be broken up into separate functions"""
    PATCH_DEC = -35
    PATCH_RA  = 149
    def df_in_patch(df_ra,df_dec,):
        in_patch = ((PATCH_RA  < df_ra)  & (df_ra  < PATCH_RA+PATCH_SIZE) &
                    (PATCH_DEC < df_dec) & (df_dec < PATCH_DEC+PATCH_SIZE))
        return in_patch

    # import TGSS
    tgss_df = pd.read_csv('TGSSADR1_7sigma_catalog.tsv',delimiter='\t',
                          index_col=0,usecols=(0,1,3,5,7,9,11,13))
    tgss_df = tgss_df.sort_values(by=['DEC'])

    tgss_df['Total_flux'] = tgss_df['Total_flux']*1e-3
    tgss_df['Peak_flux']  = tgss_df['Peak_flux']*1e-3

    tgss_df = tgss_df[df_in_patch(tgss_df['RA'],tgss_df['DEC'])]

    tgss_df.index.names = ['name_TGSS']
    tgss_df.columns = ['ra_TGSS','dec_TGSS','integrated_TGSS','peak_TGSS',
                       'major_ax_TGSS','minor_ax_TGSS','posangle_TGSS']

    tgss_df.to_csv('tgss.csv')

    # import NVSS
    with fits.open('CATALOG.FIT') as hdulist:
        data = hdulist[1].data
        nvss_data = np.column_stack((data['RA(2000)'],data['DEC(2000)'],data['PEAK INT'],
                                data['MAJOR AX'],data['MINOR AX'],data['POSANGLE'],
                                data['Q CENTER'],data['U CENTER'],data['P FLUX'],
                                data['RES PEAK'],data['RES FLUX']))
        nvss_columns = ['RA(2000)','DEC(2000)','PEAK INT','MAJOR AX','MINOR AX','POSANGLE',
                        'Q CENTER','U CENTER','P FLUX','RES PEAK','RES FLUX']
        nvss_df = pd.DataFrame(data = nvss_data, columns = nvss_columns)
        nvss_df = nvss_df.sort_values(by=['DEC(2000)']).reset_index(drop = True)

        nvss_df = nvss_df[df_in_patch(nvss_df['RA(2000)'],nvss_df['DEC(2000)'])]

    nvss_labels = np.array([iau_designation(p[0],p[1]) for p in
                            nvss_df[['RA(2000)','DEC(2000)']].values])
    nvss_df['name_NVSS'] = nvss_labels
    nvss_df.set_index('name_NVSS',inplace=True)

    nvss_df.columns = ['ra_NVSS','dec_NVSS','peak_NVSS','major_ax_NVSS','minor_ax_NVSS','posangle_NVSS',
                      'q_centre_NVSS','u_centre_NVSS','polarised_NVSS','res_peak_NVSS','res_flux_NVSS']

    nvss_df.to_csv('nvss.csv')

    # positional matching, a la positional_catalogue.py
    tgss = tgss_df[['ra_TGSS','dec_TGSS']].values
    nvss = nvss_df[['ra_NVSS','dec_NVSS']].values

    nvss_dec_min = round(nvss[:,1].min(),1)
    nvss_dec_max = round(nvss[:,1].max(),1)

    patch_matches = []
    tqdmbar = tqdm(total=len(tgss))
    for i1,p1 in enumerate(tgss):
        for i2,p2 in enumerate(nvss):
            if (abs((p1[0]-p2[0])*np.cos(p1[1]*np.pi/180)) < SEPARATION_LIMIT
                    and abs(p1[1]-p2[1]) < SEPARATION_LIMIT):
                patch_matches.append((i1,i2))
        tqdmbar.postfix = 'matches = {}'.format(len(patch_matches))
        tqdmbar.update(1)
    patch_matches = np.array(patch_matches)

    tmp_patch_matches = []
    for i1,i2 in tqdm(patch_matches):
        p1,p2 = tgss[i1],nvss[i2]
        d = degdist(p1,p2)
        if d < SEPARATION_LIMIT:
            tmp_patch_matches.append([i1,i2])
    patch_matches = np.array(tmp_patch_matches)

    patch_cat_columns = np.concatenate((tgss_df.reset_index().columns.values,
                                        nvss_df.reset_index().columns.values))
    patch_cat = pd.DataFrame(columns=patch_cat_columns)

    FREQ_TGSS,FREQ_NVSS = 150e6,1.4e9

    for i1,i2 in tqdm(patch_matches):
        obj_t = tgss_df.reset_index().iloc[i1]
        obj_n = nvss_df.reset_index().iloc[i2]
        match_row = {**obj_t,**obj_n}

        sepa = degdist((obj_t['ra_TGSS'],obj_t['dec_TGSS']),
                       (obj_n['ra_NVSS'],obj_n['dec_NVSS']))
        match_row['separation'] = sepa

        alpha = np.log(obj_t['peak_TGSS']/obj_n['peak_NVSS'])/np.log(FREQ_NVSS/FREQ_TGSS)
        match_row['spectral_alpha'] = alpha

        patch_cat = patch_cat.append(match_row, ignore_index=True)

    patch_cat.set_index(['name_TGSS','name_NVSS'], inplace=True)

    patch_cat.to_csv('patch_catalogue.csv')

if __name__ == '__main__':
    main()
