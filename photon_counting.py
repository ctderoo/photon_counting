from numpy import *
import matplotlib.pyplot as plt
import os
import scipy.ndimage.measurements as img_meas
import astropy.io.fits as pyfits

import time
import pdb
################################
# Utility functions.

def group(lst,n):
    return zip(*[lst[i::n] for i in range(n)])

def load_CCD_images(filename,dark_file):
    all_img_data = pyfits.getdata(filename)
    dark_frame = pyfits.getdata(dark_file)
    return all_img_data,dark_frame

################################
# Functions for performing the photon counting on actual data frames.

def photon_count(img,dark_frame,sigma_thres = 10):
    # Construct the master dark frame and the variance dark frame, which will be used to
    # threshold events.
    dark_master,dark_var = median(dark_frame,axis = 0),std(dark_frame,axis = 0)
    # Calculate where the image has cleared the threshold. For a 2k X 2k image,
    # 5*sigma means that only 2 pixels should clear the threshold (assuming a Gaussian distributed background).
    binary_img = (img - dark_master - sigma_thres*dark_var > 0)
    # Counting the total number of events, and identifying individual events
    counted_img, N = img_meas.label(binary_img*1)
    CoMx,CoMy,ADCs,event_size = zeros(N),zeros(N),zeros(N),zeros(N)
    for i in range(N):
        # Constructing the binary mask which allows only certain pixels to be summed
        # for a given photon event. The +1 is needed to avoid an OBO error with the labeling
        # (starts at 1) and range(N) (starts at 0).
        mask = (counted_img == i + 1)*1
        CoMy[i],CoMx[i] = img_meas.center_of_mass((img - dark_master)*mask)
        ADCs[i] = sum((img - dark_master)*mask)
        event_size[i] = sum(mask)
    return CoMx,CoMy,ADCs,event_size

def make_all_data_products(filenames,dark_file,data_directory = '/Data',list_directory = '/List',fits_directory = '/FITS Files'):
    for filename in filenames:
        photon_x = array([])
        photon_y = array([])
        photon_adc = array([])
        photon_size = array([])
    
        os.chdir(data_directory)
        all_img_data,dark_frame = load_CCD_images(filename,dark_file)
        for i in range(shape(all_img_data)[0]):
            print 'Working on ' + str(filename) + ', frame number ' + str(i) + '...'
            tempx,tempy,tempadc,tempsize = photon_count(all_img_data[i],dark_frame,sigma_thres = 10)
            photon_x = hstack((photon_x,tempx))
            photon_y = hstack((photon_y,tempy))
            photon_adc = hstack((photon_adc,tempadc))
            photon_size = hstack((photon_size,tempsize))
    
        os.chdir(list_directory)
        savetxt(filename.replace('.fits','_PhotonList.txt'),vstack((photon_x,photon_y,photon_adc,photon_size)))
        
        os.chdir(fits_directory)
        ccd = make_photon_image(photon_x,photon_y)
        hdu = pyfits.PrimaryHDU(ccd)
        hdu.writeto(filename.replace('.fits','_PhotonCounted.fits'))

###################################################
# Data 'cleaning' functions.
def scrub_cosmics(x,y,adc,spread):
    spread_rej = where(spread < 9)
    x_scrub,y_scrub,adc_scrub,spread_scrub = x[spread_rej],y[spread_rej],adc[spread_rej],spread[spread_rej]
    adc_rej = where(adc < 300)
    x_scrub,y_scrub,adc_scrub,spread_scrub = x[adc_rej],y[adc_rej],adc[adc_rej],spread[adc_rej]
    return x_scrub,y_scrub,adc_scrub,spread_scrub

def scrub_thermal(x,y,adc,spread,photon_thres = 50):
    thres_rej = where(adc > 50)
    x_scrub,y_scrub,adc_scrub,spread_scrub = x[thres_rej],y[thres_rej],adc[thres_rej],spread[thres_rej]
    return x_scrub,y_scrub,adc_scrub,spread_scrub
    
def scrub_position(x,y,adc,spread,x_window = (0,2048),y_window = (0,2048)):
    ind = where((x > x_window[0]) & (x < x_window[1]) & (y > y_window[0]) & (y < y_window[1]))
    return x[ind],y[ind],adc[ind],spread[ind]

def load_valid_events(photon_list):
    tempx,tempy,tempadc,tempspread = loadtxt(photon_list)
    tempx,tempy,tempadc,tempspread = pcounts.scrub_cosmics(tempx,tempy,tempadc,tempspread)
    x,y,adc,spread = pcounts.scrub_thermal(tempx,tempy,tempadc,tempspread)
    return x,y,adc,spread

###################################################
# Outputting and computing data products.

def make_photon_image(x,y,max_x_pix = 2048,max_y_pix = 2048):
    ccd = zeros((max_y_pix,max_x_pix))
    ind1,ind2 = around(y).astype('int'),around(x).astype('int')
    for i in range(len(ind1)):
        ccd[ind1[i],ind2[i]] = ccd[ind1[i],ind2[i]] + 1
    return ccd

def centroid_calculation(filename,CCD_V,CCD_H,y_window = (0,2048),x_window = (0,2048)):
    '''
    CCD_V and CCD_H are given in terms of centimeters, so the pixel size = 0.00135 cm (13.5 micron).
    '''
    x_nat,y_nat,adc_nat,spread_nat = loadtxt(filename)
    x,y,adc,spread = scrub_cosmics(x_nat,y_nat,adc_nat,spread_nat)
    x,y,adc,spread = scrub_thermal(x,y,adc,spread)
    x,y,adc,spread = scrub_position(x,y,adc,spread,x_window = x_window,y_window = y_window)
    
    # Converting to real physical positions: we take the CCD_H position (which is the cross-dispersion direction,
    # or the direction of the OPG reflection), and add the photon position (2048 - the photon positon, since the maximum pixel
    # on the CCD is closest to the SPO focus).
    real_x = CCD_H + (2048 - x)*0.00135
    real_y = CCD_V + (2048 - y)*0.00135
    return real_x,real_y


