#!python
#cython: boundscheck=False
#cython: wraparound=False

#from libcpp.string cimport string
#from libc.stdint cimport uint32_t, int64_t, uint64_t
#from libcpp cimport bool
import  numpy as np
cimport numpy as np

np.import_array()


cdef extern from "global_def.h":

    cdef struct Data:

        # image info
        int rows
        int cols
        int channels
        int size
        double *Image
        double *MImage
    
        # data domain info
        double *Domain
        double *MDomain
    
        # time info
#        hItem **heap
#        hItem *Tfield
        double *ordered_points
        int nof_points2inpaint
    
        # parameters
        int radius
        double epsilon
        double kappa
        double sigma
        double rho
        double thresh
        double delta_quant4
        double *convex
    
        # smoothing kernels and buffer
        int lenSK1
        int lenSK2
        double *SKernel1
        double *SKernel2
        double *Shelp
    
        # inpaint buffer
        double *Ihelp
    
        # flags
        int ordergiven
        int guidance
        int inpaint_undefined
    
        # extension
        double *GivenGuidanceT

    void InpaintImage(Data *data)
    void SetKernels(Data *data)
    void FreeKernels(Data *data)
    void SetDefaults(Data *data)
    void AllocateData(Data *data)
    void ClearMemory(Data *data);
    int GetMask(double *arg, int M, int N, Data *data)
    int GetOrder(double *arg, int M, int N, Data *data)


def inpaintBCT(np.ndarray[np.float64_t, ndim=3, mode='fortran'] img,
               np.ndarray[np.float64_t, ndim=2, mode='fortran'] mask,
               double epsilon, double kappa, double sigma, double rho, double threshold):

    cdef Data data

    SetDefaults(&data)

    data.epsilon = epsilon
    data.kappa = kappa
    data.sigma = sigma
    data.rho = rho
    data.thresh = threshold

    data.radius = (int) (data.epsilon + 0.5)

    cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] Im = img.copy('F')
    cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] out_img = img.copy('F')

    data.rows = img.shape[0]
    data.cols = img.shape[1]
    data.size = data.rows * data.cols;
    if img.ndim == 2:
        data.channels = 1;
    else:
        data.channels = img.shape[2]

#    data.Image = <double*> out_img.data
#    data.MImage = <double*> Im.data
    data.Image = &out_img[0, 0, 0]
    data.MImage = &Im[0, 0, 0]

    data.ordergiven = 0
    data.guidance = 1

    AllocateData(&data)
    if GetMask(&mask[0, 0], mask.shape[0], mask.shape[1], &data):
        print "some getmask error occured"

    SetKernels(&data)

    InpaintImage(&data)

    ClearMemory(&data)

    return out_img


    
